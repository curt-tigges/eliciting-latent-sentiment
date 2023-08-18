from enum import Enum
from functools import partial
import einops
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import Callable, Union, List, Tuple
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
import wandb
from utils.circuit_analysis import get_logit_diff, logit_diff_denoising
from utils.prompts import PromptType, get_dataset
from utils.residual_stream import get_resid_name


class FittingMethod(Enum):
    DAS = 'das'


class InverseRotateLayer(torch.nn.Module):
    """The inverse of a given `LinearLayer` module."""
    def __init__(self, lin_layer):
        super().__init__()
        self.lin_layer = lin_layer

    def forward(self, x):
        output = torch.matmul(x, self.lin_layer.weight.T)
        return output


class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""
    def __init__(self, n, device: torch.device, init_orth: bool = True):
        super().__init__()
        weight = torch.empty(n,n, device=device)
        # we don't need init if the saved checkpoint has a nice 
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(
            weight, requires_grad=True,
        ).to(device)
        
    def forward(self, x):
        return torch.matmul(x, self.weight)


def hook_fn_base(
    input: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    layer: int,
    position: int,
    new_value: Float[Tensor, "d_model"],
):
    assert 'resid' in hook.name
    if hook.layer() != layer:
        return input
    input[:, position] = new_value
    return input
    

def act_patch_simple(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_value: Float[Tensor, "d_model"],
    layer: int,
    position: int,
    patching_metric: Callable,
) -> Float[Tensor, ""]:
    model.reset_hooks()
    hook_fn = partial(hook_fn_base, layer=layer, position=position, new_value=new_value)
    act_name = get_resid_name(layer, model)
    logits = model.run_with_hooks(
        orig_input,
        fwd_hooks=[(act_name, hook_fn)],
    )
    assert logits.requires_grad or not model.training, (
        "logits should require grad, otherwise we can't backpropagate through them. "
        f"Layer: {layer}, position: {position}, new_value: {new_value.requires_grad}"
    )
    return patching_metric(logits)


class RotationModule(torch.nn.Module):
    def __init__(
        self, 
        model: HookedTransformer,
        orig_tokens: Int[Tensor, "batch pos"],
        patching_metric: Callable,
        n_directions: int = 1,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.cfg.device
        self.register_buffer('orig_tokens', orig_tokens)
        self.d_model = model.cfg.d_model
        self.n_directions = n_directions
        rotate_layer = RotateLayer(model.cfg.d_model, self.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, use_trivialization=False
        )
        self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)
        self.patching_metric = patching_metric

    def apply_rotation(
        self,
        orig_resid: Float[Tensor, "batch d_model"],
        new_resid: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch d_model"]:
        rotated_orig_act: Float[
            Tensor, "batch d_model"
        ] = self.rotate_layer(orig_resid)
        rotated_new_act: Float[
            Tensor, "batch d_model"
        ] = self.rotate_layer(new_resid)
        d_model_index = einops.repeat(
            torch.arange(self.model.cfg.d_model, device=self.device),
            "d -> batch d",
            batch=orig_resid.shape[0],
        )
        rotated_patch_act = torch.where(
            d_model_index < self.n_directions,
            rotated_new_act,
            rotated_orig_act,
        )
        patch_act = self.inverse_rotate_layer(rotated_patch_act)
        return patch_act

    def forward(
        self, 
        orig_resid: Float[Tensor, "batch pos d_model"],
        new_resid: Float[Tensor, "batch pos d_model"],
        layer: int,
        position: int,
    ) -> Float[Tensor, ""]:
        patched_resid: Float[Tensor, "batch d_model"] = self.apply_rotation(
            orig_resid=orig_resid,
            new_resid=new_resid,
        )
        metric: Float[Tensor, ""] = act_patch_simple(
            model=self.model,
            orig_input=self.orig_tokens,
            new_value=patched_resid,
            patching_metric=self.patching_metric,
            layer=layer,
            position=position,
        )
        return metric


class TrainingConfig:

    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def fit_rotation(
    orig_tokens: Int[Tensor, "batch pos"],
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    patching_metric: Callable,
    model: HookedTransformer, 
    **given_config_dict
) -> Tuple[HookedTransformer, List[Tensor]]:
    # Initialize wandb
    config_dict = dict(
        seed=0,
        lr=1e-3,
        epochs=50,
        n_directions=1,
        wandb_enabled=True,
    )
    config_dict.update(given_config_dict)
    if config_dict["wandb_enabled"]:
        wandb.init(project='train_rotation', config=config_dict)
        config = wandb.config
    else:
        config = TrainingConfig(config_dict)

    train_act_name = get_resid_name(config.train_layer, model)
    eval_act_name = get_resid_name(config.eval_layer, model)

    # Create a list to store the losses and models
    losses = []
    models = [] 
    step = 0
    torch.manual_seed(config.seed)
        
    # Create the rotation module
    rotation_module = RotationModule(
        model=model,
        orig_tokens=orig_tokens,
        patching_metric=patching_metric,
        n_directions=config.n_directions,
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(rotation_module.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        rotation_module.train()
        optimizer.zero_grad()
        loss = rotation_module(
            orig_cache[train_act_name][:, config.train_position, :],
            new_cache[train_act_name][:, config.train_position, :],
            layer=config.train_layer,
            position=config.train_position,
        )
        loss.backward()
        optimizer.step()
        rotation_module.eval()
        with torch.inference_mode():
            eval_loss = rotation_module(
                orig_cache[eval_act_name][:, config.eval_position, :],
                new_cache[eval_act_name][:, config.eval_position, :],
                layer=config.eval_layer,
                position=config.eval_position,
            )
        if config.wandb_enabled:
            wandb.log({"training_loss": loss.item(), "validation_loss": eval_loss.item()}, step=step)

        # Store the loss and model for this seed
        losses.append(loss.item())
        models.append(rotation_module.state_dict())
        step += 1
    direction = rotation_module.rotate_layer.weight[0, :]

    best_model_idx = min(range(len(losses)), key=losses.__getitem__)

    # Log the best model's loss and save the model
    if config.wandb_enabled:
        wandb.log({"Best Loss": losses[best_model_idx]})
        wandb.save("best_rotation.pt")
        wandb.finish()

    # Load the best model
    best_model_state_dict = models[best_model_idx]
    rotation_module.load_state_dict(best_model_state_dict)
    return direction


def get_das_dataset(
    prompt_type: PromptType, layer: int, model: HookedTransformer, device: torch.device,
):
    all_prompts, answer_tokens, new_tokens, orig_tokens = get_dataset(model, device, prompt_type=prompt_type)
    name_filter = lambda name: name in ('blocks.0.attn.hook_z', get_resid_name(layer, model))
    orig_logits, orig_cache = model.run_with_cache(
        orig_tokens, names_filter=name_filter
    )
    new_logits, new_cache = model.run_with_cache(
        new_tokens, names_filter=name_filter
    )
    orig_cache.to(device)
    new_cache.to(device)
    orig_logit_diff = get_logit_diff(orig_logits, answer_tokens)
    new_logit_diff = get_logit_diff(new_logits, answer_tokens)
    loss_fn = partial(
        logit_diff_denoising, 
        answer_tokens=answer_tokens, 
        flipped_logit_diff=new_logit_diff, 
        clean_logit_diff=orig_logit_diff
    )
    return all_prompts, orig_tokens, orig_cache, new_cache, loss_fn


def train_das_direction(
    model: HookedTransformer, device: torch.device, 
    train_type: PromptType, train_pos: str, train_layer: int,
    test_type: PromptType, test_pos: str, test_layer: int,
    **config_arg,
):
    assert train_type == test_type, "train and test prompts must be the same"
    assert train_layer == test_layer, "train and test layers must be the same"
    all_prompts, orig_tokens, orig_cache, new_cache, loss_fn = get_das_dataset(
        train_type, layer=train_layer, model=model, device=device,
    )
    example = model.to_str_tokens(all_prompts[0])
    placeholders = train_type.get_placeholder_positions(example)
    config = dict(
        train_layer=train_layer,
        train_position=placeholders[train_pos][-1],
        eval_layer=test_layer,
        eval_position=placeholders[test_pos][-1],
    )
    config.update(config_arg)
    return fit_rotation(
        orig_tokens=orig_tokens,
        orig_cache=orig_cache,
        new_cache=new_cache,
        patching_metric=loss_fn,
        model=model,
        **config,
    )