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
import wandb
from utils.circuit_analysis import get_logit_diff, logit_diff_denoising
from utils.prompts import PromptType, get_dataset
from utils.residual_stream import get_resid_name
from utils.store import save_array


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
        
    def forward(self, x: Float[Tensor, "batch d_model"]):
        return torch.matmul(x, self.weight)


def hook_fn_base(
    resid: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    layer: int,
    position: Union[None, int],
    new_value: Float[Tensor, "batch *pos d_model"]
):
    batch_size, seq_len, d_model = resid.shape
    assert 'resid' in hook.name
    if hook.layer() != layer:
        return resid
    if position is None:
        assert new_value.shape == resid.shape
        return new_value
    new_value_repeat = einops.repeat(
        new_value,
        "batch d_model -> batch pos d_model",
        pos=seq_len,
    )
    position_index = einops.repeat(
        torch.arange(seq_len, device=resid.device),
        "pos -> batch pos d_model",
        batch=batch_size,
        d_model=d_model,
    )
    return torch.where(
        position_index == position,
        new_value_repeat,
        resid,
    )
    

def act_patch_simple(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_value: Float[Tensor, "d_model"],
    layer: int,
    position: Union[None, int],
    patching_metric: Callable,
) -> Float[Tensor, ""]:
    assert layer <= model.cfg.n_layers
    act_name, patch_layer = get_resid_name(layer, model)
    model.reset_hooks()
    hook_fn = partial(hook_fn_base, layer=patch_layer, position=position, new_value=new_value)
    logits = model.run_with_hooks(
        orig_input,
        fwd_hooks=[(act_name, hook_fn)],
    )
    return patching_metric(logits)


class RotationModule(torch.nn.Module):
    def __init__(
        self, 
        model: HookedTransformer,
        d_das: int = 1,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.cfg.device
        self.d_model = model.cfg.d_model
        self.d_das = d_das
        rotate_layer = RotateLayer(model.cfg.d_model, self.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, use_trivialization=False
        )
        self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)

    def apply_rotation(
        self,
        orig_resid: Float[Tensor, "batch *pos d_model"],
        new_resid: Float[Tensor, "batch *pos d_model"],
    ) -> Float[Tensor, "batch d_model"]:
        rotated_orig_act: Float[
            Tensor, "batch *pos d_model"
        ] = self.rotate_layer(orig_resid)
        rotated_new_act: Float[
            Tensor, "batch *pos d_model"
        ] = self.rotate_layer(new_resid)
        d_model_index: Float[
            Tensor, "batch *pos d_model"
        ] = torch.arange(
            self.model.cfg.d_model, device=self.device
        ).expand(orig_resid.shape)
        rotated_patch_act: Float[
            Tensor, "batch *pos d_model"
        ] = torch.where(
            d_model_index < self.d_das,
            rotated_new_act,
            rotated_orig_act,
        )
        patch_act = self.inverse_rotate_layer(rotated_patch_act)
        return patch_act

    def forward(
        self, 
        orig_resid: Float[Tensor, "batch *pos d_model"],
        new_resid: Float[Tensor, "batch *pos d_model"],
        layer: int,
        position: Union[None, int],
        orig_tokens: Int[Tensor, "batch pos"],
        patching_metric: Callable,
    ) -> Float[Tensor, ""]:
        patched_resid: Float[Tensor, "batch d_model"] = self.apply_rotation(
            orig_resid=orig_resid,
            new_resid=new_resid,
        )
        metric: Float[Tensor, ""] = act_patch_simple(
            model=self.model,
            orig_input=orig_tokens,
            new_value=patched_resid,
            patching_metric=patching_metric,
            layer=layer,
            position=position,
        )
        return metric


class TrainingConfig:
    train_layer: int
    train_position: Union[None, int]
    eval_layer: int
    eval_position: Union[None, int]

    def __init__(self, config_dict: dict):
        self.seed = config_dict.get("seed", 0)
        self.lr = config_dict.get("lr", 1e-3)
        self.weight_decay = config_dict.get("weight_decay", 0)
        self.betas = config_dict.get("betas", (0.9, 0.999))
        self.epochs = config_dict.get("epochs", 64)
        self.d_das = config_dict.get("d_das", 1)
        self.wandb_enabled = config_dict.get("wandb_enabled", True)
        self.model_name = config_dict.get("model_name", "unnamed-model")
        self.clip_grad_norm = config_dict.get("clip_grad_norm", 1.0)
        for k, v in config_dict.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            attr: getattr(self, attr) 
            for attr in dir(self) 
            if not callable(getattr(self, attr)) and not attr.startswith('_')
        }


def fit_rotation(
    orig_tokens_train: Int[Tensor, "batch pos"],
    orig_cache_train: ActivationCache,
    new_cache_train: ActivationCache,
    orig_tokens_test: Int[Tensor, "batch pos"],
    orig_cache_test: ActivationCache,
    new_cache_test: ActivationCache,
    metric_train: Callable,
    metric_test: Callable,
    model: HookedTransformer, 
    project: str = None,
    **config_dict
) -> Tuple[HookedTransformer, List[Tensor]]:
    """
    Entrypoint for training a DAS subspace given
    a counterfactual patching dataset.
    """
    if 'model_name' not in config_dict:
        config_dict['model_name'] = model.cfg.model_name
    config = TrainingConfig(config_dict)
    # Initialize wandb
    if config.wandb_enabled:
        wandb.init(config=config.to_dict(), project=project)
        config = wandb.config

    train_act_name, _ = get_resid_name(config.train_layer, model)
    eval_act_name, _ = get_resid_name(config.eval_layer, model)

    orig_resid_train_base = orig_cache_train[train_act_name]
    new_resid_train_base = new_cache_train[train_act_name]
    if config.train_position is not None:
        orig_resid_train_base = orig_resid_train_base[:, config.train_position, :]
        new_resid_train_base = new_resid_train_base[:, config.train_position, :]
    orig_resid_test_base = orig_cache_test[eval_act_name]
    new_resid_test_base = new_cache_test[eval_act_name]
    if config.eval_position is not None:
        orig_resid_test_base = orig_resid_test_base[:, config.eval_position, :]
        new_resid_test_base = new_resid_test_base[:, config.eval_position, :]

    # Create a list to store the losses and models
    losses_train = []
    losses_test = []
    models = [] 
    step = 0
    torch.manual_seed(config.seed)
        
    # Create the rotation module
    rotation_module = RotationModule(
        model=model,
        d_das=config.d_das,
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(
        rotation_module.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    for epoch in range(config.epochs):
        rotation_module.train()
        optimizer.zero_grad()
        loss = rotation_module(
            orig_resid_train_base.clone().requires_grad_(True),
            new_resid_train_base.clone().requires_grad_(True),
            layer=config.train_layer,
            position=config.train_position,
            orig_tokens=orig_tokens_train,
            patching_metric=metric_train,
        )
        assert loss.requires_grad, (
            "The loss must be a scalar that requires grad. \n"
            f"loss: {loss}, loss.requires_grad: {loss.requires_grad}, "
            f"train layer: {config.train_layer}, train position: {config.train_position}, "
            f"train act name: {train_act_name}, "
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rotation_module.parameters(), config.clip_grad_norm)
        optimizer.step()
        rotation_module.eval()
        with torch.inference_mode():
            eval_loss = rotation_module(
                orig_resid_test_base,
                new_resid_test_base,
                layer=config.eval_layer,
                position=config.eval_position,
                orig_tokens=orig_tokens_test,
                patching_metric=metric_test,
            )
        if config.wandb_enabled:
            wandb.log({"training_loss": loss.item(), "validation_loss": eval_loss.item()}, step=step)

        # Store the loss and model for this seed
        losses_train.append(loss.item())
        losses_test.append(eval_loss.item())
        models.append(rotation_module.state_dict())
        step += 1

    best_model_idx = min(range(len(losses_train)), key=losses_train.__getitem__)

    # Log the best model's loss and save the model
    if config.wandb_enabled:
        wandb.log({"best_training_loss": losses_train[best_model_idx]})
        wandb.log({"best_validation_loss": losses_test[best_model_idx]})
        wandb.save("best_rotation.pt")
        wandb.finish()

    # Load the best model
    best_model_state_dict = models[best_model_idx]
    rotation_module.load_state_dict(best_model_state_dict)
    return rotation_module.rotate_layer.weight[:, :config.d_das]


def get_das_dataset(
    prompt_type: PromptType, layer: int, model: HookedTransformer, device: torch.device,
):
    """
    Wrapper for utils.prompts.get_dataset that returns a dataset in a useful form for DAS
    """
    clean_corrupt_data = get_dataset(model, device, prompt_type=prompt_type)
    new_tokens = clean_corrupt_data.clean_tokens
    orig_tokens = clean_corrupt_data.corrupted_tokens
    name_filter = lambda name: name in ('blocks.0.attn.hook_z', get_resid_name(layer, model)[0])
    with torch.inference_mode():
        orig_logits, orig_cache = model.run_with_cache(
            orig_tokens, names_filter=name_filter
        )
        new_logits, new_cache = model.run_with_cache(
            new_tokens, names_filter=name_filter
        )
    orig_cache.to(device)
    new_cache.to(device)
    orig_logit_diff = get_logit_diff(orig_logits, clean_corrupt_data.answer_tokens)
    new_logit_diff = get_logit_diff(new_logits, clean_corrupt_data.answer_tokens)
    loss_fn = partial(
        logit_diff_denoising, 
        answer_tokens=clean_corrupt_data.answer_tokens, 
        flipped_value=new_logit_diff, 
        clean_value=orig_logit_diff,
        return_tensor=True,
    )
    return clean_corrupt_data.all_prompts, orig_tokens, orig_cache, new_cache, loss_fn


def train_das_subspace(
    model: HookedTransformer, device: torch.device, 
    train_type: PromptType, train_pos: Union[None, str], train_layer: int,
    test_type: PromptType, test_pos: Union[None, str], test_layer: int,
    **config_arg,
):
    """
    Entrypoint to be used in directional patching experiments
    Given training/validation datasets, train a DAS subspace
    """
    all_prompts, orig_tokens, orig_cache, new_cache, loss_fn = get_das_dataset(
        train_type, layer=train_layer, model=model, device=device,
    )
    all_prompts_val, orig_tokens_val, orig_cache_val, new_cache_val, loss_fn_val = get_das_dataset(
        test_type, layer=test_layer, model=model, device=device,
    )
    example_train = model.to_str_tokens(all_prompts[0])
    placeholders_train = train_type.get_placeholder_positions(example_train)
    example_val = model.to_str_tokens(all_prompts_val[0])
    placeholders_val = test_type.get_placeholder_positions(example_val)
    config = dict(
        train_layer=train_layer,
        train_position=placeholders_train[train_pos][-1] if train_pos is not None else None,
        eval_layer=test_layer,
        eval_position=placeholders_val[test_pos][-1] if test_pos is not None else None,
    )
    config.update(config_arg)
    directions = fit_rotation(
        orig_tokens_train=orig_tokens,
        orig_cache_train=orig_cache,
        new_cache_train=new_cache,
        orig_tokens_test=orig_tokens_val,
        orig_cache_test=orig_cache_val,
        new_cache_test=new_cache_val,
        metric_train=loss_fn,
        metric_test=loss_fn_val,
        model=model,
        **config,
    )
    save_array(
        directions.detach().cpu().numpy(), 
        f'das_{train_type.value}_{train_pos}_layer{train_layer}', 
        model,
    )
    return directions