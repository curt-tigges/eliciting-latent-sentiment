from contextlib import nullcontext
from functools import partial
import warnings
import einops
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity
from jaxtyping import Float, Int, Bool
from typing import Callable, Optional, Union, List, Tuple
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, ActivationCache
import wandb
from tqdm.auto import tqdm
from utils.circuit_analysis import get_logit_diff, logit_diff_denoising
from utils.prompts import PromptType, get_dataset
from utils.residual_stream import get_resid_name
from utils.store import save_array
from utils.treebank import ReviewScaffold
from utils.methods import FittingMethod


class GradientMethod(FittingMethod):
    DAS = 'das'
    DAS2D = 'das2d'
    DAS3D = 'das3d'

    def get_dimension(self):
        if self == GradientMethod.DAS:
            return 1
        elif self == GradientMethod.DAS2D:
            return 2
        elif self == GradientMethod.DAS3D:
            return 3
        else:
            raise ValueError("Invalid GradientMethod")


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
    position: Optional[Union[int, Int[Tensor, "batch"]]],
    new_value: Float[Tensor, "batch *pos d_model"]
):
    batch_size, seq_len, d_model = resid.shape
    assert hook.name is not None and 'resid' in hook.name
    if hook.layer() != layer:
        return resid
    if position is None:
        assert new_value.shape == resid.shape
        return new_value
    if isinstance(position, int):
        position = torch.tensor(position, device=resid.device)
        position = einops.repeat(
            position,
            " -> batch",
            batch=batch_size,
        )
    new_value_repeat = einops.repeat(
        new_value,
        "batch d_model -> batch pos d_model",
        pos=seq_len,
    )
    seq_len_rep = einops.repeat(
        torch.arange(seq_len, device=resid.device),
        "pos -> batch pos",
        batch=batch_size,
    )
    position_rep = einops.repeat(
        position,
        "batch -> batch pos",
        pos=seq_len,
    )
    position_mask = einops.repeat(
        position_rep == seq_len_rep,
        "batch pos -> batch pos d_model",
        d_model=d_model,
    )
    out = torch.where(
        position_mask,
        new_value_repeat,
        resid,
    )
    if new_value.requires_grad:
        assert out.requires_grad
    return out
    

def act_patch_simple(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_value: Float[Tensor, "d_model"],
    layer: int,
    patching_metric: Callable,
    position: Optional[Int[Tensor, "batch"]] = None,
    verbose: bool = False,
) -> Float[Tensor, ""]:
    assert layer <= model.cfg.n_layers
    act_name, patch_layer = get_resid_name(layer, model)
    model.reset_hooks()
    hook_fn = partial(hook_fn_base, layer=patch_layer, position=position, new_value=new_value)
    logits = model.run_with_hooks(
        orig_input,
        fwd_hooks=[(act_name, hook_fn)],
    )
    metric = patching_metric(logits)
    if verbose:
        print(f"logits.shape: {logits.shape}, metric: {metric}")
    if new_value.requires_grad:
        assert logits.requires_grad, (
            "Output of run_with_hooks should require grad. "
            "Are you sure that the hook was applied? "
            f"act_name={act_name}, patch_layer={patch_layer}" 
        )
        assert metric.requires_grad
    return metric


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
        orig_tokens: Int[Tensor, "batch pos"],
        patching_metric: Callable,
        position: Optional[Int[Tensor, "batch"]],
        check_requires_grad: bool = False,
        verbose: bool = False,
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
            verbose=verbose,
        )
        if check_requires_grad:
            assert patched_resid.requires_grad
            assert metric.requires_grad
        return metric


class TrainingConfig:
    train_layer: int
    eval_layer: int

    def __init__(self, config_dict: dict):
        self.batch_size = config_dict.get("batch_size", 32)
        self.seed = config_dict.get("seed", 0)
        self.lr = config_dict.get("lr", 1e-3)
        self.weight_decay = config_dict.get("weight_decay", 0)
        self.betas = config_dict.get("betas", (0.9, 0.999))
        self.epochs = config_dict.get("epochs", 1)
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
    trainloader: DataLoader,
    testloader: DataLoader,
    metric_train: Callable,
    metric_test: Callable,
    model: HookedTransformer, 
    device: torch.device,
    project: Optional[str] = None,
    profiler: bool = False,
    downcast: bool = False,
    verbose: bool = False,
    position_train: Optional[Int[Tensor, "batch"]] = None,
    position_test: Optional[Int[Tensor, "batch"]] = None,
    **config_dict
) -> Float[Tensor, "d_model d_das"]:
    """
    Entrypoint for training a DAS subspace given
    a counterfactual patching dataset.
    """
    torch.cuda.empty_cache()
    loss_context = autocast() if downcast else nullcontext()
    scaler = GradScaler() if downcast else None
    if device != model.cfg.device:
        model = model.to(device)
    if 'model_name' not in config_dict:
        config_dict['model_name'] = model.cfg.model_name
    config = TrainingConfig(config_dict)
    # Initialize wandb
    if config.wandb_enabled:
        wandb.init(config=config.to_dict(), project=project)
        config = wandb.config

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
    epoch_bar = tqdm(range(config.epochs), disable=config.epochs == 1)
    for epoch in epoch_bar:
        epoch_train_loss = 0
        epoch_test_loss = 0
        rotation_module.train()
        train_bar = tqdm(trainloader, disable=config.epochs > 1)
        train_bar.set_description(
            f"Epoch {epoch}: training. Batch size: {trainloader.batch_size}. Device: {device}"
        )
        for orig_tokens_train, orig_resid_train, new_resid_train, answers_train in train_bar:
            train_bar.set_description(
                f"Epoch {epoch} training: moving data to device={device}"
            )
            orig_tokens_train = orig_tokens_train.to(device)
            orig_resid_train = orig_resid_train.to(device)
            new_resid_train = new_resid_train.to(device)
            answers_train = answers_train.to(device)
            optimizer.zero_grad()
            train_bar.set_description(
                f"Epoch {epoch} training: computing loss"
            )
            with loss_context:
                loss = rotation_module(
                    orig_resid_train,
                    new_resid_train,
                    layer=config.train_layer,
                    position=position_train,
                    orig_tokens=orig_tokens_train,
                    patching_metric=partial(metric_train, answer_tokens=answers_train),
                    check_requires_grad=True,
                    verbose=verbose,
                )
            if downcast:
                scaled_loss = scaler.scale(loss)
            else:
                scaled_loss = loss
            assert scaled_loss.requires_grad, (
                "The loss must be a scalar that requires grad. \n"
                f"loss: {scaled_loss}, loss.requires_grad: {scaled_loss.requires_grad}, "
                f"train layer: {config.train_layer}, train position: {config.train_position} "
            )
            train_bar.set_description(
                f"Epoch {epoch} training: backpropagating. Profiler: {profiler}. Device: {device}. Downcast: {downcast}"
            )
            if profiler:
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                    with record_function("backpropagation"):
                        scaled_loss.backward()
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            else:
                scaled_loss.backward()
            train_bar.set_description(
                f"Epoch {epoch} training: stepping"
            )
            if downcast:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(rotation_module.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(rotation_module.parameters(), config.clip_grad_norm)
                optimizer.step()
            step += 1
            if config.wandb_enabled:
                wandb.log({"training_loss": scaled_loss.detach().item()}, step=step)
            epoch_train_loss += scaled_loss.detach().item()
        rotation_module.eval()
        torch.cuda.empty_cache()
        with torch.inference_mode():
            test_bar = tqdm(testloader, disable=config.epochs > 1)
            test_bar.set_description(
                f"Epoch {epoch}: validation. Batch size: {testloader.batch_size}. Device: {device}"
            )
            for orig_tokens_test, orig_resid_test, new_resid_test, answers_test in test_bar:
                orig_tokens_test = orig_tokens_test.to(device)
                orig_resid_test = orig_resid_test.to(device)
                new_resid_test = new_resid_test.to(device)
                answers_test = answers_test.to(device)
                eval_loss = rotation_module(
                    orig_resid_test,
                    new_resid_test,
                    layer=config.eval_layer,
                    position=position_test,
                    orig_tokens=orig_tokens_test,
                    patching_metric=partial(metric_test, answer_tokens=answers_test),
                )
                epoch_test_loss += eval_loss.item()

        if config.wandb_enabled:
            wandb.log({
                "epoch_training_loss": epoch_train_loss / len(trainloader),
                "epoch_validation_loss": epoch_test_loss / len(testloader),
            }, step=step)
        # Store the loss and model for this seed
        losses_train.append(epoch_train_loss)
        losses_test.append(epoch_test_loss)
        models.append(rotation_module.state_dict())

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
    prompt_type: Union[PromptType, List[PromptType]], 
    position: Union[None, str, List[str]], 
    layer: int, 
    model: HookedTransformer,
    batch_size: int = 32, 
    max_dataset_size: Optional[int] = None, 
    scaffold: Optional[ReviewScaffold] = None,
    pin_memory: bool = True, 
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
    verbose: bool = False,
    label: Optional[str] = None,
):
    """
    Wrapper for utils.prompts.get_dataset that returns a dataset in a useful form for DAS
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if prompt_type is None or prompt_type == PromptType.NONE:
        return DataLoader([]), None, None
    clean_corrupt_data = get_dataset(
        model, device, prompt_type=prompt_type, scaffold=scaffold,
        position=position, label=label,
    )
    if max_dataset_size is not None:
        clean_corrupt_data = clean_corrupt_data.get_subset(
            list(range(max_dataset_size))
        )
    results = clean_corrupt_data.run_with_cache(
        model,
        names_filter=lambda name: name in ('blocks.0.attn.hook_z', get_resid_name(layer, model)[0]),
        batch_size=batch_size,
        device=device,
        requires_grad=requires_grad,
    )
    act_name, _ = get_resid_name(layer, model)
    loss_fn = partial(
        logit_diff_denoising, 
        flipped_value=results.clean_logit_diff.item(), 
        clean_value=results.corrupted_logit_diff.item(),
        return_tensor=True,
    )
    if verbose:
        print(
            f"clean logit diff: {results.clean_logit_diff}, \n"
            f"corrupted logit diff: {results.corrupted_logit_diff}"
        )
    orig_resid: Float[Tensor, "batch *pos d_model"] = results.corrupted_cache[act_name]
    new_resid: Float[Tensor, "batch *pos d_model"] = results.clean_cache[act_name]
    if orig_resid.ndim == 3:
        orig_resid = orig_resid[
            torch.arange(len(orig_resid)), 
            clean_corrupt_data.position.to(orig_resid.device), 
            :
        ]
        new_resid = new_resid[
            torch.arange(len(new_resid)), 
            clean_corrupt_data.position.to(new_resid.device), 
            :
        ]
    # Create a TensorDataset from the tensors
    das_dataset = TensorDataset(
        clean_corrupt_data.corrupted_tokens.detach().cpu(), 
        orig_resid.detach().cpu().requires_grad_(requires_grad), 
        new_resid.detach().cpu().requires_grad_(requires_grad), 
        clean_corrupt_data.answer_tokens.detach().cpu(),
    )
    # Create a DataLoader from the dataset
    das_dataloader = DataLoader(
        das_dataset, batch_size=batch_size, pin_memory=pin_memory
        )
    return das_dataloader, loss_fn, clean_corrupt_data.position


def train_das_subspace(
    model: HookedTransformer, device: torch.device,
    train_type: Union[PromptType, List[PromptType]], train_pos: Union[None, str], train_layer: int,
    test_type: PromptType, test_pos: Union[None, str], test_layer: int,
    batch_size: int = 32, max_dataset_size: Optional[int] = None, 
    profiler: bool = False,
    downcast: bool = False, 
    scaffold: Optional[ReviewScaffold] = None,
    data_requires_grad: bool = False, 
    verbose: bool = False,
    d_das: int = 1, 
    train_label: Optional[str] = None,
    **config_arg,
) -> Tuple[Float[Tensor, "batch d_model"], str]:
    """
    Entrypoint to be used in directional patching experiments
    Given training/validation datasets, train a DAS subspace.
    """
    torch.cuda.empty_cache()
    if data_requires_grad:
        warnings.warn("data_requires_grad is True. This is not recommended.")
    trainloader, loss_fn, train_position = get_das_dataset(
        train_type, position=train_pos, layer=train_layer, model=model,
        batch_size=batch_size, max_dataset_size=max_dataset_size,
        scaffold=scaffold, device=device, requires_grad=data_requires_grad,
        verbose=verbose, label=train_label
    )
    if test_type != train_type or test_pos != train_pos or test_layer != train_layer:
        testloader, loss_fn_val, test_position = get_das_dataset(
            test_type, position=test_pos, layer=test_layer, model=model,
            batch_size=batch_size, max_dataset_size=max_dataset_size,
            scaffold=scaffold, device=device, requires_grad=data_requires_grad,
            verbose=verbose,
        )
    else:
        testloader, loss_fn_val, test_position = trainloader, loss_fn, train_position
    config = dict(
        train_layer=train_layer,
        train_position=train_position,
        eval_layer=test_layer,
        eval_position=test_position,
        batch_size=batch_size,
        d_das=d_das,
    )
    config.update(config_arg)
    directions = fit_rotation(
        trainloader=trainloader,
        testloader=testloader,
        metric_train=loss_fn,
        metric_test=loss_fn_val,
        position_train=train_position,
        position_test=test_position,
        model=model,
        device=device,
        profiler=profiler,
        downcast=downcast,
        verbose=verbose,
        **config,
    )
    d_das_str = f'{d_das}d' if d_das > 1 else ''
    train_pos_str = f'_{train_pos}' if train_pos is not None else ''
    save_path = f'das{d_das_str}_{train_label}{train_pos_str}_layer{train_layer}'
    save_array(
        directions.detach().cpu().squeeze(1).numpy(), 
        save_path, 
        model,
    )
    return directions, save_path