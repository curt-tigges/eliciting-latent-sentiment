from typing import Callable, List, Literal, Union
import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache


def resample_cache_component(
    component: Float[Tensor, "batch..."], seed: int = 77
) -> Float[Tensor, "batch..."]:
    """Resamples a batch tensor according to the index of the first dimension"""
    batch_size = component.shape[0]
    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Resample the batch
    indices = np.random.choice(batch_size, batch_size, replace=True)
    component = component[indices]
    return component


def mean_over_cache_component(component: Float[Tensor, "batch..."]) -> Float[Tensor, "batch..."]:
    """
    Computes the mean over the cache component of a tensor.

    :param component: the tensor to compute the mean over the batch dim of
    :return: the mean over the cache component of the tensor
    """
    copy_to_ablate = component.clone()
    batch_mean = component.mean(dim=0)
    # make every batch item a copy of the batch mean
    for row in range(component.shape[0]):
        copy_to_ablate[row] = batch_mean
    assert copy_to_ablate.shape == component.shape
    return copy_to_ablate


def zero_cache_component(component: torch.tensor) -> torch.tensor:
    return torch.zeros_like(component)


def freeze_attn_pattern_hook(
    pattern: Float[Tensor, "batch head seq_Q seq_K"], hook: HookPoint, cache: ActivationCache, 
    layer: int = 0, head_idx: int = 0,
) -> Float[Tensor, "batch head seq_Q seq_K"]:
    assert 'pattern' in hook.name
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_idx, :, :] 
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_idx, :, :]
    return pattern


def freeze_attn_head_pos_hook(
    component: Float[Tensor, "batch pos head d_head"], hook: HookPoint, 
    cache: ActivationCache, component_type: str = "hook_z", 
    pos: Union[Literal["each"], int, List[int]] = -1, layer: int = 0, head_idx: int = 0
):
    assert component_type in hook.name
    if isinstance(pos, int):
        pos = [pos]

    if pos == "each":
        pos = torch.tensor(list(range(component.shape[1])))

    for p in pos:
        component[:, p, head_idx, :] = cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :]
    
    return component


def freeze_mlp_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"], hook: HookPoint, cache: ActivationCache, 
    component_type: str = "hook_post", pos: Union[int, List[int]] = -1, layer: int = 0
):
    assert component_type in hook.name
    if isinstance(pos, int):
        pos = [pos]

    for p in pos:
        component[:, p, :] = cache[f"blocks.{layer}.mlp.{component_type}"][:, p, :]
    
    return component


def freeze_layer_pos_hook(
    component: Float[Tensor, "batch pos ..."],
    hook: HookPoint,
    cache: ActivationCache,
    component_type: str = "hook_resid_post",
    pos: Union[int, List[int]] = -1,
    layer: int = 0
) -> Float[Tensor, "batch pos ..."]:
    assert component_type in hook.name
    if isinstance(pos, int):
        pos = [pos]
    for p in pos:
        component[:, p, :] = cache[f"blocks.{layer}.{component_type}"][:, p, :]
    return component


def ablate_attn_head_pos_hook(
    component: Float[Tensor, "batch pos d_head"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]] = None,
    component_type: str = "hook_z",
    pos: Union[int, List[int]] = -1,
    layer: int = 0,
    head_idx: int = 0
) -> Float[Tensor, "batch pos d_head"]:
    assert component_type in hook.name
    if isinstance(pos, int):
        pos = [pos]
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos:
        component[:, p, head_idx, :] = ablation_func(cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :])
    return component


def ablate_mlp_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]] = None,
    component_type: str = "hook_post",
    pos: Union[int, List[int]] = -1,
    layer: int = 0
) -> Float[Tensor, "batch pos d_mlp"]:
    assert component_type in hook.name
    if isinstance(pos, int):
        pos = [pos]
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos:
        component[:, p, :] = ablation_func(cache[f"blocks.{layer}.mlp.{component_type}"][:, p, :])
    return component


def ablate_layer_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]] = None,
    component_type: str = "hook_resid_post",
    pos: Union[int, List[int]] = -1,
    layer: int = 0
) -> Float[Tensor, "batch pos d_mlp"]:
    assert component_type in hook.name
    if isinstance(pos, int):
        pos = [pos]
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos:
        component[:, p, :] = ablation_func(cache[f"blocks.{layer}.{component_type}"][:, p, :])
    return component