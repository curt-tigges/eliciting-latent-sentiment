from typing import Callable, List, Literal, Union
import torch
from torch import Tensor
from jaxtyping import Float, Int
import numpy as np
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache


def handle_position(
    pos: Union[Literal["each"], int, List[int]],
    component: Float[Tensor, "batch pos ..."],
) -> Int[Tensor, "subset_pos"]:
    """Handles the position argument for ablation functions"""
    if isinstance(pos, int):
        pos = torch.tensor([pos])
    if pos == "each":
        pos = torch.tensor(list(range(component.shape[1])))
    return pos


def resample_cache_component(
    component: Float[Tensor, "batch ..."], seed: int = 77
) -> Float[Tensor, "batch ..."]:
    """Resample-ablates a batch tensor according to the index of the first dimension"""
    batch_size = component.shape[0]
    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Resample the batch
    indices = np.random.choice(batch_size, batch_size, replace=True)
    component = component[indices]
    return component


def mean_over_cache_component(component: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    """
    Mean-ablates a batch tensor

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
    """Zero-ablates a batch tensor"""
    return torch.zeros_like(component)


def zero_attention_pos_hook(
    pattern: Float[Tensor, "batch head seq_Q seq_K"], hook: HookPoint,
    pos_by_batch: List[List[int]], layer: int = 0, head_idx: int = 0,
) -> Float[Tensor, "batch head seq_Q seq_K"]:
    """Zero-ablates an attention pattern tensor at a particular position"""
    assert 'pattern' in hook.name

    batch_size = pattern.shape[0]
    assert len(pos_by_batch) == batch_size

    for i in range(batch_size):
        for p in pos_by_batch[i]:
            pattern[i, head_idx, p, p] = 0
            
    return pattern


def freeze_attn_pattern_hook(
    pattern: Float[Tensor, "batch head seq_Q seq_K"], hook: HookPoint, cache: ActivationCache, 
    layer: int = 0, head_idx: int = 0,
) -> Float[Tensor, "batch head seq_Q seq_K"]:
    """Freeze the attention pattern for a given position, layer and head"""
    assert 'pattern' in hook.name
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_idx, :, :] 
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_idx, :, :]
    return pattern


def freeze_layer_pos_hook(
    component: Float[Tensor, "batch pos ..."],
    hook: HookPoint,
    cache: ActivationCache,
    component_type: str = "hook_resid_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0
) -> Float[Tensor, "batch pos ..."]:
    """Base function to freeze the layer for a given position, layer and head"""
    assert component_type in hook.name
    pos = handle_position(pos, component)
    for p in pos:
        component[:, p, :] = cache[f"blocks.{layer}.{component_type}"][:, p, :]
    return component


def freeze_mlp_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"], hook: HookPoint, cache: ActivationCache, 
    component_type: str = "hook_post", pos: Union[Literal["each"], int, List[int]] = -1, layer: int = 0
):
    """Freeze the mlp for a given position, layer and head"""
    return freeze_layer_pos_hook(
        component, hook, cache, f"mlp.{component_type}", pos, layer
    )


def freeze_attn_head_pos_hook(
    component: Float[Tensor, "batch pos head d_head"], hook: HookPoint, 
    cache: ActivationCache, component_type: str = "hook_z", 
    pos: Union[Literal["each"], int, List[int]] = -1, layer: int = 0, head_idx: int = 0
):
    """Freeze the attention head for a given position, layer and head"""
    assert component_type in hook.name
    pos = handle_position(pos, component)
    for p in pos:
        component[:, p, head_idx, :] = cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :]
    return component


def ablate_layer_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]] = None,
    component_type: str = "hook_resid_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0
) -> Float[Tensor, "batch pos d_mlp"]:
    """Base function to ablate the layer for a given position, layer and head"""
    assert component_type in hook.name
    pos = handle_position(pos, component)
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos:
        component[:, p, :] = ablation_func(cache[f"blocks.{layer}.{component_type}"][:, p, :])
    return component


def ablate_mlp_pos_hook(
    component: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]] = None,
    component_type: str = "hook_post",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0
) -> Float[Tensor, "batch pos d_mlp"]:
    return ablate_layer_pos_hook(
        component, hook, cache, ablation_func, f"mlp.{component_type}", pos, layer
    )


def ablate_attn_head_pos_hook(
    component: Float[Tensor, "batch pos d_head"],
    hook: HookPoint,
    cache: ActivationCache,
    ablation_func: Callable[[Float[Tensor, "batch ..."]], Float[Tensor, "batch ..."]] = None,
    component_type: str = "hook_z",
    pos: Union[Literal["each"], int, List[int]] = -1,
    layer: int = 0,
    head_idx: int = 0
) -> Float[Tensor, "batch pos d_head"]:
    assert component_type in hook.name
    pos = handle_position(pos, component)
    if ablation_func is None:
        ablation_func = lambda x: x
    for p in pos:
        component[:, p, head_idx, :] = ablation_func(cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :])
    return component


def ablate_resid_with_precalc_mean(
    component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    cached_means: Float[Tensor, "layer d_model"],
    pos_by_batch: List[List[int]],
    layer: int = 0,
) -> Float[Tensor, "batch pos d_model"]:
    """
    Mean-ablates a batch tensor

    :param component: the tensor to compute the mean over the batch dim of
    :return: the mean over the cache component of the tensor
    """
    assert 'resid' in hook.name

    batch_size = component.shape[0]
    assert len(pos_by_batch) == batch_size

    for i in range(batch_size):
        for p in pos_by_batch[i]:
            component[i, p, :] = cached_means[layer]
            
    return component