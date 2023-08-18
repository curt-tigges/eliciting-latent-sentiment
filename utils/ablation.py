import torch
import numpy as np

def resample_cache_component(component: torch.tensor) -> torch.tensor:
    """Resamples a batch tensor according to the index of the first dimension"""

    # set seeds
    np.random.seed(77)
    torch.manual_seed(77)

    batch_size = component.shape[0]

    # Resample the batch
    indices = np.random.choice(batch_size, batch_size, replace=True)
    component = component[indices]

    return component


def mean_over_cache_component(component: torch.tensor) -> torch.tensor:
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


def freeze_attn_pattern_hook(pattern, hook, cache, layer=0, head_idx=0):
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_idx, :, :] 
    pattern[:, head_idx, :, :] = cache[f"blocks.{layer}.attn.hook_pattern"][:, head_idx, :, :]
    return pattern


def freeze_attn_head_pos_hook(c, hook, cache, component_type="hook_z", pos=-1, layer=0, head_idx=0):
    
    if isinstance(pos, int):
        pos = [pos]

    if pos == "each":
        pos = torch.tensor(list(range(c.shape[1])))

    for p in pos:
        c[:, p, head_idx, :] = cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :]
    
    return c


def freeze_mlp_pos_hook(c, hook, cache, component_type="hook_post", pos=-1, layer=0):

    if isinstance(pos, int):
        pos = [pos]

    for p in pos:
        c[:, p, :] = cache[f"blocks.{layer}.mlp.{component_type}"][:, p, :]
    
    return c


def freeze_layer_pos_hook(c, hook, cache, component_type="hook_resid_post", pos=-1, layer=0):

    if isinstance(pos, int):
        pos = [pos]

    for p in pos:
        c[:, p, :] = cache[f"blocks.{layer}.{component_type}"][:, p, :]
    
    return c


def ablate_attn_head_pos_hook(c, hook, cache, ablation_func=None, component_type="hook_z", pos=-1, layer=0, head_idx=0):

    if isinstance(pos, int):
        pos = [pos]

    if ablation_func is None:
        ablation_func = lambda x: x

    for p in pos:
        c[:, p, head_idx, :] = ablation_func(cache[f"blocks.{layer}.attn.{component_type}"][:, p, head_idx, :])

    return c


def ablate_mlp_pos_hook(c, hook, cache, ablation_func=None, component_type="hook_post", pos=-1, layer=0):

    if isinstance(pos, int):
        pos = [pos]

    if ablation_func is None:
        ablation_func = lambda x: x

    for p in pos:
        c[:, p, :] = ablation_func(cache[f"blocks.{layer}.mlp.{component_type}"][:, p, :])

    return c


def ablate_layer_pos_hook(c, hook, cache, ablation_func=None, component_type="hook_resid_post", pos=-1, layer=0):

    if isinstance(pos, int):
        pos = [pos]

    if ablation_func is None:
        ablation_func = lambda x: x

    for p in pos:
        c[:, p, :] = ablation_func(cache[f"blocks.{layer}.{component_type}"][:, p, :])

    return c