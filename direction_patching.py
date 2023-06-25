#%%
import einops
import numpy as np
from jaxtyping import Float, Int
import plotly.express as px
from utils.prompts import get_dataset
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from typing import Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from path_patching import act_patch, Node, IterNode
#%% # Model loading
device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.cfg.use_attn_result = True
#%% # Direction loading
sentiment_dir: Float[np.ndarray, "d_model"] = np.load(
    'data/km_line_embed_and_mlp0.npy'
)
sentiment_dir /= np.linalg.norm(sentiment_dir)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(
    device, dtype=torch.float32
)
#%% # Data loading
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
#%%
def get_logit_diff(logits, answer_token_indices, per_prompt=False):
    """Gets the difference between the logits of the provided tokens (e.g., the correct and incorrect tokens in IOI)

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    left_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    right_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if per_prompt:
        print(left_logits - right_logits)

    return (left_logits - right_logits).mean()
#%%
# ============================================================================ #
# Directional activation patching
# N.B. corrupt -> clean
def name_filter(name: str):
    return (
        name.endswith('result') or 
        (name == 'blocks.0.attn.hook_q') or 
        (name == 'blocks.0.attn.hook_z')
    )
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter = name_filter
)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
print('clean logit diff', clean_logit_diff)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens, names_filter = name_filter
)
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
print('corrupted logit diff', corrupted_logit_diff)
#%%
def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    flipped_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff)).item()
#%% # Create new cache
def create_cache_for_dir_patching(
    clean_cache: ActivationCache, 
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"]
) -> ActivationCache:
    # N.B. corrupt -> clean
    cache_dict = dict()
    for act_name, clean_value in clean_cache.items():
        if act_name.endswith('q') or act_name.endswith('z'):
            cache_dict[act_name] = clean_value
            continue
        clean_value = clean_value.to(device)
        corrupt_value = corrupted_cache[act_name].to(device)
        corrupt_proj = einops.einsum(
            corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
        )
        clean_proj = einops.einsum(
            clean_value, sentiment_dir, 'b s h d, d -> b s h'
        )
        sentiment_dir_broadcast = einops.repeat(
            sentiment_dir, 'd -> b s h d', 
            b=corrupt_value.shape[0], 
            s=corrupt_value.shape[1], 
            h=corrupt_value.shape[2]
        )
        proj_diff = einops.repeat(
            clean_proj - corrupt_proj, 
            'b s h -> b s h d', 
            d=corrupt_value.shape[3]
        )
        sentiment_adjustment = proj_diff * sentiment_dir_broadcast
        cache_dict[act_name] = (
            corrupt_value + sentiment_adjustment
        )
    return ActivationCache(cache_dict, model)
#%%
new_cache = create_cache_for_dir_patching(
    clean_cache, corrupted_cache, sentiment_dir
)
print(new_cache.keys())
#%%
patch_results: Float[Tensor, "layer head"] = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=new_cache,
    patching_nodes=IterNode(["result"]),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
#%%
fig = px.imshow(
    patch_results['result'] * 100,
    title="Patching KM component of attention heads (corrupted -> clean)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    width=600,
)
fig.update_layout(dict(
    coloraxis=dict(colorbar_ticksuffix = "%"),
    # border=True,
    margin={"r": 100, "l": 100}
))
# %%
# ============================================================================ #
# Mean ablation
def create_cache_for_mean_ablation(
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"],
) -> ActivationCache:
    cache_dict = dict()
    for act_name, corrupt_value in corrupted_cache.items():
        if act_name.endswith('q') or act_name.endswith('z'):
            cache_dict[act_name] = corrupt_value
            continue
        corrupt_value: Float[Tensor, "b s h d"] = corrupt_value.to(device)
        corrupt_proj = einops.einsum(
            corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
        )
        mean_proj: Float[Tensor, ""] = einops.reduce(
            corrupt_proj, 'b s h -> ', 'mean'
        )
        sentiment_dir_broadcast = einops.repeat(
            sentiment_dir, 'd -> b s h d', 
            b=corrupt_value.shape[0], 
            s=corrupt_value.shape[1], 
            h=corrupt_value.shape[2]
        )
        proj_broadcast = einops.repeat(
            mean_proj - corrupt_proj, 
            'b s h -> b s h d', 
            d=corrupt_value.shape[3]
        )
        sentiment_adjustment = proj_broadcast * sentiment_dir_broadcast
        cache_dict[act_name] = (
            corrupt_value + sentiment_adjustment
        )
    return ActivationCache(cache_dict, model)
#%%
mean_cache = create_cache_for_mean_ablation(
    corrupted_cache, sentiment_dir
)
print(mean_cache.keys())
#%%
mean_ablation_results: Float[Tensor, "layer head"] = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=mean_cache,
    patching_nodes=IterNode(["result"]),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
#%%
fig = px.imshow(
    mean_ablation_results['result'] * 100,
    title="Mean-ablating KM component of attention heads",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    width=600,
)
fig.update_layout(dict(
    coloraxis=dict(colorbar_ticksuffix = "%"),
    # border=True,
    margin={"r": 100, "l": 100}
))
# %%
# ============================================================================ #
# Resample ablation
def create_cache_for_resample_ablation(
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"],
) -> ActivationCache:
    cache_dict = dict()
    for act_name, corrupt_value in corrupted_cache.items():
        if act_name.endswith('q') or act_name.endswith('z'):
            cache_dict[act_name] = corrupt_value
            continue
        corrupt_value: Float[Tensor, "b s h d"] = corrupt_value.to(device)
        corrupt_proj = einops.einsum(
            corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
        )
        corrupt_proj_flat = einops.rearrange(
            corrupt_proj, 'b s h -> (b s h)'
        )
        corrupt_proj_perm = corrupt_proj_flat[torch.randperm(
            corrupt_proj_flat.shape[0]
        )]
        resample_proj: Float[Tensor, "b s h"] = einops.rearrange(
            corrupt_proj_perm, 
            '(b s h) -> b s h',
            b=corrupt_value.shape[0], 
            s=corrupt_value.shape[1], 
            h=corrupt_value.shape[2]
        )
        sentiment_dir_broadcast = einops.repeat(
            sentiment_dir, 'd -> b s h d', 
            b=corrupt_value.shape[0], 
            s=corrupt_value.shape[1], 
            h=corrupt_value.shape[2]
        )
        proj_broadcast = einops.repeat(
            resample_proj - corrupt_proj, 
            'b s h -> b s h d', 
            d=corrupt_value.shape[3]
        )
        sentiment_adjustment = proj_broadcast * sentiment_dir_broadcast
        cache_dict[act_name] = (
            corrupt_value + sentiment_adjustment
        )
    return ActivationCache(cache_dict, model)
#%%
resample_cache = create_cache_for_resample_ablation(
    corrupted_cache, sentiment_dir
)
print(resample_cache.keys())
#%%
resample_ablation_results: Float[Tensor, "layer head"] = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=resample_cache,
    patching_nodes=IterNode(["result"]),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
#%%
fig = px.imshow(
    resample_ablation_results['result'] * 100,
    title="Resample-ablating KM component of attention heads",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    width=600,
)
fig.update_layout(dict(
    coloraxis=dict(colorbar_ticksuffix = "%"),
    margin={"r": 100, "l": 100}
))

# %%
