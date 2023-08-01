#%%
import einops
import numpy as np
from jaxtyping import Float, Int
import plotly.express as px
from utils.prompts import get_dataset
from utils.circuit_analysis import get_logit_diff
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from typing import Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from path_patching import act_patch, Node, IterNode
from utils.store import save_array, load_array
#%% # Model loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.name = MODEL_NAME
model.cfg.use_attn_result = True
#%% # Direction loading
sentiment_dir: Float[np.ndarray, "d_model"] = load_array(
    'km_2c_line_embed_and_mlp0', model
)
sentiment_dir /= np.linalg.norm(sentiment_dir)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(
    device, dtype=torch.float32
)
#%% # Data loading
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
# negative -> positive
all_prompts = all_prompts[::2]
answer_tokens = answer_tokens[::2]
clean_tokens = clean_tokens[::2]
corrupted_tokens = corrupted_tokens[::2]
model.to_string(corrupted_tokens[0])
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
    '''
    We patch the sentiment direction from corrupt to clean
    '''
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
    clean_cache: ActivationCache, 
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"],
) -> ActivationCache:
    '''
    We mean ablate along the sentiment_dir
    We use the clean cache just to help get the mean projection
    '''
    cache_dict = dict()
    for act_name, corrupt_value in corrupted_cache.items():
        if act_name.endswith('q') or act_name.endswith('z'):
            cache_dict[act_name] = corrupt_value
            continue
        clean_value: Float[Tensor, "b s h d"] = clean_cache[act_name].to(device)
        corrupt_value: Float[Tensor, "b s h d"] = corrupt_value.to(device)
        corrupt_proj = einops.einsum(
            corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
        )
        stacked_value = torch.cat([clean_value, corrupt_value], dim=0)
        stacked_proj = einops.einsum(
            stacked_value, sentiment_dir, 'b s h d, d -> b s h'
        )
        mean_proj: Float[Tensor, ""] = einops.reduce(
            stacked_proj, 'b s h -> ', 'mean'
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
    clean_cache, corrupted_cache, sentiment_dir
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
    '''
    We take a randperm of the projection of the corrupted value onto the
    sentiment direction
    This only uses the corrupted cache
    '''
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
head_resample_ablation_results: Float[Tensor, "layer head"] = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=resample_cache,
    patching_nodes=IterNode(["result"]),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
#%%
fig = px.imshow(
    head_resample_ablation_results['result'] * 100,
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