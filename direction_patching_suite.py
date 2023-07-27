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
#%%
update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}

def imshow_p(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
      if f"xaxis_{setting}" in kwargs_post:
          i = 2
          while f"xaxis{i}" in fig["layout"]:
            kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
            i += 1
    fig.update_layout(**kwargs_post)
    fig.show(renderer=renderer)
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
model.set_use_attn_result(True)
model.name = MODEL_NAME
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
#%%
# ============================================================================ #
# Directional activation patching
def name_filter(name: str):
    return (
        name.endswith('result') or 
        name.endswith('resid_pre') or
        name.endswith('resid_post') or  
        name.endswith('attn_out') or 
        name.endswith('mlp_out') or 
        (name == 'blocks.0.attn.hook_q') or 
        (name == 'blocks.0.attn.hook_z')
    )
# N.B. corrupt -> clean
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens,
)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
print('clean logit diff', clean_logit_diff)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens
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
        is_result = act_name.endswith('result')
        is_resid = (
            act_name.endswith('resid_pre') or
            act_name.endswith('resid_post') or
            act_name.endswith('attn_out') or
            act_name.endswith('mlp_out')
        )
        if is_resid:
            clean_value = clean_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s d, d -> b s'
            )
            clean_proj = einops.einsum(
                clean_value, sentiment_dir, 'b s d, d -> b s'
            )
            sentiment_dir_broadcast = einops.repeat(
                sentiment_dir, 'd -> b s d', 
                b=corrupt_value.shape[0], 
                s=corrupt_value.shape[1], 
            )
            proj_diff = einops.repeat(
                clean_proj - corrupt_proj, 
                'b s -> b s d', 
                d=corrupt_value.shape[-1]
            )
            sentiment_adjustment = proj_diff * sentiment_dir_broadcast
            cache_dict[act_name] = (
                corrupt_value + sentiment_adjustment
            )
        elif is_result:
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
        else:
            cache_dict[act_name] = clean_value

    return ActivationCache(cache_dict, model)
#%%
new_cache = create_cache_for_dir_patching(
    clean_cache, corrupted_cache, sentiment_dir
)
#%%[markdown]
#### Head patching
#%%
head_results: Float[Tensor, "layer head"] = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=new_cache,
    patching_nodes=IterNode(["result"]),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
#%%
fig = px.imshow(
    head_results['result'] * 100,
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
#%%[markdown]
#### Residual stream patching
#%%
layer_results: Float[Tensor, "layer"] = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=new_cache,
    patching_nodes=IterNode(["resid_post"]),
    patching_metric=logit_diff_denoising,
    verbose=True,
)['resid_post'] * 100
#%%
fig = px.line(
    layer_results,
    title="Patching sentiment component of residual stream (corrupted -> clean)",
    labels={"index": "Layer", "value": "Logit diff (%)"},
    width=600,
)
fig.update_layout(dict(
    coloraxis=dict(colorbar_ticksuffix = "%"),
    # border=True,
    margin={"r": 100, "l": 100},
    showlegend=False,
))
#%%[markdown]
#### Attn vs. MLP patching
# %%
attn_mlp_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=new_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=logit_diff_denoising,
    verbose=True,
)
#%%
assert attn_mlp_results.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
imshow_p(
    torch.stack([r.T for r in attn_mlp_results.values()]) * 100, # we transpose so layer is on the y-axis
    facet_col=0,
    facet_labels=["resid_pre", "attn_out", "mlp_out"],
    title="Patching at resid stream & layer outputs (corrupted -> clean)",
    labels={"x": "Sequence position", "y": "Layer", "color": "Logit diff variation"},
    x=labels,
    xaxis_tickangle=45,
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1300,
    zmin=-50,
    zmax=50,
    margin={"r": 100, "l": 100}
)
#%%
