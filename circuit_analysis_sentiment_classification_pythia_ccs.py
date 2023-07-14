# %% [markdown]
# # Classification Circuit in Pythia 1.4B

# %% [markdown]
# ## Setup

# %%
import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import yaml

import einops
from fancy_einsum import einsum

import circuitsvis as cv

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.patching as patching

from torch import Tensor
from tqdm.notebook import tqdm
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from rich import print as rprint

from typing import List, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

from functools import partial

from torchtyping import TensorType as TT

from path_patching import Node, IterNode, path_patch, act_patch

from utils.visualization import get_attn_head_patterns
from utils.prompts import get_ccs_dataset
from utils.store import load_array

# %%
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
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

def hist_p(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    names = kwargs_pre.pop("names", None)
    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.histogram(x=tensor, **kwargs_pre).update_layout(**kwargs_post)
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i // 2]
    fig.show(renderer)

# %%
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def two_lines(tensor1, tensor2, renderer=None, **kwargs):
    px.line(y=[utils.to_numpy(tensor1), utils.to_numpy(tensor2)], **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %% [markdown]
# ## Exploratory Analysis
#
MODEL_NAME = "EleutherAI/pythia-1.4b"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.name = MODEL_NAME
#model.set_use_attn_result(True)
#%%
ccs_dir: Float[np.ndarray, "d_model"] = load_array("ccs", model)[0]
# normalise ccs_dir vector
ccs_dir /= np.linalg.norm(ccs_dir)
ccs_dir = torch.from_numpy(ccs_dir).to(model.cfg.device)
ccs_dir.shape

# %% [markdown]
# ### Dataset Construction

# %%
neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, _ = get_ccs_dataset(
    model, device,
)
#%%
clean_tokens = torch.where(gt_labels == 1, pos_tokens, neg_tokens)
corrupted_tokens = torch.where(gt_labels == 1, neg_tokens, pos_tokens)

# %%
clean_tokens.shape, corrupted_tokens.shape
# %%
ccs_proj_directions = einops.repeat(
    ccs_dir, "d_model -> batch d_model", batch=len(clean_tokens)
)
print("CCS projection directions shape:", ccs_proj_directions.shape)
#%%
def get_ccs_proj(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"] = ccs_proj_directions,
):
    final_residual_stream: Float[
        Tensor, "batch pos d_model"
    ] = cache["resid_post", -1]
    final_token_residual_stream: Float[
        Tensor, "batch d_model"
    ] = final_residual_stream[:, -1, :]
    # Apply LayerNorm scaling
    # pos_slice is the subset of the positions we take - 
    # here the final token of each prompt
    scaled_final_token_residual_stream: Float[
        Tensor, "batch d_model"
    ] = clean_cache.apply_ln_to_stack(
        final_token_residual_stream, layer = -1, pos_slice=-1
    )
    average_ccs_proj = einsum(
        "batch d_model, batch d_model -> ", 
        scaled_final_token_residual_stream, 
        directions
    ) / len(clean_tokens)
    return average_ccs_proj


# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_ccs_proj = get_ccs_proj(clean_cache)
clean_ccs_proj

# %%
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
corrupted_ccs_proj = get_ccs_proj(corrupted_cache)
corrupted_ccs_proj

# %%
def ccs_proj_denoising(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"] = ccs_proj_directions,
    flipped_ccs_proj: float = corrupted_ccs_proj,
    clean_ccs_proj: float = clean_ccs_proj,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_ccs_proj = get_ccs_proj(cache, directions)
    ld = (
        (patched_ccs_proj - flipped_ccs_proj) / 
        (clean_ccs_proj  - flipped_ccs_proj)
    )
    if return_tensor:
        return ld
    else:
        return ld.item()


def ccs_proj_noising(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"] = ccs_proj_directions,
    clean_ccs_proj: float = clean_ccs_proj,
    corrupted_ccs_proj: float = corrupted_ccs_proj,
    return_tensor: bool = False,
) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_ccs_proj = get_ccs_proj(cache, directions)
        ld = ((patched_ccs_proj - clean_ccs_proj) / (clean_ccs_proj - corrupted_ccs_proj))

        if return_tensor:
            return ld
        else:
            return ld.item()

ccs_proj_denoising_tensor = partial(ccs_proj_denoising, return_tensor=True)
ccs_proj_noising_tensor = partial(ccs_proj_noising, return_tensor=True)

# %% [markdown]
# ### Direct Logit Attribution


# %%
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 
final_residual_stream = clean_cache["resid_post", -1]
print("Final residual stream shape:", final_residual_stream.shape)
final_token_residual_stream = final_residual_stream[:, -1, :]
# Apply LayerNorm scaling
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream = clean_cache.apply_ln_to_stack(final_token_residual_stream, layer = -1, pos_slice=-1)

average_ccs_proj = einsum("batch d_model, batch d_model -> ", scaled_final_token_residual_stream, ccs_proj_directions)/len(all_prompts)
print("Calculated average logit diff:", average_ccs_proj.item())
print("Original CCS projection:",clean_ccs_proj.item())

# %% [markdown]
# #### Logit Lens

# %%
def residual_stack_to_ccs_proj(
    residual_stack: TT["components", "batch", "d_model"], 
    cache: ActivationCache
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer = -1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...", 
        scaled_residual_stack, ccs_proj_directions
    ) / len(clean_tokens)

# %%
accumulated_residual, labels = clean_cache.accumulated_resid(
    layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
)
logit_lens_ccs_projs = residual_stack_to_ccs_proj(
    accumulated_residual, clean_cache
)
line(
    logit_lens_ccs_projs, 
    x=np.arange(model.cfg.n_layers*1+1)/2, 
    hover_name=labels, 
    title="CCS projection From Accumulated Residual Stream",
    labels={'x': "Layer", 'y': "CCS projection"},
)

# %% [markdown]
# #### Layer Attribution

# %%
per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_ccs_projs = residual_stack_to_ccs_proj(per_layer_residual, clean_cache)

line(per_layer_ccs_projs, hover_name=labels, title="CCS projection From Each Layer")

# %% [markdown]
# #### Head Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

per_head_residual, labels = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_ccs_projs = residual_stack_to_ccs_proj(per_head_residual, clean_cache)
per_head_ccs_projs = einops.rearrange(per_head_ccs_projs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
per_head_ccs_projs_pct = per_head_ccs_projs
imshow(
    per_head_ccs_projs_pct, 
    labels={"x":"Head", "y":"Layer"}, 
    title="CCS projection From Each Head"
)

# %% [markdown]
# ### Activation Patching

# %% [markdown]
# #### Attention Heads

# %%
results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode("z"), # iterating over all heads' output in all layers
    patching_metric=ccs_proj_denoising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
imshow_p(
    results['z'] * 100,
    title="Patching output of attention heads (corrupted -> clean)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

# %% [markdown]
# #### Head Output by Component

# %%
# iterating over all heads' output in all layers

results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
    patching_metric=ccs_proj_denoising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
assert results.keys() == {"z", "q", "k", "v", "pattern"}
#assert all([r.shape == (12, 12) for r in results.values()])

imshow_p(
    torch.stack(tuple(results.values())) * 100,
    facet_col=0,
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Patching output of attention heads (corrupted -> clean)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1500,
    margin={"r": 100, "l": 100}
)

# %% [markdown]
# #### Residual Stream & Layer Outputs

# %%
# patching at each (layer, sequence position) for each of (resid_pre, attn_out, mlp_out) in turn

results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=ccs_proj_denoising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
assert results.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
imshow_p(
    torch.stack([r.T for r in results.values()]) * 100, # we transpose so layer is on the y-axis
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

# %%
import transformer_lens.patching as patching
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

attn_head_out_act_patch_results = patching.get_act_patch_attn_head_out_by_pos(model, corrupted_tokens, clean_cache, ccs_proj_denoising_tensor)
attn_head_out_act_patch_results = einops.rearrange(attn_head_out_act_patch_results, "layer pos head -> (layer head) pos")

# %%
imshow(attn_head_out_act_patch_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
        y=ALL_HEAD_LABELS,
        height=1200,
        width=1200,
        zmin=-0.1,
        zmax=0.1,
        title="attn_head_out Activation Patching By Pos")

# %% [markdown]
# ### Circuit Analysis With Patch Patching & Attn Visualization

# %% [markdown]
# #### Heads Influencing Logit Diff

# %%
results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode('z'), # This means iterate over all heads in all layers
    receiver_nodes=Node('resid_post', 23), # This is resid_post at layer 11
    patching_metric=ccs_proj_noising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
imshow_p(
    results['z'],
    title="Direct effect on logit diff (patch from head output -> final resid)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

# %%
from utils.visualization import (
    plot_attention_heads,
    scatter_attention_and_contribution
)

import circuitsvis as cv

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.5])

# %%
from utils.visualization import get_attn_head_patterns

top_k = 6
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, all_prompts[21], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %%
from visualization_utils import scatter_attention_and_contribution_sentiment

from plotly.subplots import make_subplots

# Get the figures
fig1 = scatter_attention_and_contribution_sentiment(model, (10, 11), all_prompts, [22 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)
fig2 = scatter_attention_and_contribution_sentiment(model, (13, 8), all_prompts, [22 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)
fig3 = scatter_attention_and_contribution_sentiment(model, (16, 10), all_prompts, [22 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)
fig4 = scatter_attention_and_contribution_sentiment(model, (18, 0), all_prompts, [22 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)

# Create subplot
fig = make_subplots(rows=2, cols=2, subplot_titles=("Head 10.11", "Head 13.8", "Head 16.10", "Head 18.0"))

# Add each figure's data to the subplot
for i, subplot_fig in enumerate([fig1, fig2, fig3, fig4], start=1):
    row = (i-1)//2 + 1
    col = (i-1)%2 + 1
    for trace in subplot_fig['data']:
        # Only show legend for the first subplot
        trace.showlegend = (i == 1)
        fig.add_trace(trace, row=row, col=col)

# Update layout
fig.update_layout(height=600, title_text="DAE Heads")

# Update axes labels
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text="Attn Prob on Word", row=i, col=j)
        fig.update_yaxes(title_text="Dot w Sentiment Output Embed", row=i, col=j)

fig.show()

# %% [markdown]
# #### Direct Attribute Extraction Heads

# %% [markdown]
# ##### Overall

# %%
DAE_HEADS = [(21, 0)]

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in DAE_HEADS],
    patching_metric=ccs_proj_noising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
imshow_p(
        results["z"][:22] * 100,
        title=f"Direct effect on DAE Heads' values)",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.5])

# %%
top_k = 3
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, all_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %% [markdown]
# ##### Key Positions

# %%
import pickle
results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in DAE_HEADS],
    patching_metric=ccs_proj_noising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
attn_head_pos_results = einops.rearrange(results['z'], "pos layer head -> (layer head) pos")
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
imshow(attn_head_pos_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
        y=ALL_HEAD_LABELS,
        height=1200,
        width=1200,
        #zmin=-0.1,
        #zmax=0.1,
        title="attn_head_out Path Patching for DAE Heads By Pos")

# %%
for i in range(0, results["z"].shape[0]):
    imshow_p(
        results["z"][i][:22] * 100,
        title=f"Direct effect on Sentiment Attenders' values from position {i} ({tokens[i]})",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.1])

# %%
top_k = 4
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, all_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %% [markdown]
# #### Intermediate Attribute Extraction Heads

# %% [markdown]
# ##### Overall

# %%
IAE_HEADS = [(11, 1), (13, 13), (16, 4), (18, 2), (21, 14)]

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in IAE_HEADS],
    patching_metric=ccs_proj_noising,
    verbose=True,
    apply_metric_to_cache=True,
)


# %%
imshow_p(
        results["z"][:18] * 100,
        title=f"Direct effect on Intermediate AE Heads' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.5])

# %%
top_k = 3
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, all_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %% [markdown]
# ##### By Position

# %%
results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in IAE_HEADS],
    patching_metric=ccs_proj_noising,
    verbose=True,
    apply_metric_to_cache=True,
)
# %%
attn_head_pos_results = einops.rearrange(results['z'], "pos layer head -> (layer head) pos")
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
imshow(attn_head_pos_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
        y=ALL_HEAD_LABELS,
        height=1200,
        width=1200,
        #zmin=-0.1,
        #zmax=0.1,
        title="attn_head_out Path Patching for IAE Heads By Pos")

# %%
