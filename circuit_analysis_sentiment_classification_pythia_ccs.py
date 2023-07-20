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
from utils.cache import residual_sentiment_sim_by_head

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
PROMPT_TYPE = "classification_4" # CHANGEME: classification_4
neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, _ = get_ccs_dataset(
    model, device, prompt_type=PROMPT_TYPE
)
#%%
pos_pos_tokens = pos_tokens[gt_labels == 1]
neg_pos_tokens = pos_tokens[gt_labels == 0]
pos_neg_tokens = neg_tokens[gt_labels == 1]
neg_neg_tokens = neg_tokens[gt_labels == 0]
#%%
gt_labels_d_model = einops.repeat(
    gt_labels, "batch -> batch d_model", d_model=model.cfg.d_model
)
#%% # defining clean/corrupt
clean_tokens = torch.cat((pos_pos_tokens, neg_neg_tokens))# CHANGEME
corrupted_tokens = torch.cat((neg_pos_tokens, pos_neg_tokens)) # CHANGEME
ccs_proj_directions = einops.repeat(
    ccs_dir, "d_model -> batch d_model", batch=len(clean_tokens)
)
ccs_signed_directions = ccs_proj_directions * einops.repeat(torch.cat((
    torch.ones(len(pos_pos_tokens)), -torch.ones(len(pos_pos_tokens)) # CHANGEME
)), "batch -> batch d_model", d_model=model.cfg.d_model).to(device=device)
clean_tokens.shape, corrupted_tokens.shape
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
#%%
small_cache = ActivationCache(
    {key: value[:, :-1] for key, value in clean_cache.items()},
    model
)

# %%
def ccs_proj_denoising(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"] = ccs_proj_directions,
    flipped_ccs_proj: float = corrupted_ccs_proj,
    clean_ccs_proj: float = clean_ccs_proj,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of CCS projection, calibrated so that it equals 0 when performance is
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

average_ccs_proj = einsum(
    "batch d_model, batch d_model -> ", 
    scaled_final_token_residual_stream, 
    ccs_proj_directions
) / len(clean_tokens)
print("Calculated average CCS projection:", average_ccs_proj.item())
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
clean_accumulated_residual, labels = clean_cache.accumulated_resid(
    layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
)
clean_logit_lens_ccs_projs = residual_stack_to_ccs_proj(
    clean_accumulated_residual, clean_cache
)
line(
    clean_logit_lens_ccs_projs, 
    x=np.arange(model.cfg.n_layers*1+1), 
    hover_name=labels, 
    title="CCS projection From Accumulated Residual Stream (clean prompts)",
    labels={'x': "Layer", 'y': "CCS projection"},
)
# %%
corrupt_accumulated_residual, labels = corrupted_cache.accumulated_resid(
    layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
)
corrupt_logit_lens_ccs_projs = residual_stack_to_ccs_proj(
    corrupt_accumulated_residual, corrupted_cache
)
line(
    corrupt_logit_lens_ccs_projs, 
    x=np.arange(model.cfg.n_layers*1+1), 
    hover_name=labels, 
    title="CCS projection From Accumulated Residual Stream (corrupt prompts)",
    labels={'x': "Layer", 'y': "CCS projection"},
)

# %% [markdown]
# #### Layer Attribution

# %%
clean_per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
clean_per_layer_ccs_projs = residual_stack_to_ccs_proj(clean_per_layer_residual, clean_cache)

line(
    clean_per_layer_ccs_projs, 
    x=labels,
    hover_name=labels, 
    title="CCS projection From Each Layer (clean)",
    labels={'x': "Layer/Attn-MLP", 'y': "CCS projection"},

)
# %%
corrupt_per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
corrupt_per_layer_ccs_projs = residual_stack_to_ccs_proj(corrupt_per_layer_residual, corrupted_cache)

line(
    corrupt_per_layer_ccs_projs, 
    x=labels,
    hover_name=labels, 
    title="CCS projection From Each Layer (corrupt)",
    labels={'x': "Layer/Attn-MLP", 'y': "CCS projection"},

)

# %% [markdown]
# #### Head Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)
#%%
clean_per_head_residual, labels = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
clean_per_head_ccs_projs = residual_stack_to_ccs_proj(clean_per_head_residual, clean_cache)
clean_per_head_ccs_projs = einops.rearrange(
    clean_per_head_ccs_projs, 
    "(layer head_index) -> layer head_index", 
    layer=model.cfg.n_layers, 
    head_index=model.cfg.n_heads
)
imshow(
    clean_per_head_ccs_projs, 
    labels={"x":"Head", "y":"Layer"}, 
    title="CCS projection From Each Head (clean)"
)
#%%
corrupt_per_head_residual, labels = corrupted_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
corrupt_per_head_ccs_projs = residual_stack_to_ccs_proj(corrupt_per_head_residual, clean_cache)
corrupt_per_head_ccs_projs = einops.rearrange(
    corrupt_per_head_ccs_projs, 
    "(layer head_index) -> layer head_index", 
    layer=model.cfg.n_layers, 
    head_index=model.cfg.n_heads
)
imshow(
    clean_per_head_ccs_projs, 
    labels={"x":"Head", "y":"Layer"}, 
    title="CCS projection From Each Head (corrupt)"
)
# %% [markdown]
# #### Head Attribution (signed)
clean_per_head_ccs_projs_signed = residual_sentiment_sim_by_head(
    small_cache,
    ccs_signed_directions,
    centre_residuals=True,
    normalise_residuals=False,
    layers=model.cfg.n_layers,
    heads=model.cfg.n_heads,
)
imshow(
    clean_per_head_ccs_projs_signed, 
    labels={"x":"Head", "y":"Layer"}, 
    title="Signed CCS projection From Each Head (clean)"
)

# %% [markdown]
# ### Activation Patching

# %% [markdown]
# #### Attention Heads

# %%
head_results = act_patch(
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
    head_results['z'] * 100,
    title="Patching output of attention heads (corrupted -> clean)",
    labels={"x": "Head", "y": "Layer", "color": "CCS proj variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

# %% [markdown]
# #### Head Output by Component

# %%
# iterating over all heads' output in all layers

head_component_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
    patching_metric=ccs_proj_denoising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
assert head_component_results.keys() == {"z", "q", "k", "v", "pattern"}
#assert all([r.shape == (12, 12) for r in results.values()])

imshow_p(
    torch.stack(tuple(head_component_results.values())) * 100,
    facet_col=0,
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Patching output of attention heads (corrupted -> clean)",
    labels={"x": "Head", "y": "Layer", "color": "CCS proj variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1500,
    margin={"r": 100, "l": 100}
)

# %% [markdown]
# #### Residual Stream & Layer Outputs

# %%
# patching at each (layer, sequence position) for each of (resid_pre, attn_out, mlp_out) in turn

attn_mlp_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=ccs_proj_denoising,
    verbose=True,
    apply_metric_to_cache=True,
)

# %%
assert attn_mlp_results.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
imshow_p(
    torch.stack([r.T for r in attn_mlp_results.values()]) * 100, # we transpose so layer is on the y-axis
    facet_col=0,
    facet_labels=["resid_pre", "attn_out", "mlp_out"],
    title="Patching at resid stream & layer outputs (corrupted -> clean)",
    labels={"x": "Sequence position", "y": "Layer", "color": "CCS proj variation"},
    x=labels,
    xaxis_tickangle=45,
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1300,
    zmin=-100,
    zmax=100,
    margin={"r": 100, "l": 100}
)

# %% [markdown]
# #### Heads Influencing CCS Projection

# %%
head_path_results = path_patch(
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
    head_path_results['z'],
    title="Direct effect on CCS projection (patch from head output -> final resid)",
    labels={"x": "Head", "y": "Layer", "color": "CCS proj variation"},
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

#%%
