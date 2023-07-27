# %% [markdown] id="5vLV3GuDd415"
# ## Setup

# %%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %% id="8QQvkqmWcB2v"
import os
import pathlib
from typing import Iterable, List, Optional, Union

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

from torch import Tensor
from tqdm.notebook import tqdm
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from rich import print as rprint

from typing import List, Union
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import re

from functools import partial

from torchtyping import TensorType as TT

from path_patching import Node, IterNode, path_patch, act_patch

from utils.visualization import get_attn_head_patterns
from utils.prompts import get_dataset
from utils.circuit_analysis import get_logit_diff, logit_diff_denoising, logit_diff_noising
from utils.store import load_array

# %%
pio.renderers.default = "notebook"
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
POSITIVE_PROMPTS = False
POSITIVE_DIRECTION = True

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


# %% id="0c0JbzPpI0-D"
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


# %% [markdown] id="y5jV1EnY0dpf"
# ## Exploratory Analysis
#

# %% colab={"base_uri": "https://localhost:8080/"} id="bjeWvBNOn2VT" outputId="dff069ed-56d5-414d-d649-2c70f073b1fc"
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
model.name = MODEL_NAME

# %% [markdown]
# ### Dataset Construction
#%%
def embed_and_mlp0(
    tokens: Int[Tensor, "batch 1"],
    transformer: HookedTransformer = model
):
    block0 = transformer.blocks[0]
    resid_mid = transformer.embed(tokens)
    mlp_out = block0.mlp((resid_mid))
    resid_post = resid_mid + mlp_out
    return block0.ln2(resid_post)
#%%
if POSITIVE_DIRECTION:
    ablate_dir = load_array('km_2c_line_embed_and_mlp0', model)
else:
    ablate_dir = load_array('rotation_direction0', model)
ablate_dir /= np.linalg.norm(ablate_dir)
ablate_dir = torch.from_numpy(ablate_dir).to(dtype=torch.float32, device=device)
# %%
_, pos_neg_answers, pos_neg_tokens, _ = get_dataset(
    model, device, n_pairs=1, 
    comparison=("positive", "negative",)
)
#%%
example_prompt = model.to_str_tokens(pos_neg_tokens[0])
adjective_token = 6
verb_token = 9
example_prompt_indexed = [f'{i}: {s}' for i, s in enumerate(example_prompt)]
seq_len = len(example_prompt)
print(example_prompt_indexed)
#%%
pos_neg_embeddings: Float[Tensor, "batch pos d_model"] = embed_and_mlp0(
    pos_neg_tokens, model
).to(torch.float32)
#%%
def compute_mean_projection(
    embeddings: Float[Tensor, "batch pos d_model"],
    direction: Float[Tensor, "d_model"], 
    tokens: Iterable[int] = (adjective_token, verb_token),
) -> Float[Tensor, ""]:
    assert torch.isclose(direction.norm(), torch.tensor(1.0), rtol=0, atol=.001)
    projections: Float[Tensor, "batch"] = einops.einsum(
        embeddings, direction, "b s d, d -> b s"
    )
    return projections[:, tokens].mean().item()
#%%
print(compute_mean_projection(pos_neg_embeddings[::2], ablate_dir))
print(compute_mean_projection(pos_neg_embeddings[1::2], ablate_dir))
#%%
average_projection: float = compute_mean_projection(pos_neg_embeddings, ablate_dir)
average_projection
#%%
if POSITIVE_PROMPTS:
    orig_tokens = pos_neg_tokens[::2]
    answer_tokens = pos_neg_answers[::2]
    orig_embeddings = pos_neg_embeddings[::2]
else:
    orig_tokens = pos_neg_tokens[1::2]
    answer_tokens = pos_neg_answers[1::2]
    orig_embeddings = pos_neg_embeddings[1::2]
print(model.to_str_tokens(orig_tokens[0]))
print(model.to_str_tokens(answer_tokens[0]))
#%%
#%%
def mean_ablate_direction(
    input: Float[Tensor, "batch pos d_model"],
    direction: Float[Tensor, "d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
    average_projection: float = average_projection,
    multiplier: float = 1.0,
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Designed for use on hook_resid_post
    '''
    assert torch.isclose(direction.norm(), torch.tensor(1.0), rtol=0, atol=.001)
    assert input.ndim == 3
    proj: Float[Tensor, "batch pos"] = einops.einsum(
        input, direction, "b p d, d -> b p"
    )
    avg_broadcast: Float[Tensor, "batch pos"] = einops.repeat(
        torch.tensor(average_projection, device=device), 
        " -> b p", 
        b=input.shape[0], p=input.shape[1]
    )
    proj_diff: Float[Tensor, "batch pos 1"] = (
        avg_broadcast - proj
    )[:, tokens].unsqueeze(dim=-1)
    input[:, tokens, :] += multiplier * proj_diff * direction
    return input
#%%
def ablation_hook_base(
    input: Float[Tensor, "batch pos head d_model"],
    hook: HookPoint,
    tokens: Iterable[int] = (adjective_token, verb_token),
    layer: Optional[int] = None,
    multiplier: float = 1.0,
) -> Float[Tensor, "batch pos head d_model"]:
    assert 'hook_resid_post' in hook.name
    if layer is not None and hook.layer() != layer:
        return input
    return mean_ablate_direction(
        input, ablate_dir, tokens=tokens, multiplier=multiplier
    )
#%%
ablation_hook = partial(
    ablation_hook_base, layer=0, tokens=np.arange(orig_tokens.shape[1]),
)
#%%
def maybe_add_hook(
    model: HookedTransformer,
    ablate: bool
):
    model.reset_hooks()
    if ablate:
        model.add_hook(lambda name: name.endswith('resid_post'), ablation_hook)

# %% [markdown] id="TfiWnZtelFMV"
# ### Direct Logit Attribution
#%%
def cache_to_logit_diff(
    cache: ActivationCache
):
    final_residual_stream: Float[Tensor, "batch pos d_model"] = cache["resid_post", -1]
    final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]
    scaled_residual_stack: Float[Tensor, "components batch d_model"] = cache.apply_ln_to_stack(final_token_residual_stream, layer = -1, pos_slice=-1)
    answer_residual_directions: Float[Tensor, "batch pair correct d_model"] = model.tokens_to_residual_directions(answer_tokens)
    answer_residual_directions = answer_residual_directions.mean(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    diff_from_unembedding_bias: Float[Tensor, "batch"] = (
        model.b_U[answer_tokens[:, :, 0]] - 
        model.b_U[answer_tokens[:, :, 1]]
    ).mean(dim=1)
    prod = scaled_residual_stack * logit_diff_directions
    return (prod.sum(dim=-1) + diff_from_unembedding_bias).mean()
#%% [markdown]
#### Sanity check ablation
#%%
batch_index = 0
pair_index = 0
answer_index = 0
top_k = 10
example_prompt = model.to_string(orig_tokens[batch_index])
example_answer = model.to_string(answer_tokens[
    batch_index, pair_index, answer_index
])
# %%
print("Without ablation:")
maybe_add_hook(model, ablate=False)
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=top_k)
#%%
print("With ablation:")
maybe_add_hook(model, ablate=True)
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=top_k)
# %%
print("Without ablation:")
maybe_add_hook(model, ablate=False)
clean_logits, clean_cache = model.run_with_cache(orig_tokens)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
clean_logit_diff
# %%
print("With ablation:")
maybe_add_hook(model, ablate=True)
corrupted_logits, corrupted_cache = model.run_with_cache(orig_tokens)
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
corrupted_logit_diff
# %%
model.reset_hooks()
average_logit_diff = cache_to_logit_diff(clean_cache)
print("Calculated average logit diff:", average_logit_diff.item())
print("Original logit difference:",clean_logit_diff.item())
# %%
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

def logit_diff_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float = clean_logit_diff,
        corrupted_logit_diff: float = corrupted_logit_diff,
        answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    ) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()

#%%
def residual_stack_to_logit_diff(
    residual_stack: TT["components", "batch", "d_model"], cache: ActivationCache
) -> float:
    answer_residual_directions: Float[Tensor, "batch pair correct d_model"] = model.tokens_to_residual_directions(answer_tokens)
    answer_residual_directions = answer_residual_directions.mean(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    scaled_residual_stack: Float[Tensor, "components batch d_model"] = cache.apply_ln_to_stack(
        residual_stack, layer = -1, pos_slice=-1
    )
    diff_from_unembedding_bias: Float[Tensor, "batch"] = (
        model.b_U[answer_tokens[:, :, 0]] - 
        model.b_U[answer_tokens[:, :, 1]]
    ).mean(dim=1)
    prod: Float[Tensor, "components batch d_model"] = scaled_residual_stack * logit_diff_directions
    logit_diff_per_prompt: Float[Tensor, "components batch"] = prod.sum(dim=-1)# + diff_from_unembedding_bias
    return logit_diff_per_prompt.mean(dim=-1)
# %% [markdown] id="Nb2nC45lIohT"
# #### Logit Lens

# %% 
clean_accumulated_residual, labels = clean_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
clean_accumulated_logit_diffs = residual_stack_to_logit_diff(clean_accumulated_residual, clean_cache)
corrupted_accumulated_residual, labels = corrupted_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
corrupted_accumulated_logit_diffs = residual_stack_to_logit_diff(corrupted_accumulated_residual, corrupted_cache)
#%%
line(
    clean_accumulated_logit_diffs - corrupted_accumulated_logit_diffs, 
    x=np.arange(model.cfg.n_layers*2+1)/2, 
    hover_name=labels, 
    title="Logit Difference From Accumulate Residual Stream (clean - corrupt)",
    labels={"x":"Layer", "y":"Logit Difference"},
)


# %% [markdown] id="s60emfYIbTuT"
# #### Layer Attribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="yGgAVYgIJi9Z" outputId="2d6b1ffe-b701-419d-a786-24f0d24d2b54"
clean_per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
clean_per_layer_logit_diffs = residual_stack_to_logit_diff(clean_per_layer_residual, clean_cache)
corrupted_per_layer_residual, labels = corrupted_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
corrupted_per_layer_logit_diffs = residual_stack_to_logit_diff(corrupted_per_layer_residual, corrupted_cache)

line(
    clean_per_layer_logit_diffs - corrupted_per_layer_logit_diffs, 
    hover_name=labels, 
    title="Logit Difference From Each Layer (clean - corrupt)",
    x=labels,
)


# %% [markdown]
# #### Head Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

clean_per_head_residual, labels = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
clean_per_head_logit_diffs = residual_stack_to_logit_diff(clean_per_head_residual, clean_cache)
clean_per_head_logit_diffs = einops.rearrange(clean_per_head_logit_diffs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
corrupted_per_head_residual, labels = corrupted_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
corrupted_per_head_logit_diffs = residual_stack_to_logit_diff(corrupted_per_head_residual, corrupted_cache)
corrupted_per_head_logit_diffs = einops.rearrange(corrupted_per_head_logit_diffs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
#%%
imshow(clean_per_head_logit_diffs - corrupted_per_head_logit_diffs, labels={"x":"Head", "y":"Layer"}, title="Logit Difference From Each Head (clean - corrupt)")

# %% [markdown]
# ### Activation Patching

# %% [markdown]
# #### Attention Heads

# %%
model.reset_hooks()
results = act_patch(
    model=model,
    orig_input=orig_tokens,
    new_cache=corrupted_cache,
    patching_nodes=IterNode("z"), # iterating over all heads' output in all layers
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
imshow_p(
    results['z'] * 100,
    title="Patching output of attention heads (clean -> corrupted)",
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
model.reset_hooks()
results = act_patch(
    model=model,
    orig_input=orig_tokens,
    new_cache=corrupted_cache,
    patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
assert results.keys() == {"z", "q", "k", "v", "pattern"}
assert all([r.shape == (12, 12) for r in results.values()])

imshow_p(
    torch.stack(tuple(results.values())) * 100,
    facet_col=0,
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Patching output of attention heads (clean -> corrupted)",
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
model.reset_hooks()
results = act_patch(
    model=model,
    orig_input=orig_tokens,
    new_cache=corrupted_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
assert results.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(orig_tokens[0]))]
imshow_p(
    torch.stack([r.T for r in results.values()]) * 100, # we transpose so layer is on the y-axis
    facet_col=0,
    facet_labels=["resid_pre", "attn_out", "mlp_out"],
    title="Patching at resid stream & layer outputs (clean -> corrupted)",
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
