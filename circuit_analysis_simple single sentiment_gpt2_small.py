# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="CbZUo-Tev4QM"
# # Sentiment Analysis for GPT-2

# %% [markdown] id="5vLV3GuDd415"
# ## Setup

# %%
# !ls

# %%
# %cd eliciting-latent-sentiment

# %%
# !pip install git+https://github.com/neelnanda-io/TransformerLens.git
# !pip install circuitsvis
# !pip install jaxtyping==0.2.13
# !pip install einops
# !pip install protobuf==3.20.*
# !pip install plotly
# !pip install torchtyping
# !pip install git+https://github.com/neelnanda-io/neel-plotly.git

# %%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %% id="8QQvkqmWcB2v"
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
from neel_plotly import imshow as imshow_n
from transformer_lens.patching import get_act_patch_attn_head_out_by_pos

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
from neel_plotly import imshow as imshow_n

from utils.visualization import get_attn_head_patterns, plot_attention_heads, scatter_attention_and_contribution_simple


import circuitsvis as cv
from utils.circuit_analysis import get_logit_diff, logit_diff_denoising, logit_diff_noising, get_log_probs

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


# %%
#source_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
#rlhf_model = AutoModelForCausalLM.from_pretrained("curt-tigges/gpt2-negative-movie-reviews")

#hooked_source_model = HookedTransformer.from_pretrained(model_name="gpt2", hf_model=source_model)
#model = HookedTransformer.from_pretrained(model_name="EleutherAI/pythia-410m")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# %% [markdown] id="y5jV1EnY0dpf"
# ## Positive Sentiment Analysis
#

# %% [markdown]
# ### Initial Examination

# %%
example_prompt = "I thought this movie was beautiful, I loved it. \nConclusion: This movie is"
example_answer = " good"

# %%
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=10)

# %%
example_prompt = "I thought this movie was passable, I watched it. \nConclusion: This movie is"
example_prompt = model.to_string(pos_prompts[0][1:])
example_answer = " average"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=10)

# %% [markdown]
# ### Dataset Construction

# %%
from utils.prompts import get_onesided_datasets
prompt_tokens, answer_tokens = get_onesided_datasets(
    model,
    device=device,
    n_answers=3,
    prompt_type="simple",
    dataset_sentiments=["positive", "neutral", "negative"],
    answer_sentiment="positive"
)

# %%
pos_prompts = prompt_tokens["positive"]
neut_prompts = prompt_tokens["neutral"]
neg_prompts = prompt_tokens["negative"]

control_prompts = prompt_tokens["neutral"]
sentiment_prompts = prompt_tokens["positive"]

# %%
for i in range(sentiment_prompts.shape[0]):
    logits, _ = model.run_with_cache(sentiment_prompts[i])
    print(model.to_string(sentiment_prompts[i]))
    print(model.to_string(answer_tokens[i]))
    print(get_log_probs(logits, answer_tokens[i].unsqueeze(0)))

# %%
#from utils.circuit_analysis import get_log_probs
sentiment_logits, sentiment_cache = model.run_with_cache(sentiment_prompts)
sentiment_log_probs = get_log_probs(sentiment_logits, answer_tokens)
sentiment_log_probs

# %%
control_logits, control_cache = model.run_with_cache(control_prompts)
control_log_probs = get_log_probs(control_logits, answer_tokens)
control_log_probs

# %%
from utils.circuit_analysis import get_log_probs

def log_prob_denoising(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch n_pairs"] = answer_tokens,
        flipped_log_prob: float = control_log_probs,
        clean_log_prob: float = sentiment_log_probs,
        return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()
    
def log_prob_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch n_pairs"] = answer_tokens,
        flipped_log_prob: float = control_log_probs,
        clean_log_prob: float = sentiment_log_probs,
        return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - clean_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()
    
log_prob_denoising_tensor = partial(log_prob_denoising, return_tensor=True)
log_prob_noising_tensor = partial(log_prob_noising, return_tensor=True)

# %%

# %% [markdown]
# ### Direct Logit Attribution

# %%
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)
#logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
#print("Logit difference directions shape:", logit_diff_directions.shape)

# %%
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 
final_residual_stream = clean_cache["resid_post", -1]
print("Final residual stream shape:", final_residual_stream.shape)
final_token_residual_stream = final_residual_stream[:, -1, :]
# Apply LayerNorm scaling
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream = clean_cache.apply_ln_to_stack(final_token_residual_stream, layer = -1, pos_slice=-1)

average_logit_diff = einsum("batch d_model, batch d_model -> ", scaled_final_token_residual_stream, logit_diff_directions)/len(all_prompts)
print("Calculated average logit diff:", average_logit_diff.item())
print("Original logit difference:",clean_logit_diff.item())


# %% [markdown]
# #### Logit Lens

# %%
def residual_stack_to_logit_diff(residual_stack: TT["components", "batch", "d_model"], cache: ActivationCache) -> float:
    scaled_residual_stack = clean_cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)/len(all_prompts)


# %%
accumulated_residual, labels = clean_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, clean_cache)
line(logit_lens_logit_diffs, x=np.arange(model.cfg.n_layers*2+1)/2, hover_name=labels, title="Logit Difference From Accumulate Residual Stream")

# %% [markdown]
# #### Layer Attribution

# %%
per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, clean_cache)

line(per_layer_logit_diffs, hover_name=labels, title="Logit Difference From Each Layer")


# %% [markdown]
# #### Head Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

per_head_residual, labels = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, clean_cache)
per_head_logit_diffs = einops.rearrange(per_head_logit_diffs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
per_head_logit_diffs_pct = per_head_logit_diffs/clean_logit_diff
imshow(per_head_logit_diffs_pct, labels={"x":"Head", "y":"Layer"}, title="Logit Difference From Each Head")

# %% [markdown]
# ### Activation Patching

# %% [markdown]
# #### Attention Heads

# %%
results = act_patch(
    model=model,
    orig_input=neut_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode("z"), # iterating over all heads' output in all layers
    patching_metric=log_prob_denoising,
    verbose=True,
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
    orig_input=neut_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
    patching_metric=log_prob_denoising,
    verbose=True,
)

# %%
assert results.keys() == {"z", "q", "k", "v", "pattern"}
assert all([r.shape == (12, 12) for r in results.values()])

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
    orig_input=control_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=log_prob_denoising,
    verbose=True,
)

# %%
assert results.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(pos_prompts[0]))]
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
    margin={"r": 100, "l": 100}
)

# %%
results = act_patch(
    model=model,
    orig_input=control_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode("z", seq_pos="each"),
    patching_metric=log_prob_denoising,
    verbose=True,
)

# %%
results["z"].shape

# %%
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
attn_head_out_act_patch_results = einops.rearrange(results["z"], "pos layer head -> (layer head) pos")
imshow_n(attn_head_out_act_patch_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(pos_prompts[0]))],
        y=ALL_HEAD_LABELS,
        height=800,
        width=800,
        title="attn_head_out Activation Patching By Pos")

# %% [markdown]
# ### Circuit Analysis With Patch Patching & Attn Visualization

# %% [markdown]
# #### Heads Influencing Log Probs

# %%
results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode('z'), # This means iterate over all heads in all layers
    receiver_nodes=Node('resid_post', 11), # This is resid_post at layer 11
    patching_metric=log_prob_noising,
    verbose=True
)

# %%
imshow_p(
    results['z'] * 100,
    title="Direct effect on logit diff (patch from head output -> final resid)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.5])

# %%
top_k = 5
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, sentiment_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %%

from plotly.subplots import make_subplots

# Get the figures
fig1 = scatter_attention_and_contribution_simple(model, (10, 1), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)
fig2 = scatter_attention_and_contribution_simple(model, (10, 4), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)
fig3 = scatter_attention_and_contribution_simple(model, (10, 11), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)
fig4 = scatter_attention_and_contribution_simple(model, (11, 9), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)

# Create subplot
fig = make_subplots(rows=2, cols=2, subplot_titles=("Head 10.1", "Head 10.4", "Head 10.11", "Head 11.9"))

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
DAE_HEADS = [(10, 1), (10, 4), (11, 9), (10, 11)]

results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in DAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on DAE Heads' values)",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.1])

# %% [markdown]
# ##### Key Positions

# %%
results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in DAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
attn_head_pos_results = einops.rearrange(results['z'], "pos layer head -> (layer head) pos")
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
imshow_n(attn_head_pos_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(sentiment_prompts[0]))],
        y=ALL_HEAD_LABELS,
        height=800,
        width=800,
        #zmin=-0.1,
        #zmax=0.1,
        title="attn_head_out Path Patching for DAE Heads By Pos")

# %% [markdown]
# #### Intermediate Attribute Extraction Heads

# %% [markdown]
# ##### Overall

# %%
IAE_HEADS = [(9, 2)]

results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z", seq_pos=17),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in IAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on Intermediate AE Heads' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=5, range_x=[0, 0.1])

# %%
top_k = 2
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, sentiment_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %% [markdown]
# ##### By Position

# %%
results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in IAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
attn_head_pos_results = einops.rearrange(results['z'], "pos layer head -> (layer head) pos")
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
imshow_n(attn_head_pos_results * 100, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(sentiment_prompts[0]))],
        y=ALL_HEAD_LABELS,
        height=800,
        width=800,
        #zmin=-0.1,
        #zmax=0.1,
        title="attn_head_out Path Patching for IAE Heads By Pos")

# %% [markdown]
# ## Negative Sentiment Analysis
#

# %% [markdown]
# ### Initial Examination

# %% [markdown]
#

# %% [markdown]
#

# %%
example_prompt = "I thought this movie was disappointing, I hated it. \nConclusion: This movie is"
example_answer = " dreadful"

# %%
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=10)

# %%
example_prompt = "I thought this movie was passable, I watched it. \nConclusion: This movie is"
example_prompt = model.to_string(pos_prompts[0][1:])
example_answer = " average"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=10)

# %% [markdown]
# ### Dataset Construction

# %%
from utils.prompts import get_onesided_datasets
prompt_tokens, answer_tokens = get_onesided_datasets(
    model,
    device=device,
    n_answers=3,
    prompt_type="simple",
    dataset_sentiments=["positive", "neutral", "negative"],
    answer_sentiment="negative"
)

# %%
pos_prompts = prompt_tokens["positive"]
neut_prompts = prompt_tokens["neutral"]
neg_prompts = prompt_tokens["negative"]

control_prompts = prompt_tokens["neutral"]
sentiment_prompts = prompt_tokens["negative"]

# %%
for i in range(sentiment_prompts.shape[0]):
    logits, _ = model.run_with_cache(sentiment_prompts[i])
    print(model.to_string(sentiment_prompts[i]))
    print(model.to_string(answer_tokens[i]))
    print(get_log_probs(logits, answer_tokens[i].unsqueeze(0)))

# %%
#from utils.circuit_analysis import get_log_probs
sentiment_logits, sentiment_cache = model.run_with_cache(pos_prompts)
sentiment_log_probs = get_log_probs(sentiment_logits, answer_tokens)
sentiment_log_probs

# %%
control_logits, control_cache = model.run_with_cache(neut_prompts)
control_log_probs = get_log_probs(control_logits, answer_tokens)
control_log_probs

# %%
from utils.circuit_analysis import get_log_probs

def log_prob_denoising(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch n_pairs"] = answer_tokens,
        flipped_log_prob: float = control_log_probs,
        clean_log_prob: float = sentiment_log_probs,
        return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()
    
def log_prob_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch n_pairs"] = answer_tokens,
        flipped_log_prob: float = control_log_probs,
        clean_log_prob: float = sentiment_log_probs,
        return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - clean_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()
    
log_prob_denoising_tensor = partial(log_prob_denoising, return_tensor=True)
log_prob_noising_tensor = partial(log_prob_noising, return_tensor=True)

# %%
for i in range(len(all_prompts)):
    logits, _ = model.run_with_cache(all_prompts[i])
    print(all_prompts[i])
    print(get_logit_diff(logits, answer_tokens[i].unsqueeze(0)))

# %% [markdown]
# ### Direct Logit Attribution

# %%
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)
#logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
#print("Logit difference directions shape:", logit_diff_directions.shape)

# %%
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 
final_residual_stream = clean_cache["resid_post", -1]
print("Final residual stream shape:", final_residual_stream.shape)
final_token_residual_stream = final_residual_stream[:, -1, :]
# Apply LayerNorm scaling
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream = clean_cache.apply_ln_to_stack(final_token_residual_stream, layer = -1, pos_slice=-1)

average_logit_diff = einsum("batch d_model, batch d_model -> ", scaled_final_token_residual_stream, logit_diff_directions)/len(all_prompts)
print("Calculated average logit diff:", average_logit_diff.item())
print("Original logit difference:",clean_logit_diff.item())


# %% [markdown]
# #### Logit Lens

# %%
def residual_stack_to_logit_diff(residual_stack: TT["components", "batch", "d_model"], cache: ActivationCache) -> float:
    scaled_residual_stack = clean_cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)/len(all_prompts)


# %%
accumulated_residual, labels = clean_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, clean_cache)
line(logit_lens_logit_diffs, x=np.arange(model.cfg.n_layers*2+1)/2, hover_name=labels, title="Logit Difference From Accumulate Residual Stream")

# %% [markdown]
# #### Layer Attribution

# %%
per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, clean_cache)

line(per_layer_logit_diffs, hover_name=labels, title="Logit Difference From Each Layer")


# %% [markdown]
# #### Head Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

per_head_residual, labels = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, clean_cache)
per_head_logit_diffs = einops.rearrange(per_head_logit_diffs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
per_head_logit_diffs_pct = per_head_logit_diffs/clean_logit_diff
imshow(per_head_logit_diffs_pct, labels={"x":"Head", "y":"Layer"}, title="Logit Difference From Each Head")

# %% [markdown]
# ### Activation Patching

# %% [markdown]
# #### Attention Heads

# %%
results = act_patch(
    model=model,
    orig_input=neut_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode("z"), # iterating over all heads' output in all layers
    patching_metric=log_prob_denoising,
    verbose=True,
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
    orig_input=neut_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
    patching_metric=log_prob_denoising,
    verbose=True,
)

# %%
assert results.keys() == {"z", "q", "k", "v", "pattern"}
assert all([r.shape == (12, 12) for r in results.values()])

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
    orig_input=control_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=log_prob_denoising,
    verbose=True,
)

# %%
assert results.keys() == {"resid_pre", "attn_out", "mlp_out"}
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(pos_prompts[0]))]
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
    margin={"r": 100, "l": 100}
)

# %%
results = act_patch(
    model=model,
    orig_input=control_prompts,
    new_cache=sentiment_cache,
    patching_nodes=IterNode("z", seq_pos="each"),
    patching_metric=log_prob_denoising,
    verbose=True,
)

# %%
results["z"].shape

# %%
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
attn_head_out_act_patch_results = einops.rearrange(results["z"], "pos layer head -> (layer head) pos")
imshow_n(attn_head_out_act_patch_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(pos_prompts[0]))],
        y=ALL_HEAD_LABELS,
        height=800,
        width=800,
        title="attn_head_out Activation Patching By Pos")

# %% [markdown]
# ### Circuit Analysis With Patch Patching & Attn Visualization

# %% [markdown]
# #### Heads Influencing Logit Diff

# %%
results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode('z'), # This means iterate over all heads in all layers
    receiver_nodes=Node('resid_post', 11), # This is resid_post at layer 11
    patching_metric=log_prob_noising,
    verbose=True
)

# %%
imshow_p(
    results['z'] * 100,
    title="Direct effect on logit diff (patch from head output -> final resid)",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=600,
    margin={"r": 100, "l": 100}
)

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.5])

# %%
top_k = 5
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, sentiment_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %%

from plotly.subplots import make_subplots

# Get the figures
fig1 = scatter_attention_and_contribution_simple(model, (7, 1), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)
fig2 = scatter_attention_and_contribution_simple(model, (9, 2), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)
fig3 = scatter_attention_and_contribution_simple(model, (10, 1), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)
fig4 = scatter_attention_and_contribution_simple(model, (10, 4), sentiment_prompts, [6 for _ in range(len(sentiment_prompts))], answer_residual_directions, return_fig=True)

# Create subplot
fig = make_subplots(rows=2, cols=2, subplot_titles=("Head 7.1", "Head 9.2", "Head 10.1", "Head 10.4"))

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
DAE_HEADS = [(9, 1), (10, 4), (11, 9), (10, 11)]

results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in DAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on DAE Heads' values)",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.1])

# %% [markdown]
# ##### Key Positions

# %%
results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in DAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
attn_head_pos_results = einops.rearrange(results['z'], "pos layer head -> (layer head) pos")
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
imshow_n(attn_head_pos_results, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(sentiment_prompts[0]))],
        y=ALL_HEAD_LABELS,
        height=800,
        width=800,
        #zmin=-0.1,
        #zmax=0.1,
        title="attn_head_out Path Patching for DAE Heads By Pos")

# %%

# %% [markdown]
# #### Intermediate Attribute Extraction Heads

# %% [markdown]
# ##### Overall

# %%
IAE_HEADS = [(9, 2)]

results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z", seq_pos=17),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in IAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on Intermediate AE Heads' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
plot_attention_heads(-results['z'].cuda(), top_n=5, range_x=[0, 0.1])

# %%
top_k = 2
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, sentiment_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %% [markdown]
# ##### By Position

# %%
results = path_patch(
    model,
    orig_input=sentiment_prompts,
    new_input=control_prompts,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in IAE_HEADS],
    patching_metric=log_prob_noising,
    verbose=True,
)

# %%
attn_head_pos_results = einops.rearrange(results['z'], "pos layer head -> (layer head) pos")
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
imshow_n(attn_head_pos_results * 100, 
        yaxis="Head Label", 
        xaxis="Pos", 
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(sentiment_prompts[0]))],
        y=ALL_HEAD_LABELS,
        height=800,
        width=800,
        #zmin=-0.1,
        #zmax=0.1,
        title="attn_head_out Path Patching for IAE Heads By Pos")
