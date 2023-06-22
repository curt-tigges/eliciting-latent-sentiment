# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="CbZUo-Tev4QM"
# # Initial Exploratory Analysis

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
from typing import List, Optional, Union

import torch
import numpy as np

import einops
from fancy_einsum import einsum

import circuitsvis as cv
import transformer_lens
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
from plotly.subplots import make_subplots

from functools import partial

from torchtyping import TensorType as TT

from path_patching import Node, IterNode, path_patch, act_patch
import prompt_utils

from visualization_utils import get_attn_head_patterns

# %%
torch.set_grad_enabled(False)
device = 'cpu'
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device=device,
)

# %% [markdown]
# ### Initial Examination

# %%
example_prompt = "I thought this movie was lousy, I hated it. \nConclusion: This movie is"
example_answer = " terrible"

# %%
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=10)

# %%
example_prompt = "I thought this movie was perfect, I loved it. \nConclusion: This movie is"
example_answer = " bad"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, top_k=10)

# %% [markdown]
# ### Dataset Construction
# %%
(
    all_prompts, answer_tokens, clean_tokens, corrupted_tokens
) = prompt_utils.get_dataset(
    model, device, n_pairs=5
)
# %%
for i in range(len(all_prompts)):
    logits, _ = model.run_with_cache(all_prompts[i])
    print(all_prompts[i])
    print(prompt_utils.get_logit_diff(logits, answer_tokens[i].unsqueeze(0)))

# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logit_diff = prompt_utils.get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
clean_logit_diff

# %%
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
corrupted_logit_diff = prompt_utils.get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
corrupted_logit_diff


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
    patched_logit_diff = prompt_utils.get_logit_diff(logits, answer_tokens)
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
        patched_logit_diff = prompt_utils.get_logit_diff(logits, answer_tokens)
        return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()


# %% [markdown] id="TfiWnZtelFMV"
# ### Direct Logit Attribution

# %% colab={"base_uri": "https://localhost:8080/"} id="bt_jzrazlMAK" outputId="39683745-1153-4a0f-bdbf-5f3be977abe3"
answer_residual_directions: Float[
    Tensor, "batch n_pairs 2 d_model"
] = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)
logit_diff_directions: Float[Tensor, "batch d_model"] = (
    answer_residual_directions[:, :, 0, :] - 
    answer_residual_directions[:, :, 1, :]
).mean(dim=1)
print("Logit difference directions shape:", logit_diff_directions.shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="LsDE7VUGIX8l" outputId="226c2ad4-fb5b-44f4-b872-1d06eee7cbd5"
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


# %% [markdown] id="Nb2nC45lIohT"
# #### Logit Lens

# %% id="DvRDK2krIrid"
def residual_stack_to_logit_diff(residual_stack: TT["components", "batch", "d_model"], cache: ActivationCache) -> float:
    scaled_residual_stack = clean_cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)/len(all_prompts)


# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="7vxP1pNuPMhr" outputId="616ac0ef-ddd2-4b1e-bccd-8ee3a3ebce23"
accumulated_residual, labels = clean_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, clean_cache)
line(logit_lens_logit_diffs, x=np.arange(model.cfg.n_layers*2+1)/2, hover_name=labels, title="Logit Difference From Accumulate Residual Stream")

# %% [markdown] id="s60emfYIbTuT"
# #### Layer Attribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="yGgAVYgIJi9Z" outputId="2d6b1ffe-b701-419d-a786-24f0d24d2b54"
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
imshow(per_head_logit_diffs, labels={"x":"Head", "y":"Layer"}, title="Logit Difference From Each Head")

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
    patching_metric=logit_diff_denoising,
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
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
    patching_metric=logit_diff_denoising,
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
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    patching_metric=logit_diff_denoising,
    verbose=True,
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
    margin={"r": 100, "l": 100}
)

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
    receiver_nodes=Node('resid_post', 11), # This is resid_post at layer 11
    patching_metric=logit_diff_noising,
    verbose=True
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
from visualization_utils import (
    plot_attention_heads,
    scatter_attention_and_contribution
)

import circuitsvis as cv

# %%
plot_attention_heads(-results['z'].cuda(), top_n=15, range_x=[0, 0.5])

# %%
from visualization_utils import get_attn_head_patterns

top_k = 5
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, all_prompts[21], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %%
heads

# %%
from visualization_utils import scatter_attention_and_contribution_sentiment

from plotly.subplots import make_subplots

# Get the figures
fig1 = scatter_attention_and_contribution_sentiment(model, (7, 1), all_prompts, [6 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)
fig2 = scatter_attention_and_contribution_sentiment(model, (9, 2), all_prompts, [6 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)
fig3 = scatter_attention_and_contribution_sentiment(model, (10, 1), all_prompts, [6 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)
fig4 = scatter_attention_and_contribution_sentiment(model, (10, 4), all_prompts, [6 for _ in range(len(all_prompts))], answer_residual_directions, return_fig=True)

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
fig.update_layout(height=600, title_text="Sentiment-Attender Heads")

# Update axes labels
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text="Attn Prob on Word", row=i, col=j)
        fig.update_yaxes(title_text="Dot w Sentiment Embed", row=i, col=j)

fig.show()


# %% [markdown]
# Observations: There appear to be at least three types of heads contributing to the final logit difference: 
# - UNIVERSAL SENTIMENT ATTENDERS: One type attends to the sentiment word--"perfect" in this case--and writes out in the direction of the positive or negative class, regardless of whether the word is positive or negative. L10H4 is the biggest example of this.
# - NEGATIVE SENTIMENT ATTENDERS: Another type seems to focus mostly on negative-sentiment words, and then writes out in the direction of the correct class when the answer is negative (but not in the positive case). L9H2 is the primary example. L10H1 seems to be midway between L10H4 and L9H2 in its behavior. L7H1 seems to do this in a weaker way.
# - POSITIVITY BOOSTERS: L8H5 displays interestingly different behavior. It does not attend to the sentiment word, but writes out in the positive or negative class direction--but much more when the sentiment word is positive.

# %% [markdown]
# #### Heads Influencing Sentiment-Attenders

# %% [markdown]
# ##### Overall

# %%
SENTIMENT_ATTENDERS = [(10, 4), (10, 1), (9, 2), (7, 1)]

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in SENTIMENT_ATTENDERS],
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on Sentiment Attenders' values)",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %% [markdown]
# ##### Key Positions

# %%
results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z", seq_pos="each"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in SENTIMENT_ATTENDERS],
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
results["z"].shape

# %%


for i in [6, 17]:
    imshow_p(
        results["z"][i][:10] * 100,
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
# #### SJB1-Attenders

# %%
MOVIE_ATTENDERS = [(7, 1), (7, 5), (9, 2)]

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z"),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in MOVIE_ATTENDERS],
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on Movie Attenders' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %%
top_k = 2
top_heads = torch.topk(-results['z'].flatten(), k=top_k).indices.cpu().numpy()
heads = [(head // model.cfg.n_heads, head % model.cfg.n_heads) for head in top_heads]
tokens, attn, names = get_attn_head_patterns(model, all_prompts[0], heads)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)

# %%

# %% [markdown]
# #### SJB2-Attenders

# %%
SBJ2_ATTENDERS = [(8, 5), (9, 2), (9, 10)]

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z", seq_pos=17),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in SBJ2_ATTENDERS],
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on SBJ2 Attenders' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %% [markdown]
# #### ADJ-Attenders

# %%
SBJ2_ATTENDERS = [(8, 5), (9, 2), (9, 10)]

results = path_patch(
    model,
    orig_input=clean_tokens,
    new_input=corrupted_tokens,
    sender_nodes=IterNode("z", seq_pos=17),
    receiver_nodes=[Node("v", layer, head=head) for layer, head in SBJ2_ATTENDERS],
    patching_metric=logit_diff_noising,
    verbose=True,
)

# %%
imshow_p(
        results["z"][:10] * 100,
        title=f"Direct effect on SBJ2 Attenders' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=700,
        margin={"r": 100, "l": 100}
    )

# %% [markdown]
# #### Various Attention Patterns

# %%
heads = [(0, 4), (3, 11), (5, 7), (6, 4), (6, 7), (6,11), (7, 1), (7, 5), (8, 4), (8, 5), (9, 2), (9, 10), (10, 1), (10, 4), (11, 9)]
attn_list = []
for i in range(len(all_prompts)):
    tokens, attn, names = get_attn_head_patterns(model, all_prompts[i], heads)
    attn_list.append(attn)
tokens, _, names = get_attn_head_patterns(model, all_prompts[0], heads)
attn = torch.stack(attn_list, dim=0).mean(dim=0)
cv.attention.attention_heads(tokens=tokens, attention=attn, attention_head_names=names)
