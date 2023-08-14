# %% [markdown]
# ## Setup

# %%
from IPython.display import display, HTML
import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
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
import plotly.io as pio
import re

from functools import partial

from torchtyping import TensorType as TT

from path_patching import Node, IterNode, path_patch, act_patch

from utils.visualization import plot_attention
from utils.store import load_array
from utils.cache import residual_sentiment_sim_by_head
from utils.circuit_analysis import get_logit_diff

# %%
pio.renderers.default = "notebook"
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
MODEL_NAME = "gpt2-small"
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
sentiment_dir: Float[np.ndarray, "d_model"] = load_array("km_2c_line_embed_and_mlp0", model)
sentiment_dir /= np.linalg.norm(sentiment_dir)
sentiment_dir = torch.tensor(sentiment_dir).to(device=model.cfg.device, dtype=torch.float32)
sentiment_dir.shape

# %% [markdown]
# ### Dataset Construction
# %%
# text = "Here are my honest thoughts. You never fail. You're not bad at all. You will always"
# answers = [" succeed", " fail"]
# text = "Here are my honest thoughts. Don't doubt yourself. You have nothing to fear. You are not wrong. You are very much"
# answers = [" right", " wrong"]
# text = "Don't be sad. You have nothing to be ashamed of. You are a truly"
# answers = [" great", " terrible"]
pos_adjectives = [
    ' mesmerizing',
    ' heartwarming',
    ' captivating',
    ' enlightening',
    ' transcendent',
    ' riveting',
    ' spellbinding',
    ' masterful',
    ' exhilarating',
    ' uplifting',
    ' electrifying',
]
neg_adjectives = [
    ' forgettable',
    ' overwrought',
    ' pretentious',
    ' unimaginative',
    ' disengaging',
    ' incoherent',
    ' contrived',
    ' overrated',
    ' ponderous',
    ' formulaic',
    ' dissonant'
]
for adj in pos_adjectives + neg_adjectives:
    adj_tokens = model.to_str_tokens(adj, prepend_bos=False)
    assert len(adj_tokens) == 2, f"Bad length {len(adj_tokens)} for {adj_tokens}"
text = [
    f"The movie was {adj}. I thought it was" for adj in pos_adjectives + neg_adjectives
]
seq_len = len(model.to_str_tokens(text[0]))
for s in text:
    assert len(model.to_str_tokens(s)) == seq_len, (
        f"Sequence length mismatch: {len(model.to_str_tokens(s))} != {seq_len}"
        f"for {model.to_str_tokens(s)}"
    )
answers = [[" great", " terrible"]] * len(pos_adjectives) + [[" terrible", " great"]] * len(neg_adjectives)
answer_tokens = model.to_tokens(answers, prepend_bos=False)
clean_tokens = model.to_tokens(text)
assert len(clean_tokens) == len(answer_tokens)
answer_tokens = einops.repeat(answer_tokens, "batch correct -> batch 1 correct")
clean_tokens.shape, answer_tokens.shape
#%% # defining clean/corrupt
clean_logits, clean_cache = model.run_with_cache(text)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens=answer_tokens)
#%%
example_prompt_indexed = [f'{i}: {s}' for i, s in enumerate(model.to_str_tokens(text[0]))]
#%%
def name_filter(name: str):
    names = ["resid_pre", "attn_out", "mlp_out", "z", "q", "k", "v", "pattern", "resid_post", "hook_scale"]
    return any([name.endswith(n) for n in names])
#%%
def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    answer_tokens: Int[Tensor, "batch pair correct"], 
    model: HookedTransformer,
    cache: ActivationCache, 
    pos: int = -1,
    biased: bool = False,
):
    scaled_residual_stack: Float[Tensor, "... batch d_model"] = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=pos)
    answer_residual_directions: Float[Tensor, "batch pair correct d_model"] = model.tokens_to_residual_directions(answer_tokens)
    answer_residual_directions = answer_residual_directions.mean(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    batch_logit_diffs: Float[Tensor, "... batch"] = einops.einsum(
        scaled_residual_stack, 
        logit_diff_directions, 
        "... batch d_model, batch d_model -> ... batch",
    )
    if not biased:
        diff_from_unembedding_bias: Float[Tensor, "batch"] = (
            model.b_U[answer_tokens[:, :, 0]] - 
            model.b_U[answer_tokens[:, :, 1]]
        ).mean(dim=1)
        batch_logit_diffs += diff_from_unembedding_bias
    return einops.reduce(batch_logit_diffs, "... batch -> ...", 'mean')
#%%
def cache_to_logit_diff(
    cache: ActivationCache,
    answer_tokens: Int[Tensor, "batch pair correct"], 
    pos: int = -1,
):
    final_residual_stream: Float[Tensor, "batch pos d_model"] = cache["resid_post", -1]
    token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, pos, :]
    return residual_stack_to_logit_diff(
        token_residual_stream, 
        answer_tokens=answer_tokens, 
        model=model,
        cache=cache, 
        pos=pos,
    )
    
# %% [markdown]
# ### Direct Logit Attribution
#%%
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
logit_diff_directions = (answer_residual_directions[:, :, 0] - answer_residual_directions[:, :, 1]).squeeze(1)
answer_residual_directions.shape, logit_diff_directions.shape
# %%[markdown]
#### Position attribution
average_logit_diff = cache_to_logit_diff(clean_cache, answer_tokens, -1)
print("Calculated average Logit difference:", average_logit_diff.item())
print("Original Logit difference:",clean_logit_diff.item())
torch.testing.assert_close(average_logit_diff, clean_logit_diff, rtol=0, atol=1e-4)

#%%
clean_logit_diff_by_pos: Float[Tensor, "pos"] = torch.tensor([
    cache_to_logit_diff(clean_cache, answer_tokens, seq_pos) 
    for seq_pos in range(clean_tokens.shape[1])
])
#%%
line(
    clean_logit_diff_by_pos, 
    x=example_prompt_indexed,
    hover_name=example_prompt_indexed, 
    title="Logit difference At Each Position",
    labels={'x': "Position", 'y': "Logit difference"},
)

# %% [markdown]
# #### Logit Lens
# %%
clean_accumulated_residual: Float[Tensor, "layer 1 d_model"]
clean_accumulated_residual, labels = clean_cache.accumulated_resid(
    layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
)
clean_logit_lens_logit_diffs = residual_stack_to_logit_diff(
    clean_accumulated_residual, answer_tokens, model, clean_cache, biased=True
)
#%%
line(
    clean_logit_lens_logit_diffs, 
    x=np.arange(model.cfg.n_layers*1+1), 
    hover_name=labels, 
    title="Logit difference From Accumulated Residual Stream",
    labels={'x': "Layer", 'y': "Logit difference"},
)

# %% [markdown]
# #### Layer Attribution

# %%
clean_per_layer_residual, labels = clean_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
clean_per_layer_logit_diffs = residual_stack_to_logit_diff(clean_per_layer_residual, answer_tokens, model, clean_cache, biased=True)

# %%
line(
    clean_per_layer_logit_diffs, 
    x=labels,
    hover_name=labels, 
    title="Logit difference From Each Layer",
    labels={'x': "Layer/Attn-MLP", 'y': "Logit difference"},

)

# %% [markdown]
# #### Head Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)
#%%
clean_per_head_residual, labels = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
clean_per_head_logit_diffs = residual_stack_to_logit_diff(clean_per_head_residual, answer_tokens, model, clean_cache, biased=True)
clean_per_head_logit_diffs = einops.rearrange(
    clean_per_head_logit_diffs, 
    "(layer head_index) -> layer head_index", 
    layer=model.cfg.n_layers, 
    head_index=model.cfg.n_heads
)
#%%
imshow(
    clean_per_head_logit_diffs, 
    labels={"x":"Head", "y":"Layer"}, 
    title="Logit difference From Each Head"
)
# %% [markdown]
# #### Neuron Attribution

# %%
def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)
#%%
clean_per_neuron_residual, labels = clean_cache.stack_neuron_results(layer=-1, pos_slice=-1, return_labels=True)
clean_per_neuron_logit_diffs = residual_stack_to_logit_diff(clean_per_neuron_residual, answer_tokens, model, clean_cache, biased=True)
#%%
neuron_df = pd.DataFrame({
    "logit_diff": clean_per_neuron_logit_diffs.cpu(), "neuron": labels
}).sort_values("logit_diff", ascending=False).reset_index(drop=True)
neuron_df.head()
#%%
neuron_df.head(10).style.background_gradient(cmap="RdBu", subset=["logit_diff"]).format({"logit_diff": "{:.2f}"})
#%%
neuron_df.tail(10).style.background_gradient(cmap="RdBu", subset=["logit_diff"]).format({"logit_diff": "{:.2f}"})
#%%
