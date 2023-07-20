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

from utils.visualization import get_attn_head_patterns
from utils.prompts import get_ccs_dataset
from utils.store import load_array, get_labels
from utils.cache import residual_sentiment_sim_by_head

# %%
torch.set_grad_enabled(False)
device = 'cpu' # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROMPT_TYPE = "classification_4"
#%%
MODEL_NAME = "EleutherAI/pythia-1.4b"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
)
model.name = MODEL_NAME
#%%
neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, _ = get_ccs_dataset(
    model, device, prompt_type=PROMPT_TYPE
)
#%%
position_labels = model.to_str_tokens(pos_tokens[0])
position_labels[position_labels.index(" perfect")] = "ADJ1"
position_labels[position_labels.index(" fantastic")] = "ADJ2"
position_labels[position_labels.index(" good")] = "ADJ3"
position_labels[position_labels.index(" good")] = "ADJ4"
position_labels[position_labels.index(" loved")] = "VRB"
position_labels[position_labels.index(".")] = "FS1"
position_labels[position_labels.index(".")] = "FS2"
position_labels[position_labels.index(":")] = "COLON"
position_labels[position_labels.index(" Positive")] = "LABEL"
position_labels = [f"{label} {i}" for i, label in enumerate(position_labels)]
position_labels
#%%
def quad_to_tri_mapper(s: str) -> str:
    s = [int(char) for char in s]
    assert len(s) == 4
    out = ["0"] * 3
    out[0] = int(s[0] == s[2])
    out[1] = int(s[1] == s[3])
    out[2] = int(s[0] + s[1] % 2 == s[2] + s[3] % 2)
    return ''.join([str(1 - i) for i in out])
#%%
# get available experimental results
exp_names = get_labels('ccs_act_patching_z_*', model)
exp_names = [re.sub(r'ccs_act_patching_z_(.*).npy', r'\1', name) for name in exp_names]
exp_names
#%%
review_flip_labels = [e for e in exp_names if quad_to_tri_mapper(e)[0] == '1']
review_const_labels = [e for e in exp_names if quad_to_tri_mapper(e)[0] == '0']
review_flip_labels, review_const_labels
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

#%%
def plot_results(labels: List[str], group: str):
    head_results = sum([load_array(f'ccs_act_patching_z_{label}', model) for label in labels]) / len(labels)
    head_component_results = sum([load_array(f'ccs_act_patching_qkv_{label}', model) for label in labels]) / len(labels)
    attn_mlp_results = sum([load_array(f'ccs_act_patching_attn_mlp_{label}', model) for label in labels]) / len(labels)

    imshow_p(
        head_results,
        title=f"Patching metric for attention heads, averaged over {group}",
        labels={"x": "Head", "y": "Layer", "color": "CCS proj variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=600,
        margin={"r": 100, "l": 100}
    )

    imshow_p(
        head_component_results,
        facet_col=0,
        facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
        title=f"Patching metric for attention heads, averaged over {group}",
        labels={"x": "Head", "y": "Layer", "color": "CCS proj variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1500,
        margin={"r": 100, "l": 100}
    )

    imshow_p(
        attn_mlp_results, # we transpose so layer is on the y-axis
        facet_col=0,
        facet_labels=["resid_pre", "attn_out", "mlp_out"],
        title=f"Patching metric for resid stream & layer outputs, averaged over {group}",
        labels={"x": "Sequence position", "y": "Layer", "color": "CCS proj variation"},
        x=position_labels,
        xaxis_tickangle=45,
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1300,
        zmin=-100,
        zmax=100,
        margin={"r": 100, "l": 100}
    )
#%%
def prettify_exp_string(exp_string: str) -> str:
    exp_string = exp_string.replace('0', '-').replace('1', '+')
    return exp_string[:2] + ' to ' + exp_string[2:]
#%% # plot individual experiments
for label in exp_names:
    plot_results([label, ], prettify_exp_string(label))

#%%
plot_results(review_flip_labels, "review flips")
#%%
plot_results(review_const_labels, "review constants")
#%%
