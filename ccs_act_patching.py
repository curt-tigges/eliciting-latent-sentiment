# %%
import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import yaml

import einops
from fancy_einsum import einsum

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
from utils.store import load_array, save_array
from utils.cache import residual_sentiment_sim_by_head

# %%
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
PROMPT_TYPE = "classification_4"
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
#%%
def shuffle_tensor(tensor: Tensor, dim: int = 0) -> Tensor:
    return tensor[torch.randperm(tensor.shape[dim])]

# %%
neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, _ = get_ccs_dataset(
    model, device, prompt_type=PROMPT_TYPE
)
#%%
torch.manual_seed(0)
pos_pos_tokens = shuffle_tensor(pos_tokens[gt_labels == 1])
neg_pos_tokens = shuffle_tensor(pos_tokens[gt_labels == 0])
pos_neg_tokens = shuffle_tensor(neg_tokens[gt_labels == 1])
neg_neg_tokens = shuffle_tensor(neg_tokens[gt_labels == 0])
#%%
gt_labels_d_model = einops.repeat(
    gt_labels, "batch -> batch d_model", d_model=model.cfg.d_model
)
#%%
def get_clean_corrupt_tokens(exp_string: str) -> Tuple[Tensor, Tensor]:
    if exp_string[:2] == '00':
        corrupted_tokens = neg_neg_tokens
    elif exp_string[:2] == '01':
        corrupted_tokens = neg_pos_tokens
    elif exp_string[:2] == '10':
        corrupted_tokens = pos_neg_tokens
    elif exp_string[:2] == '11':
        corrupted_tokens = pos_pos_tokens
    if exp_string[2:] == '00':
        clean_tokens = neg_neg_tokens
    elif exp_string[2:] == '01':
        clean_tokens = neg_pos_tokens
    elif exp_string[2:] == '10':
        clean_tokens = pos_neg_tokens
    elif exp_string[2:] == '11':
        clean_tokens = pos_pos_tokens
    return clean_tokens, corrupted_tokens
#%%
def get_ccs_proj_base(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"],
):
    final_residual_stream: Float[
        Tensor, "batch pos d_model"
    ] = cache["resid_post", -1]
    batch_size, seq_len, d_model = final_residual_stream.shape
    final_token_residual_stream: Float[
        Tensor, "batch d_model"
    ] = final_residual_stream[:, -1, :]
    # Apply LayerNorm scaling
    # pos_slice is the subset of the positions we take - 
    # here the final token of each prompt
    scaled_final_token_residual_stream: Float[
        Tensor, "batch d_model"
    ] = cache.apply_ln_to_stack(
        final_token_residual_stream, layer = -1, pos_slice=-1
    )
    average_ccs_proj = einsum(
        "batch d_model, batch d_model -> ", 
        scaled_final_token_residual_stream, 
        directions
    ) / batch_size
    return average_ccs_proj
#%% 
def ccs_proj_denoising_base(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"],
    flipped_ccs_proj: float,
    clean_ccs_proj: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of CCS projection, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_ccs_proj = get_ccs_proj_base(cache, directions)
    ld = (
        (patched_ccs_proj - flipped_ccs_proj) / 
        (clean_ccs_proj  - flipped_ccs_proj)
    )
    if return_tensor:
        return ld
    else:
        return ld.item()


def ccs_proj_noising_base(
    cache: ActivationCache,
    directions: Float[Tensor, "batch d_model"],
    clean_ccs_proj: float,
    corrupted_ccs_proj: float,
    return_tensor: bool = False,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_ccs_proj = get_ccs_proj_base(cache, directions)
    ld = ((patched_ccs_proj - clean_ccs_proj) / (clean_ccs_proj - corrupted_ccs_proj))

    if return_tensor:
        return ld
    else:
        return ld.item()
#%%
def check_patching_result(
    array: Float[Tensor, "facet layer head"],
):
    '''
    Checks 2 or 3 dimensional array of percentages from act patching.
    '''
    if array.ndim == 2:
        array = array.unsqueeze(0)
    zero = torch.tensor(0, device=device, dtype=torch.float32)
    assert torch.isfinite(array).all()
    assert torch.isclose(array[0, 0, 0], zero, rtol=0, atol=1), (
        f"Expected first element to be 0, got {array[0, 0, 0]}"
    )
#%%
def run_act_patching(
    model: HookedTransformer,
    corrupted_tokens: Tensor,
    clean_cache: ActivationCache,
    ccs_proj_denoising: Callable[[ActivationCache], float],
    exp_label: str,
):
    batch_size, seq_len = corrupted_tokens.shape
    positions_before_last = einops.repeat(
        torch.arange(seq_len - 1), 'pos -> batch pos', batch=batch_size
    )
    head_results_except_last: Float[Tensor, "layer head"] = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=clean_cache,
        patching_nodes=IterNode("z", seq_pos=positions_before_last), # iterating over all heads' output in all layers
        patching_metric=ccs_proj_denoising,
        verbose=False,
        apply_metric_to_cache=True,
    )['z'] * 100
    check_patching_result(head_results_except_last)
    head_results_last: Float[Tensor, "layer head"] = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=clean_cache,
        patching_nodes=IterNode("z", seq_pos=seq_len - 1), # iterating over all heads' output in all layers
        patching_metric=ccs_proj_denoising,
        verbose=False,
        apply_metric_to_cache=True,
    )['z'] * 100
    check_patching_result(head_results_last)
    # stack head results with new dim 0
    head_results: Float[Tensor, "pos layer head"] = torch.stack(
        (head_results_except_last, head_results_last), dim=0
    )
    check_patching_result(head_results)
    save_array(
        head_results,
        f'ccs_act_patching_z_{exp_label}',
        model
    )

    head_component_results = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=clean_cache,
        patching_nodes=IterNode(["z", "q", "k", "v", "pattern"]),
        patching_metric=ccs_proj_denoising,
        verbose=False,
        apply_metric_to_cache=True,
    )
    assert head_component_results.keys() == {"z", "q", "k", "v", "pattern"}
    head_components_stacked: Float[Tensor, "qkv layer head"] = torch.stack(tuple(head_component_results.values())) * 100
    check_patching_result(head_components_stacked)
    save_array(
        head_components_stacked,
        f'ccs_act_patching_qkv_{exp_label}',
        model
    )

    attn_mlp_results = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=clean_cache,
        patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
        patching_metric=ccs_proj_denoising,
        verbose=False,
        apply_metric_to_cache=True,
    )
    assert attn_mlp_results.keys() == {"resid_pre", "attn_out", "mlp_out"}
    attn_mlp_stacked: Float[Tensor, "mlp_attn layer head"] = torch.stack([r.T for r in attn_mlp_results.values()]) * 100
    check_patching_result(attn_mlp_stacked)
    save_array(
        attn_mlp_stacked,
        f'ccs_act_patching_attn_mlp_{exp_label}',
        model
    )
#%%
# compute baselines for denominator of patching metric
correct_tokens = torch.cat((pos_pos_tokens, neg_neg_tokens))
incorrect_tokens = torch.cat((pos_neg_tokens, neg_pos_tokens))
ccs_proj_directions = einops.repeat(
        ccs_dir, "d_model -> batch d_model", batch=len(correct_tokens)
)
# run model on clean prompts
_, correct_cache = model.run_with_cache(correct_tokens, return_type=None)
correct_ccs_proj = get_ccs_proj_base(correct_cache, directions=ccs_proj_directions)
# run model on corrupted prompts
_, incorrect_cache = model.run_with_cache(incorrect_tokens, return_type=None)
incorrect_ccs_proj = get_ccs_proj_base(incorrect_cache, directions=ccs_proj_directions)
correct_ccs_proj, incorrect_ccs_proj
#%%
def prettify_exp_string(exp_string: str) -> str:
    exp_string = exp_string.replace('0', '-').replace('1', '+')
    return exp_string[:2] + ' to ' + exp_string[2:]
#%%
def run_experiment(idx: int):
    exp_string = format(idx, '04b')
    print(prettify_exp_string(exp_string))
    clean_tokens, corrupted_tokens = get_clean_corrupt_tokens(exp_string)
    ccs_proj_directions = einops.repeat(
        ccs_dir, "d_model -> batch d_model", batch=len(clean_tokens)
    )
    # run model on clean prompts
    _, clean_cache = model.run_with_cache(clean_tokens, return_type=None)
    clean_ccs_proj = get_ccs_proj_base(clean_cache, directions=ccs_proj_directions)
    # run model on corrupt prompts
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, return_type=None)
    corrupted_ccs_proj = get_ccs_proj_base(corrupted_cache, directions=ccs_proj_directions)
    # set uniform scale across experiments
    proj_diff_sign = 1 if clean_ccs_proj - corrupted_ccs_proj >= 0 else -1
    clean_ccs_proj_rescaled = (
        corrupted_ccs_proj + torch.abs(correct_ccs_proj - incorrect_ccs_proj) * proj_diff_sign
    )
    # print(clean_ccs_proj, corrupted_ccs_proj, clean_ccs_proj_rescaled)
    # define patching metric
    ccs_proj_denoising = partial(
        ccs_proj_denoising_base,
        directions=ccs_proj_directions,
        flipped_ccs_proj=corrupted_ccs_proj,
        clean_ccs_proj=clean_ccs_proj_rescaled,
    )
    run_act_patching(
        model=model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        ccs_proj_denoising=ccs_proj_denoising,
        exp_label=exp_string,
    )
#%%
# N.B. act patching below is corrupted -> clean, i.e. "denoising"
for exp_idx in tqdm(range(4, 6)):
    run_experiment(exp_idx)
#%%
