#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import einops
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Union, Optional
from jaxtyping import Float, Int
from typeguard import typechecked
from torch import Tensor
from functools import partial
import copy
import os
import itertools
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import wandb
#%% [markdown]
#### Model loading
#%%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    # center_unembed=True,
    # center_writing_weights=True,
    # fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% [markdown]
#### Data generation
# %%
utils.test_prompt('I thought the movie was great. I', ' loved', model)
# %%
utils.test_prompt('I thought the movie was great. I', ' hated', model)
# %%
utils.test_prompt('I thought the movie was terrible. I', ' hated', model)
# %%
utils.test_prompt('I thought the movie was terrible. I', ' loved', model)
#%%
def write_adjective_lengths_to_file():
    possible_adjectives = [
        "good",
        "bad",
        "great",
        "terrible",
        "horrible",
        "very bad",
        "very good",
        "fantastic",
        "excellent",
        "amazing",
        "superb",
        "wonderful",
        "awful",
        "exceptional",
        "fabulous",
        "spectacular",
        "marvelous",
        "outstanding", 
        "horrendous",
        "dire",
        "dull",
        "boring",
        "abysmal",
        "lousy",
        "disappointing",
    ]
    adjective_token_lengths = dict()
    for adj in possible_adjectives:
        adj_len = model.to_tokens(' ' + adj, prepend_bos=False).shape[1]
        if adj_len not in adjective_token_lengths:
            adjective_token_lengths[adj_len] = [adj]
        else:
            adjective_token_lengths[adj_len].append(adj)
    with open('adjective_token_lengths.txt', 'w') as f:
        f.write(str(adjective_token_lengths))
#%%
write_adjective_lengths_to_file()
#%%
def check_lengths(adjectives: List[str]):
    for adj in adjectives:
        token_len = len(model.to_tokens(' ' + adj, prepend_bos=False)[0, :])
        assert token_len == 1, f"Adjective '{adj}' had bad length: {token_len}"
#%%
# N.B. using single token adjectives
positive_adjectives = [
    "good",
    "great",
    "fantastic",
    "excellent",
    "amazing"
]
negative_adjectives = [
    "terrible",
    "horrible",
    "awful",
    "horrendous",
    "dull",
]
adjectives = positive_adjectives + negative_adjectives
sentiments = [1] * len(positive_adjectives) + [-1] * len(negative_adjectives)
question = "Did I like the movie?"
clean_prompts = [
    f"I thought the movie was {adjective}. {question}" 
    for adjective in adjectives
]
corrupted_prompts = [
    f"I thought the movie was {adjective}. {question}"
    for adjective in adjectives[::-1]
]
# Define the answers for each prompt, in the form (correct, incorrect)
answers = [(' Yes', ' No')[::sent] for sent in sentiments]
# Define the answer tokens (same shape as the answers)
answer_tokens = torch.concat([
    model.to_tokens(names, prepend_bos=False).T for names in answers
])
# Assert that all adjectives consist of 2 tokens
check_lengths(adjectives)
#%%
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / 
    # subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(
        dim=-1, index=answer_tokens
    )
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()
#%%
clean_tokens: Int[Tensor, "batch pos"] = model.to_tokens(
    clean_prompts, prepend_bos=True
).to(device)
corrupted_tokens: Int[Tensor, "batch pos"] = model.to_tokens(
    corrupted_prompts, prepend_bos=True
).to(device)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_differences = logits_to_ave_logit_diff(
    clean_logits, answer_tokens, per_prompt=True
)
corrupted_logit_differences = logits_to_ave_logit_diff(
    corrupted_logits, answer_tokens, per_prompt=True
)

clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
assert (clean_tokens[:, -1] != model.tokenizer.pad_token_id).all()
assert (corrupted_tokens[:, -1] != model.tokenizer.pad_token_id).all()
#%%
print(f"Clean logit diff: {clean_logit_diff:.4f}")
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
#%%
print(f"Clean logit differences: {clean_logit_differences}")
print(f"Corrupted logit differences: {corrupted_logit_differences}")
#%%
prompt_idx = 0
print('answers', answers[prompt_idx])
print('answer tokens', answer_tokens[prompt_idx])
print('clean prompt', clean_prompts[prompt_idx])
print('clean tokens', clean_tokens[prompt_idx])
print('clean logit', clean_logits[prompt_idx, -1, answer_tokens[prompt_idx][0]])
print('clean logit diff', clean_logit_differences[prompt_idx])
utils.test_prompt(clean_prompts[prompt_idx], answers[prompt_idx][0], model)
# %% [markdown]
#### Logit lens
answer_residual_directions: Float[
    Tensor, "batch 2 d_model"
] = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)

(
    correct_residual_directions, 
    incorrect_residual_directions
) = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[
    Tensor, "batch d_model"
] = correct_residual_directions - incorrect_residual_directions
#%%
def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size
#%%
accumulated_residual, residual_labels = clean_cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
)
# accumulated_residual has shape (component, batch, d_model)

logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
    accumulated_residual, clean_cache
)

fig = px.line(
    logit_lens_logit_diffs.detach().cpu().numpy(), 
    title="Logit Difference From Accumulated Residual Stream",
    labels={"x": "Layer", "y": "Logit Diff"},
)
fig.update_xaxes(title_text="Layer")
fig.update_yaxes(title_text="Logit Diff")
fig.update_layout(dict(
    hovermode="x unified",
    xaxis=dict(
        tickmode="array",
        tickvals=np.arange(len(residual_labels)),
        ticktext=residual_labels,
    )
))
fig.show()
#%%
per_layer_residual, per_layer_labels = clean_cache.decompose_resid(
    layer=-1, pos_slice=-1, return_labels=True
)
per_layer_logit_diffs = residual_stack_to_logit_diff(
    per_layer_residual, clean_cache
)

fig = px.line(
    per_layer_logit_diffs.detach().cpu().numpy(), 
    title="Logit Difference From Each Layer",
    labels={"x": "Layer", "y": "Logit Diff"},
)
fig.update_xaxes(title_text="Layer")
fig.update_yaxes(title_text="Logit Diff")
fig.update_layout(dict(
    hovermode="x unified",
    xaxis=dict(
        tickmode="array",
        tickvals=np.arange(len(per_layer_labels)),
        ticktext=per_layer_labels,
    )
))
fig.show()
#%%
per_head_residual, labels = clean_cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)
per_head_logit_diffs = residual_stack_to_logit_diff(
    per_head_residual, clean_cache
)

fig = px.imshow(
    per_head_logit_diffs.detach().cpu().numpy(), 
    labels={"x":"Head", "y":"Layer"}, 
    title="Logit Difference From Each Head",
    color_continuous_scale="RdBu",
)
fig.show()
#%% [markdown]
## Activation patching
#%% [markdown]
#### Positional patching
#%%
def resid_pre_pos_patching_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    """
    Each HookPoint has a name attribute giving the name of the hook.
    """
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre
#%%
def run_with_pos_patching() -> Float[torch.Tensor, "layer pos"]:
    num_positions = len(clean_tokens[0])
    resid_pre_pos_patching_result = torch.zeros(
        (model.cfg.n_layers, num_positions), device=model.cfg.device
    )
    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        for position in range(num_positions):
            # Use functools.partial to create a temporary hook function with 
            # the position fixed
            temp_hook_fn = partial(resid_pre_pos_patching_hook, position=position)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
                (utils.get_act_name("resid_pre", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits).detach()
            # Store the result, normalizing by the clean and corrupted logit 
            # difference so it's between 0 and 1 (ish)
            resid_pre_pos_patching_result[layer, position] = (
                patched_logit_diff - corrupted_logit_diff
            )/ (clean_logit_diff - corrupted_logit_diff)
    return resid_pre_pos_patching_result
#%%
resid_pre_pos_patching_result = run_with_pos_patching()
#%%
# Add the index to the end of the label, because plotly doesn't like 
# duplicate labels.
token_labels = [
    f"{token}_{index}" 
    for index, token in enumerate(model.to_str_tokens(clean_prompts[0]))
]
fig = px.imshow(
    resid_pre_pos_patching_result.detach().cpu().numpy(), 
    x=token_labels, 
    title="Normalized Logit Diff After Patching Residual Stream on the ELS Task",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    labels=dict(x="Position", y="Layer", color="Normalized Logit Diff"),
)
fig.show()
#%% [markdown]
#### Head patching

#%%
def head_patching_hook(
    z: Float[torch.Tensor, "batch pos head d_head"],
    hook: HookPoint,
    head: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    """
    Each HookPoint has a name attribute giving the name of the hook.
    """
    clean_z = clean_cache[hook.name]
    z[:, :, head, :] = clean_z[:, :, head, :]
    return z

def run_with_head_patching() -> Float[torch.Tensor, "layer head"]:

    head_patching_result = torch.zeros(
        (model.cfg.n_layers, model.cfg.n_heads), 
        device=model.cfg.device
    )

    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            temp_hook_fn = partial(head_patching_hook, head=head)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
                (utils.get_act_name("z", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits).detach()
            # Store the result, normalizing by the clean and corrupted logit 
            # difference so it's between 0 and 1 (ish)
            head_patching_result[layer, head] = (
                patched_logit_diff - corrupted_logit_diff
            )/ (clean_logit_diff - corrupted_logit_diff)
    return head_patching_result
#%%
head_patching_result = run_with_head_patching()
#%%
# Add the index to the end of the label, because plotly doesn't like 
# duplicate labels.
fig = px.imshow(
    head_patching_result.detach().cpu().numpy(), 
    title="Normalized Logit Diff After Patching Residual Stream on the ELS Task",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    labels=dict(x="Head", y="Layer", color="Normalized Logit Diff"),
)
fig.show()

#%% [markdown]
#### Path patching