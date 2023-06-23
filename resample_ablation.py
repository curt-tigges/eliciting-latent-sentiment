#%%
import einops
import numpy as np
from jaxtyping import Float, Int
import plotly.express as px
from prompt_utils import get_dataset, get_logit_diff
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from typing import Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from path_patching import act_patch, Node, IterNode
#%% # Model loading
device = torch.device('cpu')
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)

#%%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(
    model, device
)
#%%
example_prompt = model.to_str_tokens(clean_tokens[0])
adj_token = example_prompt.index(' perfect')
verb_token = example_prompt.index(' loved')
s2_token = example_prompt.index(' movie', example_prompt.index(' movie') + 1)
end_token = len(example_prompt) - 1
# %%
name_filter = lambda name: name.endswith('z')
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter=name_filter
)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
print('clean logit diff', clean_logit_diff)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens, names_filter=name_filter
)
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
print('corrupted logit diff', corrupted_logit_diff)
#%%
clean_cache = clean_cache.to(device)
corrupted_cache = corrupted_cache.to(device)
#%%
assert clean_cache['blocks.0.attn.hook_z'].device == device
#%%
def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    flipped_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 
    0 when performance issame as on flipped input, and 
    1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff)).item()
#%%
# ============================================================================ #
# Node resample ablation

tokenizer_heads = [
    (0, 4),
]
dae_heads = [
    (7, 1),
    (9, 2),
    (10, 1),
    (10, 4),
    (11, 9),
]
iae_heads = [
    (8, 5),
    (9, 2),
    (9, 10),
]
iam_heads = [
    (6, 4),
    (7, 1),
    (7, 5),
]
circuit_heads = dae_heads + iae_heads + iam_heads + tokenizer_heads
non_circuit_heads = [
    (layer, head)
    for layer in range(model.cfg.n_layers)
    for head in range(model.cfg.n_heads)
    if (layer, head) not in circuit_heads
]
circuit_nodes = [Node("z", *node) for node in circuit_heads]
non_circuit_nodes = [Node("z", *node) for node in non_circuit_heads]
#%%
print(
    "unpatched logit diff",
    logit_diff_denoising(logits=corrupted_logits)
)
# %%
non_circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=non_circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching non-circuit nodes: {non_circuit_results:.0%}'
    )
# %%
circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching circuit nodes: {circuit_results:.0%}')

# %%
print(
    "fully patched logit diff",
    logit_diff_denoising(logits=clean_logits)
)

# %%
# ============================================================================ #
# Positional resample ablation
tokenizer_heads = [
    (0, 4, adj_token),
    (0, 4, verb_token),
]
dae_heads = [
    (7, 1, end_token),
    (9, 2, end_token),
    (10, 1, end_token),
    (10, 4, end_token),
    (11, 9, end_token),
]
iae_heads = [
    (8, 5, end_token),
    (9, 2, end_token),
    (9, 10, end_token),
]
iam_heads = [
    (6, 4, s2_token),
    (7, 1, s2_token),
    (7, 5, s2_token),
]
circuit_heads = dae_heads + iae_heads + iam_heads + tokenizer_heads
non_circuit_heads = [
    (layer, head, pos)
    for layer in range(model.cfg.n_layers)
    for head in range(model.cfg.n_heads)
    for pos in range(len(example_prompt))
    if (layer, head, pos) not in circuit_heads
]
circuit_nodes = [
    Node("z", layer=node[0], head=node[1], seq_pos=node[2]) 
    for node in circuit_heads
]
non_circuit_nodes = [
    Node("z",  layer=node[0], head=node[1], seq_pos=node[2]) 
    for node in non_circuit_heads
]
#%%
print(
    "unpatched logit diff",
    logit_diff_denoising(logits=corrupted_logits)
)
# %%
non_circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=non_circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching non-circuit positions/nodes: {non_circuit_results:.0%}'
    )
# %%
circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching circuit positions/nodes: {circuit_results:.0%}')

# %%
print(
    "fully patched logit diff",
    logit_diff_denoising(logits=clean_logits)
)
#%%
