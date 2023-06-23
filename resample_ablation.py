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
device ='cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
circuit_heads = dae_heads + iae_heads + iam_heads
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
