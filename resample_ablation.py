#%%
import einops
import numpy as np
from jaxtyping import Float, Int
import plotly.express as px
from utils.prompts import get_dataset
from utils.circuit_analysis import get_logit_diff
from utils.store import save_array, load_array
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
MODEL_NAME = 'gpt2-small'
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.cfg.use_attn_result = True
model.name = MODEL_NAME
#%%
# corrupted to clean patching style
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
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter=lambda name: name.endswith('z')
)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
print('clean logit diff', clean_logit_diff)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens, names_filter=lambda name: name.endswith('z')
)
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
print('corrupted logit diff', corrupted_logit_diff)
#%%
clean_cache.to(device)
corrupted_cache.to(device)
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
def logit_flips_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    flipped_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    % of clean logit diffs that are in the same direction as the patched logit diff
    0 when performance issame as on flipped input, and 
    1 when performance is same as on clean input.
    '''
    patched_logit_diffs = get_logit_diff(logits, answer_tokens, per_prompt=True)
    return torch.where(patched_logit_diffs * clean_logit_diff > 0, 1.0, 0.0).mean().item()

#%%
# ============================================================================ #
# Head resample ablation

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
    f'logit diff from patching non-circuit nodes: {non_circuit_results:.1%}'
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
    f'logit diff from patching circuit nodes: {circuit_results:.1%}')

# %%
print(
    "fully patched logit diff",
    logit_diff_denoising(logits=clean_logits)
)

#%% # start of flip metric
print(
    "unpatched logit flips",
    logit_flips_denoising(logits=corrupted_logits)
)
# %%
non_circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=non_circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching non-circuit nodes: {non_circuit_results:.1%}'
    )
# %%
circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching circuit nodes: {circuit_results:.1%}')

# %%
print(
    "fully patched logit flips",
    logit_flips_denoising(logits=clean_logits)
)
# end of flip metric

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
    f'logit diff from patching non-circuit positions/nodes: {non_circuit_results:.1%}'
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
    f'logit diff from patching circuit positions/nodes: {circuit_results:.1%}')

# %%
print(
    "fully patched logit diff",
    logit_diff_denoising(logits=clean_logits)
)
#%% # start of flip metric
print(
    "unpatched logit flips",
    logit_flips_denoising(logits=corrupted_logits)
)
# %%
non_circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=non_circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching non-circuit positions/nodes: {non_circuit_results:.1%}'
    )
# %%
circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=clean_cache,
    patching_nodes=circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching circuit positions/nodes: {circuit_results:.1%}')

# %%
print(
    "fully patched logit flips",
    logit_flips_denoising(logits=clean_logits)
)
# end of flip metric
#%%
# ============================================================================ #
# Directional resample ablation
pos_dir = load_array("km_2c_line_embed_and_mlp0", model)
pos_dir /= np.linalg.norm(pos_dir)
pos_dir = torch.tensor(pos_dir, device=device, dtype=torch.float32)
neg_dir = load_array("rotation_direction0", model)
neg_dir /= np.linalg.norm(neg_dir)
neg_dir = torch.tensor(neg_dir, device=device, dtype=torch.float32)
directions = [pos_dir, neg_dir]
#%%
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter=lambda name: name.endswith('result') or name ==  'blocks.0.attn.hook_z'
)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens, names_filter=lambda name: name.endswith('result') or name ==  'blocks.0.attn.hook_z'
)
clean_cache.to(device)
corrupted_cache.to(device)

#%%
def create_directional_cache(directions: List[np.ndarray]) -> ActivationCache:
    cache = {}
    for act_name, orig_value in corrupted_cache.items():
        new_value = clean_cache[act_name]
        for direction in directions:
            if not act_name.endswith('result'):
                cache[act_name] = orig_value
                continue
            orig_proj = einops.einsum(
                orig_value, 
                direction, 
                "batch pos head d_model, d_model -> batch pos head"
            )
            new_proj = einops.einsum(
                new_value, 
                direction, 
                "batch pos head d_model, d_model -> batch pos head"
            )
            proj_diff_rep: Float[Tensor, "batch pos head d_model"] = einops.repeat(
                new_proj - orig_proj, 
                "batch pos head -> batch pos head d_model", 
                d_model=model.cfg.d_model
            )
            orig_value += proj_diff_rep * direction
        cache[act_name] = orig_value
    return ActivationCache(cache, model)

direction_cache = create_directional_cache(directions)
#%%
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
circuit_nodes = [Node("result", *node) for node in circuit_heads]
non_circuit_nodes = [Node("result", *node) for node in non_circuit_heads]
#%%
print(
    "unpatched logit diff",
    logit_diff_denoising(logits=corrupted_logits)
)
# %%
non_circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=direction_cache,
    patching_nodes=non_circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching non-circuit directions/nodes: {non_circuit_results:.1%}'
    )
# %%
circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=direction_cache,
    patching_nodes=circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching circuit directions/nodes: {circuit_results:.1%}')
#%%
full_direction_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=direction_cache,
    patching_nodes=circuit_nodes + non_circuit_nodes,
    patching_metric=logit_diff_denoising,
    verbose=True,
)
print(
    f'logit diff from patching direction at all nodes: {full_direction_results:.1%}'
)
# %%
print(
    "fully patched logit diff",
    logit_diff_denoising(logits=clean_logits)
)
#%% # start of flips metric
print(
    "unpatched logit flips",
    logit_flips_denoising(logits=corrupted_logits)
)
# %%
non_circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=direction_cache,
    patching_nodes=non_circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching non-circuit directions/nodes: {non_circuit_results:.1%}'
    )
# %%
circuit_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=direction_cache,
    patching_nodes=circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching circuit directions/nodes: {circuit_results:.1%}')
#%%
full_direction_results = act_patch(
    model=model,
    orig_input=corrupted_tokens,
    new_cache=direction_cache,
    patching_nodes=circuit_nodes + non_circuit_nodes,
    patching_metric=logit_flips_denoising,
    verbose=True,
)
print(
    f'logit flips from patching direction at all nodes: {full_direction_results:.1%}'
)
# %%
print(
    "fully patched logit flips",
    logit_flips_denoising(logits=clean_logits)
)
# end of flips metric

# %%
