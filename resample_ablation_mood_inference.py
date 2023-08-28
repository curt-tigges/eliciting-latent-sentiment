# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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

# %%
# !ls

# %%
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
# %% # Model loading
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_NAME = 'EleutherAI/pythia-1.4b'
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.cfg.use_attn_result = True
model.name = MODEL_NAME
# %%
# corrupted to clean patching style
clean_corrupt_data = get_dataset(
    model, device, prompt_type='classification_4'
)
all_prompts = clean_corrupt_data.all_prompts
clean_tokens = clean_corrupt_data.clean_tokens
corrupted_tokens = clean_corrupt_data.corrupted_tokens
answer_tokens = clean_corrupt_data.answer_tokens
# %%
example_prompt = model.to_str_tokens(clean_tokens[0])
adj_token = example_prompt.index(' perfect')
verb_token = example_prompt.index(' loved')
s2_token = example_prompt.index(' movie', example_prompt.index(' movie') + 1)
end_token = len(example_prompt) - 1
# %%
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter=lambda name: name.endswith('result') or name == 'blocks.0.attn.hook_z'
)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
print('clean logit diff', clean_logit_diff)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens, names_filter=lambda name: name.endswith('result') or name == 'blocks.0.attn.hook_z'
)
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
print('corrupted logit diff', corrupted_logit_diff)
# %%
clean_cache.to(device)
corrupted_cache.to(device)
# %%
clean_cache['blocks.0.attn.hook_z'].device

# %%
assert clean_cache['blocks.0.attn.hook_z'].device == device
# %%
#  tokenizer_heads = [
#     (0, 4),
# ]
# dae_heads = [
#     (7, 1),
#     (9, 2),
#     (10, 1),
#     (10, 4),
#     (11, 9),
# ]
# iae_heads = [
#     (8, 5),
#     (9, 2),
#     (9, 10),
# ]
# iam_heads = [
#     (6, 4),
#     (7, 1),
#     (7, 5),
# ]
circuit_heads = [
    (6, 11),
    (11, 1),
    (13, 13),
    (18, 2),
    (21, 0),
]
non_circuit_heads = [
    (layer, head)
    for layer in range(model.cfg.n_layers)
    for head in range(model.cfg.n_heads)
    if (layer, head) not in circuit_heads
]
circuit_nodes = [
    Node("result", layer, head) for layer, head in circuit_heads
]
non_circuit_nodes = [
    Node("result", layer, head) for layer, head in non_circuit_heads
]
# %%
# circuit_heads_positions = [
#     (0, 4, adj_token),
#     (0, 4, verb_token),
#     (7, 1, end_token),
#     (9, 2, end_token),
#     (10, 1, end_token),
#     (10, 4, end_token),
#     (11, 9, end_token),
#     (8, 5, end_token),
#     (9, 2, end_token),
#     (9, 10, end_token),
#     (6, 4, s2_token),
#     (7, 1, s2_token),
#     (7, 5, s2_token),
# ]
circuit_heads_positions = [
    (6, 11, 11),
    (6, 11, 28),
    (6, 11, 29),
    (11, 1, end_token),
    (13, 13, end_token),
    (18, 2, end_token),
    (21, 0, end_token),
]
non_circuit_heads_positions = [
    (layer, head, pos)
    for layer in range(model.cfg.n_layers)
    for head in range(model.cfg.n_heads)
    for pos in range(len(example_prompt))
    if (layer, head, pos) not in circuit_heads_positions
]
circuit_position_nodes = [
    Node("result", layer=node[0], head=node[1], seq_pos=node[2]) 
    for node in circuit_heads_positions
]
non_circuit_position_nodes = [
    Node("result",  layer=node[0], head=node[1], seq_pos=node[2]) 
    for node in non_circuit_heads_positions
]
# %%
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
# %%
def logit_flips_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    % of clean logit diffs that are in the same direction as the patched logit diff
    0 when performance issame as on flipped input, and 
    1 when performance is same as on clean input.
    '''
    patched_logit_diffs = get_logit_diff(logits, answer_tokens, per_prompt=True)
    return torch.where(patched_logit_diffs * clean_logit_diff > 0, 1.0, 0.0).mean().item()

# %%
def get_patch_results_base(
    circuit_nodes: List[Node],
    non_circuit_nodes: List[Node],
    metric: Callable,
    new_cache: ActivationCache,
):
    print(
        f"unpatched {metric.__name__}: {metric(logits=corrupted_logits):.1%}",
        
    )
    non_circuit_results = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=new_cache,
        patching_nodes=non_circuit_nodes,
        patching_metric=metric,
        verbose=True,
    )
    print(
        f'{metric.__name__} from patching non-circuit nodes: {non_circuit_results:.1%}'
    )
    circuit_results = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=new_cache,
        patching_nodes=circuit_nodes,
        patching_metric=metric,
        verbose=True,
    )
    print(
        f'{metric.__name__} from patching circuit nodes: {circuit_results:.1%}')
    print(
        f"fully patched {metric.__name__}: {metric(logits=clean_logits):.1%}",
        
    )
# %%
def get_patch_results(
    circuit_nodes: List[Node],
    non_circuit_nodes: List[Node],
    new_cache: ActivationCache = clean_cache,
):
    get_patch_results_base(
        circuit_nodes=circuit_nodes,
        non_circuit_nodes=non_circuit_nodes,
        metric=logit_diff_denoising,
        new_cache=new_cache,
   )
    get_patch_results_base(
        circuit_nodes=circuit_nodes,
        non_circuit_nodes=non_circuit_nodes,
        metric=logit_flips_denoising,
        new_cache=new_cache,
    )

# %%
# ============================================================================ #
# Head resample ablation
print('Head resample ablation')

get_patch_results(
    circuit_nodes=circuit_nodes,
    non_circuit_nodes=non_circuit_nodes,
)

# %%
# ============================================================================ #
# Positional resample ablation
print('Positional resample ablation')
get_patch_results(
    circuit_nodes=circuit_position_nodes,
    non_circuit_nodes=non_circuit_position_nodes,
)
# %%

# ============================================================================ #
# Directional resample ablation
pos_dir = load_array("km_2c_line_embed_and_mlp0", model)
pos_dir /= np.linalg.norm(pos_dir)
pos_dir = torch.tensor(pos_dir, device=device, dtype=torch.float32)
neg_dir = load_array("rotation_direction0", model)
neg_dir /= np.linalg.norm(neg_dir)
neg_dir = torch.tensor(neg_dir, device=device, dtype=torch.float32)
directions = [pos_dir, neg_dir]

# %%
def create_directional_cache(directions: List[np.ndarray]) -> ActivationCache:
    cache = {}
    for act_name, orig_value in corrupted_cache.items():
        new_value = clean_cache[act_name]
        cache[act_name] = orig_value
        for direction in directions:
            if not act_name.endswith('result'):
                continue
            direction_broadcast = einops.repeat(
                direction,
                "d_model -> batch pos head d_model",
                batch=orig_value.shape[0],
                pos=orig_value.shape[1],
                head=orig_value.shape[2],
            )
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
            cache[act_name] += proj_diff_rep * direction_broadcast
        
    return ActivationCache(cache, model)

direction_cache = create_directional_cache(directions)
# %%
print('Directional resample ablation')
get_patch_results(
    circuit_nodes=circuit_nodes,
    non_circuit_nodes=non_circuit_nodes,
    new_cache=direction_cache,
)

# %%
# %%
# ============================================================================ #
# Position & direction resample ablation
print('Position & direction resample ablation')
get_patch_results(
    circuit_nodes=circuit_position_nodes,
    non_circuit_nodes=non_circuit_position_nodes,
    new_cache=direction_cache,
)
# %%
