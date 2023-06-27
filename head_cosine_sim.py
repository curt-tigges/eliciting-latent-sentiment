#%%
import itertools
import einops
import numpy as np
from jaxtyping import Float, Int, Bool
import plotly.express as px
from utils.prompts import get_dataset, get_onesided_datasets
from utils.circuit_analysis import get_log_probs
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from typing import Tuple, Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm import tqdm
#%% # Model loading
device = torch.device('cpu')
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.requires_grad_ = False
model.cfg.use_attn_results = True
#%%
#%%
SENTIMENT = 'positive'
prompt_return_dict, answer_tokens = get_onesided_datasets(
    model, 
    device, 
    answer_sentiment=SENTIMENT,
    dataset_sentiments=[SENTIMENT, 'neutral'],
    n_answers=5,
)
sent_tokens = prompt_return_dict[SENTIMENT]
neutral_tokens = prompt_return_dict['neutral']
#%%
example_prompt = model.to_str_tokens(sent_tokens[0])
adj_token = example_prompt.index(' perfect') if ' perfect' in example_prompt else example_prompt.index(' awful')
verb_token = example_prompt.index(' loved') if ' loved' in example_prompt else example_prompt.index(' hated')
s2_token = example_prompt.index(' movie', example_prompt.index(' movie') + 1)
end_token = len(example_prompt) - 1
example_prompt = [f"{i}: {tok}" for i, tok in enumerate(example_prompt)]
#%%
def name_filter(name: str) -> bool:
    return name.endswith('result') or name.endswith('z') or name.endswith('_scale')
clean_logits, clean_cache = model.run_with_cache(
    sent_tokens, 
    names_filter = name_filter,
)
clean_cache.to(device)
#%%
neutral_logits, neutral_cache = model.run_with_cache(
    neutral_tokens, 
    names_filter = name_filter,
)
neutral_cache.to(device)
#%%
cache_dict = dict()
for act_name, act in clean_cache.items():
    cache_dict[act_name] = act - neutral_cache[act_name].mean(dim=0, keepdim=True)
#%%
diff_cache = ActivationCache(cache_dict, model)
#%%
batch_size, seq_len, d_vocab = clean_logits.shape
layers = model.cfg.n_layers
heads = model.cfg.n_heads
#%%
def get_normed_mean_stack(
    cache: ActivationCache,
) -> Float[Tensor, "components pos d_model"]:
    stack: Float[
        Tensor, "components batch pos d_model"
    ] = cache.stack_head_results(layer=-1, return_labels=False)
    layer_index: Int[Tensor, "layers heads"] = einops.repeat(
        torch.arange(layers, device=device), 
        "layers -> (layers heads)", 
        heads=heads
    )
    for layer, pos in itertools.product(range(layers), range(seq_len)):
        stack[layer_index == layer, :, pos, :] = cache.apply_ln_to_stack(
            stack[layer_index == layer, :, pos, :], layer=layer, pos_slice=pos
        )
    component_means: Float[Tensor, "components pos d_model"] = einops.reduce(
        stack,
        "components batch pos d_model-> components pos d_model",
        "mean",
    )
    rearranged_means: Float[Tensor, "layer_head_pos d_model"] = einops.rearrange(
        component_means,
        "(layer head) pos d_model -> (layer head pos) d_model",
        layer=layers,
        head=heads,
    )
    return rearranged_means
#%%
clean_stack: Float[
    Tensor, "layer_head_pos d_model"
] = get_normed_mean_stack(clean_cache)
#%%
neutral_stack: Float[
    Tensor, "layer_head_pos d_model"
] = get_normed_mean_stack(neutral_cache)
#%%
diff_stack = clean_stack - neutral_stack
diff_stack = diff_stack / diff_stack.norm(dim=-1, keepdim=True)
#%%
cosine_sims = einops.einsum(
    diff_stack, 
    diff_stack, 
    "layer_head_pos_x d_model, layer_head_pos_y d_model -> layer_head_pos_x layer_head_pos_y",
).cpu().detach().numpy()
#%%
def layer_head_pos_index(layer: int, head: int, pos: int) -> int:
    return layer * heads * seq_len + head * seq_len + pos
def layer_head_pos_index_einops(layer: int, head: int, pos: int) -> int:
    layer_index = einops.repeat(
        torch.arange(layers),
        "layer -> (layer head pos)",
        head=heads,
        pos=seq_len,
    )
    head_index = einops.repeat(
        torch.arange(heads),
        "head -> (layer head pos)",
        layer=layers,
        pos=seq_len,
    )
    pos_index = einops.repeat(
        torch.arange(seq_len),
        "pos -> (layer head pos)",
        layer=layers,
        head=heads,
    )
    return torch.where(
        (layer_index == layer) & (head_index == head) & (pos_index == pos)
    )[0].item()

#%%
assert layer_head_pos_index(7, 1, s2_token ) == layer_head_pos_index_einops(7, 1, s2_token )
#%%
layer_head_positions = [
    # tokenizer heads
    (0, 4, adj_token),
    (0, 4, verb_token),
    # dae heads
    (7, 1, end_token),
    (9, 2, end_token),
    (10, 1, end_token),
    (10, 4, end_token),
    (11, 9, end_token),
    # iae_heads 
    (8, 5, end_token),
    (9, 10, end_token),
    # iam_heads 
    (6, 4, s2_token),
    (7, 1, s2_token),
    (7, 5, s2_token),
]
indices = [layer_head_pos_index(*lhp) for lhp in layer_head_positions]
labels = [
    f'{i}: L{lhp[0]}H{lhp[1]}P{example_prompt[lhp[2]]}' 
    for i, lhp in enumerate(layer_head_positions)
]
#%%
small_table = cosine_sims[indices, :][:, indices]
# set upper triangle to 0
small_table = np.tril(small_table, k=-1)
# %%
fig = px.imshow(
    small_table,
    labels={'x': 'Layed,Head,Pos', 'y': 'Layer,Head,Pos'},
    title=f'Two-way cosine similarity table: {SENTIMENT} sentiment',
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    x=labels,
    y=labels,
    # height = heads * layers * 20,
)
fig.show()
# %%
