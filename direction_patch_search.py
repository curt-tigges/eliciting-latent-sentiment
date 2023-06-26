#%%
import einops
import numpy as np
from jaxtyping import Float, Int, Bool
import plotly.express as px
from utils.prompts import get_dataset, get_onesided_datasets
from utils.circuit_analysis import get_log_probs
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from typing import Tuple, Union, List, Optional, Callable
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
model.requires_grad_ = False

#%%
prompt_return_dict, answer_tokens = get_onesided_datasets(
    model, device, answer_sentiment='negative'
)
orig_tokens = prompt_return_dict['positive']
new_tokens = prompt_return_dict['negative']
#%%
example_prompt = model.to_str_tokens(orig_tokens[0])
adj_token = example_prompt.index(' perfect')
verb_token = example_prompt.index(' loved')
s2_token = example_prompt.index(' movie', example_prompt.index(' movie') + 1)
end_token = len(example_prompt) - 1
#%%
def name_filter(name: str) -> bool:
    return name in (ACT_NAME, 'blocks.0.attn.hook_z')
# %%
ACT_NAME = 'blocks.0.hook_resid_post'
orig_logits, orig_cache = model.run_with_cache(
    orig_tokens, names_filter=name_filter
)
print(orig_cache.keys())
orig_cache.to(device)
#%%
new_logits, new_cache = model.run_with_cache(
    new_tokens, names_filter=name_filter
)
new_cache.to(device)
#%%
orig_log_prob = get_log_probs(
    orig_logits, answer_tokens, per_prompt=False
)
new_log_prob = get_log_probs(
    new_logits, answer_tokens, per_prompt=False
)
#%%
def patching_metric(
    logits: Float[Tensor, "batch pos vocab"],
    answer_tokens: Int[Tensor, "batch n_answers"] = answer_tokens,
) -> Float[Tensor, ""]:
    """
    0 is new, 1 is original
    """
    log_prob = get_log_probs(logits, answer_tokens, per_prompt=False)
    return (log_prob - new_log_prob) / (orig_log_prob - new_log_prob)

#%%
class InverseRotateLayer(torch.nn.Module):
    """The inverse of a given `LinearLayer` module."""
    def __init__(self, lin_layer):
        super().__init__()
        self.lin_layer = lin_layer

    def forward(self, x):
        output = torch.matmul(x, self.lin_layer.weight.T)
        return output

class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""
    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n,n, device=device)
        # we don't need init if the saved checkpoint has a nice 
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        
    def forward(self, x):
        return torch.matmul(x, self.weight)
#%%
class RotationModule(torch.nn.Module):
    def __init__(self, d_model: int, n_directions: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_directions = n_directions
        rotate_layer = RotateLayer(d_model)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, use_trivialization=False
        )
        self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)

#%%
def apply_rotation_to_cache(
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    rotation: RotationModule,
) -> ActivationCache:
    orig_act: Float[Tensor, "batch pos d_model"] = orig_cache[ACT_NAME]
    new_act: Float[Tensor, "batch pos d_model"] = new_cache[ACT_NAME]
    rotated_orig_act: Float[
        Tensor, "batch pos d_model"
    ] = rotation.rotate_layer(orig_act)
    rotated_new_act: Float[
        Tensor, "batch pos d_model"
    ] = rotation.rotate_layer(new_act)
    d_model_index = einops.repeat(
        torch.arange(model.cfg.d_model, device=device),
        "d -> batch pos d",
        batch=orig_act.shape[0],
        pos=orig_act.shape[1],
    )
    rotated_patch_act = torch.where(
        d_model_index < rotation.n_directions,
        rotated_new_act,
        rotated_orig_act,
    )
    patch_act = rotation.inverse_rotate_layer(rotated_patch_act)
    cache_dict = {
        name: act for name, act in orig_cache.items() if name != ACT_NAME
    }
    cache_dict[ACT_NAME] = patch_act
    return ActivationCache(cache_dict, orig_cache.model)
#%%
def patching_metric_for_module(
    rotation: RotationModule,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
)-> Float[Tensor, ""]:
    patching_cache: ActivationCache = apply_rotation_to_cache(
        orig_cache=orig_cache,
        new_cache=new_cache,
        rotation=rotation,
    )
    results: Float[Tensor, ""] = act_patch(
        model=model,
        orig_input=orig_tokens,
        new_cache=patching_cache,
        patching_nodes=Node(ACT_NAME),
        patching_metric=patching_metric,
        verbose=True,
    )
    return results
#%%
torch.manual_seed(0)
rotation_module = RotationModule(model.cfg.d_model, n_directions=1)
n_epochs = 10
optimizer = torch.optim.Adam(rotation_module.parameters(), lr=1e-3)
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = patching_metric_for_module(
        rotation=rotation_module,
        orig_cache=orig_cache,
        new_cache=new_cache,
    )
    loss.backward()
    print(f"epoch {epoch}: {loss.item()}")
    optimizer.step()
#%%
# direction found by fitted rotation module
rotation_module.rotate_layer.weight[0, :].shape
#%%
