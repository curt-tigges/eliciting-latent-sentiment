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
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from typing import Tuple, Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm import tqdm
import wandb
#%% # Model loading
device = torch.device('cpu')
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model = model.requires_grad_(False)

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
        self.weight = torch.nn.Parameter(
            weight, requires_grad=True,
        ).to(device)
        
    def forward(self, x):
        return torch.matmul(x, self.weight)
#%%
def hook_fn_base(
    input: Float[Tensor, "batch pos d_model"], 
    hook: HookPoint, 
    new_value: Float[Tensor, "batch pos d_model"]
):
    if hook.name == ACT_NAME:
        return new_value
    return input
#%%
def act_patch_simple(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    patching_metric: Callable,
    new_value: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, ""]:
    model.reset_hooks()
    hook_fn = partial(hook_fn_base, new_value=new_value)
    logits = model.run_with_hooks(
        orig_input, fwd_hooks=[(ACT_NAME, hook_fn)]
    )
    return patching_metric(logits)
#%%
class RotationModule(torch.nn.Module):
    def __init__(
        self, 
        model: HookedTransformer, 
        orig_hook_z: Float[Tensor, "batch pos head d_head"],
        orig_tokens: Int[Tensor, "batch pos"],
        n_directions: int = 1,
    ):
        super().__init__()
        self.model = model
        self.register_buffer('orig_hook_z', orig_hook_z.to(device))
        self.register_buffer('orig_tokens', orig_tokens.to(device))
        self.d_model = model.cfg.d_model
        self.n_directions = n_directions
        rotate_layer = RotateLayer(model.cfg.d_model)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, use_trivialization=False
        )
        self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)

    def apply_rotation(
        self,
        orig_resid_post: Float[Tensor, "batch pos d_model"],
        new_resid_post: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        rotated_orig_act: Float[
            Tensor, "batch pos d_model"
        ] = self.rotate_layer(orig_resid_post)
        rotated_new_act: Float[
            Tensor, "batch pos d_model"
        ] = self.rotate_layer(new_resid_post)
        d_model_index = einops.repeat(
            torch.arange(model.cfg.d_model, device=device),
            "d -> batch pos d",
            batch=orig_resid_post.shape[0],
            pos=orig_resid_post.shape[1],
        )
        rotated_patch_act = torch.where(
            d_model_index < self.n_directions,
            rotated_new_act,
            rotated_orig_act,
        )
        patch_act = self.inverse_rotate_layer(rotated_patch_act)
        return patch_act

    def forward(
        self, 
        orig_resid_post: Float[Tensor, "batch pos d_model"],
        new_resid_post: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, ""]:
        patching_tensor: ActivationCache = self.apply_rotation(
            orig_resid_post=orig_resid_post,
            new_resid_post=new_resid_post,
        )
        results: Float[Tensor, ""] = act_patch_simple(
            model=self.model,
            orig_input=orig_tokens,
            new_value=patching_tensor,
            patching_metric=patching_metric,
        )
        return results
#%%
def train_rotation(**config_dict):
    # Initialize wandb
    config_dict = {
        "num_seeds": config_dict.get("num_seeds", 5),
        "lr": config_dict.get("lr", 1e-3),
        "n_epochs": config_dict.get("n_epochs", 50),
        "n_directions": config_dict.get("n_directions", 1),
    }
    wandb.init(project='train_rotation', config=config_dict)
    config = wandb.config

    # Create a list to store the losses and models
    losses = []
    models = []
    directions = []
    random_seeds = np.arange(config.num_seeds)
    step = 0
    for seed in random_seeds:
        wandb.log({"Seed": seed})
        torch.manual_seed(seed)
        
        # Create the rotation module
        rotation_module = RotationModule(
            model,
            orig_hook_z=orig_cache['blocks.0.attn.hook_z'],
            orig_tokens=orig_tokens,
            n_directions=config.n_directions,
        )

        # Define the optimizer
        optimizer = torch.optim.Adam(rotation_module.parameters(), lr=config.lr)

        for epoch in range(config.n_epochs):
            optimizer.zero_grad()
            loss = rotation_module(orig_cache[ACT_NAME], new_cache[ACT_NAME])
            loss.backward()
            optimizer.step()
            wandb.log({"Loss": loss.item()}, step=step)

            # Store the loss and model for this seed
            losses.append(loss.item())
            models.append(rotation_module.state_dict())
            step += 1
        direction = rotation_module.rotate_layer.weight[0, :]
        directions.append(direction)

    # log the cosine similarity between the directions
    for i in range(len(directions)):
        for j in range(i+1, len(directions)):
            similarity = torch.cosine_similarity(
                directions[i], directions[j], dim=0
            )
            print({f"Cosine Similarity {i} and {j}": similarity.item()})
    best_model_idx = min(range(len(losses)), key=losses.__getitem__)

    # Log the best model's loss and save the model
    wandb.log({"Best Loss": losses[best_model_idx]})
    wandb.save("best_rotation.pt")
    wandb.finish()

    # Load the best model
    best_model_state_dict = models[best_model_idx]
    rotation_module.load_state_dict(best_model_state_dict)
    return rotation_module

#%%
rotation_module = train_rotation()

#%%
# direction found by fitted rotation module
with open("data/rotation_direction.npy", "wb") as f:
    np.save(f, rotation_module.rotate_layer.weight[0, :].cpu().detach().numpy())
#%%
