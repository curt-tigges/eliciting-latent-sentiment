#%%
import einops
from fancy_einsum import einsum
import numpy as np
from jaxtyping import Float, Int
import plotly.express as px
import plotly.io as pio
import wandb
from utils.prompts import get_dataset
from utils.circuit_analysis import get_logit_diff
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
from tqdm.notebook import tqdm
from path_patching import act_patch, Node, IterNode
from utils.store import save_array, load_array
from sklearn.linear_model import LogisticRegression
#%%
pio.renderers.default = "notebook"
update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}

def imshow_p(tensor, **kwargs):
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
    return fig
#%% # Model loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.name = MODEL_NAME
model = model.requires_grad_(False)
#%% # Data loading
all_prompts, answer_tokens, new_tokens, orig_tokens = get_dataset(model, device)
adjective_position = 6
verb_position = 9
#%%
example_prompt = [f"{i}:{tok}" for i, tok in enumerate(model.to_str_tokens(all_prompts[0]))]
example_prompt
#%% # Run model with cache
def name_filter(name: str):
    return (
        name.endswith('result') or 
        name.endswith('resid_pre') or
        name.endswith('resid_post') or  
        name.endswith('attn_out') or 
        name.endswith('mlp_out') or 
        (name == 'blocks.0.attn.hook_q') or 
        (name == 'blocks.0.attn.hook_z')
    )
orig_logits, orig_cache = model.run_with_cache(
    orig_tokens
)
orig_cache.to(device)
orig_logit_diff = get_logit_diff(orig_logits, answer_tokens, per_prompt=False)
print('original logit diff', orig_logit_diff)
new_logits, new_cache = model.run_with_cache(
    new_tokens,
)
new_cache.to(device)
new_logit_diff = get_logit_diff(new_logits, answer_tokens, per_prompt=False)
print('new logit diff', new_logit_diff)
#%%
def logit_diff_denoising_loss(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 
    0 on new
    and 1 on old
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - new_logit_diff) / (orig_logit_diff - new_logit_diff)

#%%
# ============================================================================ #
# Directional activation patching
#%%
def train_logistic_regression(
    position: int,
    layer: int,
) -> Float[np.ndarray, "d_model"]:
    X_train: Float[Tensor, "batch d_model"] = new_cache['resid_post', layer][:, position].detach().cpu().numpy()
    layers_test = [l for l in range(model.cfg.n_layers) if l != layer]
    seq_len = new_cache['resid_post', layer].shape[1]
    positions_test = [p for p in range(seq_len) if p != position]
    # define out of sample test set
    resid_test = torch.cat([new_cache['resid_post', l] for l in layers_test], dim=0)
    X_test: Float[Tensor, "layer_batch pos d_model"] = resid_test[:, positions_test].detach().cpu().numpy()
    X_test: Float[Tensor, "layer_batch_pos d_model"] = einops.rearrange(
        X_test, "batch pos d_model -> (batch pos) d_model"
    )
    # set y_train to alternating 0s and 1s
    y_train = np.zeros(X_train.shape[0])
    y_train[::2] = 1
    y_test = einops.repeat(y_train, "batch -> (layer batch pos)", layer=model.cfg.n_layers - 1, pos=seq_len - 1)
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    print(f"LR in-sample accuracy on layer {layer}, position {position}: {logreg_model.score(X_train, y_train):.1%}")
    print(f"LR out-of-sample accuracy on layer {layer}, position {position}: {logreg_model.score(X_test, y_test):.1%}")
    return logreg_model.coef_
#%%
adj_position = 6
adj_token_lr = train_logistic_regression(position=adj_position, layer=0)
end_token_lr = train_logistic_regression(position=new_tokens.shape[1] - 1, layer=model.cfg.n_layers - 1)
save_array(adj_token_lr, 'adj_token_lr', model)
save_array(end_token_lr, 'end_token_lr', model)
#%%
# ============================================================================ #
# Training DAS

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
    input: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    layer: int,
    position: int,
    new_value: Float[Tensor, "d_model"],
):
    assert 'resid_post' in hook.name
    if hook.layer() != layer:
        return input
    input[:, position] = new_value
    return input
    
#%%
def act_patch_simple(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_value: Float[Tensor, "d_model"],
    layer: int,
    position: int,
    patching_metric: Callable,
) -> Float[Tensor, ""]:
    model.reset_hooks()
    hook_fn = partial(hook_fn_base, layer=layer, position=position, new_value=new_value)
    logits = model.run_with_hooks(
        orig_input,
        fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_fn)],
    )
    assert logits.requires_grad or not model.training, (
        "logits should require grad, otherwise we can't backpropagate through them. "
        f"Layer: {layer}, position: {position}, new_value: {new_value.requires_grad}"
    )
    return patching_metric(logits)
#%%
class RotationModule(torch.nn.Module):
    def __init__(
        self, 
        model,
        orig_tokens: Int[Tensor, "batch pos"],
        orig_z: Float[Tensor, "batch pos head d_head"],
        n_directions: int = 1,
    ):
        super().__init__()
        self.model = model
        self.register_buffer('orig_z', orig_z)
        self.register_buffer('orig_tokens', orig_tokens)
        self.d_model = model.cfg.d_model
        self.n_directions = n_directions
        rotate_layer = RotateLayer(model.cfg.d_model)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, use_trivialization=False
        )
        self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)

    def apply_rotation(
        self,
        orig_resid_post: Float[Tensor, "batch d_model"],
        new_resid_post: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch d_model"]:
        rotated_orig_act: Float[
            Tensor, "batch d_model"
        ] = self.rotate_layer(orig_resid_post)
        rotated_new_act: Float[
            Tensor, "batch d_model"
        ] = self.rotate_layer(new_resid_post)
        d_model_index = einops.repeat(
            torch.arange(model.cfg.d_model, device=device),
            "d -> batch d",
            batch=orig_resid_post.shape[0],
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
        layer: int,
        position: int,
    ) -> Float[Tensor, ""]:
        patched_resid_post: Float[Tensor, "batch d_model"] = self.apply_rotation(
            orig_resid_post=orig_resid_post,
            new_resid_post=new_resid_post,
        )
        metric: Float[Tensor, ""] = act_patch_simple(
            model=self.model,
            orig_input=orig_tokens,
            new_value=patched_resid_post,
            patching_metric=logit_diff_denoising_loss,
            layer=layer,
            position=position,
        )
        return metric
#%%
class TrainingConfig:

    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)
#%%
def train_rotation(**given_config_dict) -> Tuple[HookedTransformer, List[Tensor]]:
    # Initialize wandb
    config_dict = dict(
        seeds=5,
        lr=1e-3,
        epochs=50,
        n_directions=1,
        train_layer=0,
        eval_layer=1,
        train_position=adjective_position,
        eval_position=verb_position,
        wandb_enabled=True,
    )
    config_dict.update(given_config_dict)
    if config_dict["wandb_enabled"]:
        wandb.init(project='train_rotation', config=config_dict)
        config = wandb.config
    else:
        config = TrainingConfig(config_dict)

    # Create a list to store the losses and models
    losses = []
    models = []
    directions = []
    if isinstance(config.seeds, int):
        random_seeds = np.arange(config.seeds)
    else:
        random_seeds = config.seeds
    step = 0
    for seed in random_seeds:
        print(f"Training seed {seed}")
        if config.wandb_enabled:
            wandb.log({"Seed": seed})
        torch.manual_seed(seed)
        
        # Create the rotation module
        rotation_module = RotationModule(
            model=model,
            orig_z=orig_cache["blocks.0.attn.hook_z"],
            orig_tokens=orig_tokens,
            n_directions=config.n_directions,
        )

        # Define the optimizer
        optimizer = torch.optim.Adam(rotation_module.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            rotation_module.train()
            optimizer.zero_grad()
            loss = rotation_module(
                orig_cache["resid_post", config.train_layer][:, config.train_position, :],
                new_cache["resid_post", config.train_layer][:, config.train_position, :],
                layer=config.train_layer,
                position=config.train_position,
            )
            loss.backward()
            optimizer.step()
            rotation_module.eval()
            with torch.inference_mode():
                eval_loss = rotation_module(
                    orig_cache["resid_post", config.eval_layer][:, config.eval_position, :],
                    new_cache["resid_post", config.eval_layer][:, config.eval_position, :],
                    layer=config.eval_layer,
                    position=config.eval_position,
                )
            if config.wandb_enabled:
                wandb.log({"training_loss": loss.item(), "validation_loss": eval_loss.item()}, step=step)

            # Store the loss and model for this seed
            losses.append(loss.item())
            models.append(rotation_module.state_dict())
            step += 1
        direction = rotation_module.rotate_layer.weight[0, :]
        directions.append(direction)

    best_model_idx = min(range(len(losses)), key=losses.__getitem__)

    # Log the best model's loss and save the model
    if config.wandb_enabled:
        wandb.log({"Best Loss": losses[best_model_idx]})
        wandb.save("best_rotation.pt")
        wandb.finish()

    # Load the best model
    best_model_state_dict = models[best_model_idx]
    rotation_module.load_state_dict(best_model_state_dict)
    return rotation_module, directions

#%%
# run a wandb sweep over n_directions from 1 to 10
sweep_config = {
    "method": "grid",
    "metric": {"name": "validation_loss", "goal": "minimize"},
    "parameters": {
        "n_directions": {"values": list(range(1, 11))},
        "seeds": {"values": [1]},
        "lr": {"values": [1e-3]},
        "epochs": {"values": [200]},
        "model": {"values": [model.name]},
        "train_position": {"values": [adj_position]},
        "train_layer": {"values": [0]},
        "eval_position": {"values": [verb_position]},
        "eval_layer": {"values": [1]},
    }
}
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train_rotation)

#%%
rotation_module_adj, directions_adj = train_rotation(
    seeds=1, 
    epochs=200, 
    lr=1e-3,
    train_position=adj_position, 
    train_layer=0,
    eval_position=verb_position,
    eval_layer=1,
    wandb_enabled=True,
)
#%%
if rotation_module_adj.n_directions == 1:
    save_array(
        rotation_module_adj.rotate_layer.weight[0, :].cpu().detach().numpy(), 
        f"rotation_direction_adj", model
    )
#%%
final_token = new_tokens.shape[1] - 1
final_layer = model.cfg.n_layers - 1
rotation_module_end, directions_end = train_rotation(
    seeds=1, 
    epochs=60, 
    train_position=final_token, 
    train_layer=final_layer,
    eval_position=final_token,
    eval_layer=final_layer,
)
#%%
if rotation_module_end.n_directions == 1:
    save_array(
        rotation_module_end.rotate_layer.weight[0, :].cpu().detach().numpy(), 
        f"rotation_direction_end", model
    )
    
#%%