#%%
import einops
from fancy_einsum import einsum
import numpy as np
from jaxtyping import Float, Int
import plotly.express as px
import plotly.io as pio
from utils.prompts import get_dataset
from utils.circuit_analysis import get_logit_diff
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from typing import Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm.notebook import tqdm
from path_patching import act_patch, Node, IterNode
from utils.store import save_array, load_array, save_html
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
model.set_use_attn_result(True)
model.name = MODEL_NAME
#%% # Data loading
all_prompts, all_answer_tokens, all_clean_tokens, all_corrupted_tokens = get_dataset(model, device)
positive_prompts = all_prompts[::2]
positive_answers = all_answer_tokens[::2]
positive_tokens = all_clean_tokens[::2]
negative_tokens = all_clean_tokens[1::2]
negative_prompts = all_prompts[1::2]
negative_answers = all_answer_tokens[1::2]
print(model.to_string(positive_tokens[0]), model.to_string(negative_tokens[0]))
#%%
example_prompt_indexed = [f"{i}: {s}" for i, s in enumerate(model.to_str_tokens(positive_tokens[0]))]
example_prompt_indexed
#%% # Direction loading
sentiment_direction = load_array('km_2c_line_embed_and_mlp0', model)
sentiment_direction = torch.tensor(sentiment_direction).to(device, dtype=torch.float32)
sentiment_direction = sentiment_direction / sentiment_direction.norm()
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
all_clean_logits, all_clean_cache = model.run_with_cache(
    all_clean_tokens,
)
all_clean_logit_diff = get_logit_diff(all_clean_logits, all_answer_tokens, per_prompt=False)
print('clean logit diff', all_clean_logit_diff)
all_corrupted_logits, all_corrupted_cache = model.run_with_cache(
    all_corrupted_tokens
)
all_corrupted_logit_diff = get_logit_diff(all_corrupted_logits, all_answer_tokens, per_prompt=False)
print('corrupted logit diff', all_corrupted_logit_diff)
#%%
positive_layers = []
negative_layers = []
for layer in range(model.cfg.n_layers):
    positive_projections = einsum('batch pos d_model, d_model -> batch pos', all_clean_cache['resid_post', layer][::2], sentiment_direction).mean(dim=0)
    negative_projections = einsum('batch pos d_model, d_model -> batch pos', all_clean_cache['resid_post', layer][1::2], sentiment_direction).mean(dim=0)
    positive_layers.append(positive_projections)
    negative_layers.append(negative_projections)
positive_layers = torch.stack(positive_layers, dim=0)
negative_layers = torch.stack(negative_layers, dim=0)
all_projections = torch.stack([positive_layers, negative_layers], dim=0)
#%%
fig = px.imshow(
    all_projections.detach().cpu().numpy(),
    facet_col=0,
    title="Projection by position, positive vs. negative prompts",
    labels={'facet_col': 'prompt', 'x': 'position', 'y': 'layer'},
    x=example_prompt_indexed,
)
for i, label in enumerate(['positive', 'negative']):
    fig.layout.annotations[i]['text'] = label
fig.show()
#%%
# negative -> positive
answer_tokens = all_answer_tokens[1::2]
clean_tokens = all_clean_tokens[1::2]
corrupted_tokens = all_corrupted_tokens[1::2]
print(model.to_string(clean_tokens[0]), model.to_string(corrupted_tokens[0]))
#%%
# N.B. corrupt -> clean
clean_logits = model(
    clean_tokens,
)
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
print('clean logit diff', clean_logit_diff)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens
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
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff)).item()
#%%
# ============================================================================ #
# Directional activation ablation
#%% # Create new cache
def create_cache_for_dir_ablation(
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"],
    clean_proj: float = 0.0,
) -> ActivationCache:
    '''
    We patch the sentiment direction from corrupt to clean
    '''
    cache_dict = dict()
    for act_name, corrupt_value in corrupted_cache.items():
        is_result = act_name.endswith('result')
        is_resid = (
            act_name.endswith('resid_pre') or
            act_name.endswith('resid_post') or
            act_name.endswith('attn_out') or
            act_name.endswith('mlp_out')
        )
        if is_resid:
            corrupt_value = corrupt_value.to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s d, d -> b s'
            )
            sentiment_dir_broadcast = einops.repeat(
                sentiment_dir, 'd -> b s d', 
                b=corrupt_value.shape[0], 
                s=corrupt_value.shape[1], 
            )
            proj_diff = einops.repeat(
                clean_proj - corrupt_proj, 
                'b s -> b s d', 
                d=corrupt_value.shape[-1]
            )
            sentiment_adjustment = proj_diff * sentiment_dir_broadcast
            cache_dict[act_name] = (
                corrupt_value + sentiment_adjustment
            )
        elif is_result:
            corrupt_value = corrupt_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
            )
            sentiment_dir_broadcast = einops.repeat(
                sentiment_dir, 'd -> b s h d', 
                b=corrupt_value.shape[0], 
                s=corrupt_value.shape[1], 
                h=corrupt_value.shape[2]
            )
            proj_diff = einops.repeat(
                clean_proj - corrupt_proj, 
                'b s h -> b s h d', 
                d=corrupt_value.shape[3]
            )
            sentiment_adjustment = proj_diff * sentiment_dir_broadcast
            cache_dict[act_name] = (
                corrupt_value + sentiment_adjustment
            )
        else:
            cache_dict[act_name] = corrupt_value

    return ActivationCache(cache_dict, model)
#%%
def run_act_patching(
    model: HookedTransformer,
    new_cache: ActivationCache,
    label: str
):
    # head patching
#     head_results: Float[Tensor, "layer head"] = act_patch(
#         model=model,
#         orig_input=corrupted_tokens,
#         new_cache=new_cache,
#         patching_nodes=IterNode(["result"]),
#         patching_metric=logit_diff_denoising,
#         verbose=True,
#     )
#     fig = px.imshow(
#     head_results['result'] * 100,
#     title=f"Patching {label} component of attention heads (corrupted -> clean)",
#     labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
#     color_continuous_scale="RdBu",
#     color_continuous_midpoint=0,
#     width=600,
# )
#     fig.update_layout(dict(
#         coloraxis=dict(colorbar_ticksuffix = "%"),
#         # border=True,
#         margin={"r": 100, "l": 100}
#     ))
#     fig.show()

    # resid_post patching
    layer_results: Float[Tensor, "layer"] = act_patch(
        model=model,
        orig_input=corrupted_tokens,
        new_cache=new_cache,
        patching_nodes=IterNode(["resid_post"]),
        patching_metric=logit_diff_denoising,
        verbose=True,
    )['resid_post'] * 100
    fig = px.line(
        layer_results,
        title=f"Patching {label} component of residual stream (corrupted -> clean)",
        labels={"index": "Layer", "value": "Logit diff (%)"},
        width=600,
    )
    fig.update_layout(dict(
        coloraxis=dict(colorbar_ticksuffix = "%"),
        margin={"r": 100, "l": 100},
        showlegend=False,
    ))
    fig.show()

    # attn-mlp patching
    # attn_mlp_results = act_patch(
    #     model=model,
    #     orig_input=corrupted_tokens,
    #     new_cache=new_cache,
    #     patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
    #     patching_metric=logit_diff_denoising,
    #     verbose=True,
    # )
    # assert attn_mlp_results.keys() == {"resid_pre", "attn_out", "mlp_out"}
    # labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
    # fig = imshow_p(
    #     torch.stack([r.T for r in attn_mlp_results.values()]) * 100, # we transpose so layer is on the y-axis
    #     facet_col=0,
    #     facet_labels=["resid_pre", "attn_out", "mlp_out"],
    #     title=f"Patching {label} at resid stream & layer outputs (corrupted -> clean)",
    #     labels={"x": "Sequence position", "y": "Layer", "color": "Logit diff variation"},
    #     x=labels,
    #     xaxis_tickangle=45,
    #     coloraxis=dict(colorbar_ticksuffix = "%"),
    #     border=True,
    #     width=1300,
    #     # zmin=-50,
    #     # zmax=50,
    #     margin={"r": 100, "l": 100}
    # )
    # fig.show()
#%%
for ablation_value in tqdm(range(-20, 20, 5)):
    new_cache = create_cache_for_dir_ablation(
        corrupted_cache, sentiment_direction, ablation_value
    )
    label = f"ablation_{ablation_value}".replace("-", "neg_")
    run_act_patching(model, new_cache, label)
#%%
