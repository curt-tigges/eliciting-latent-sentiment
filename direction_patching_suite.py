#%%
import einops
import re
from fancy_einsum import einsum
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.prompts import get_dataset
from utils.circuit_analysis import get_logit_diff
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from typing import Tuple, Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from IPython.display import display, HTML
from tqdm.notebook import tqdm
from path_patching import act_patch, Node, IterNode
from utils.store import save_array, load_array, save_html
from utils.prompts import PromptType
#%%
torch.set_grad_enabled(False)
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
#%%
def get_prob_diff(
    logits: Float[Tensor, "batch pos vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"], 
    per_prompt: bool = False,
):
    """
    Gets the difference between the softmax probabilities of the provided tokens 
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    logits = einops.repeat(
        logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    probs: Float[Tensor, "batch n_pairs vocab"] = logits.softmax(dim=-1)
    left_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    )
    right_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    )
    left_probs: Float[Tensor, "batch"] = left_probs.mean(dim=1)
    right_probs: Float[Tensor, "batch"] = right_probs.mean(dim=1)
    if per_prompt:
        return left_probs - right_probs

    return (left_probs - right_probs).mean()
#%% # Data loading
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
def load_data(prompt_type: str, verbose: bool = False):
    all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device, prompt_type=prompt_type)
    if verbose:
        print(all_prompts[:5])
        print(clean_tokens.shape)
    
    # Run model with cache
    # N.B. corrupt -> clean
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens,
    )
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens, per_prompt=False)
    if verbose:
        print('clean logit diff', clean_logit_diff)
    clean_prob_diff = get_prob_diff(clean_logits, answer_tokens, per_prompt=False)
    if verbose:
        print('clean prob diff', clean_prob_diff)
    corrupted_logits, corrupted_cache = model.run_with_cache(
        corrupted_tokens
    )
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens, per_prompt=False)
    if verbose:
        print('corrupted logit diff', corrupted_logit_diff)
    corrupted_prob_diff = get_prob_diff(corrupted_logits, answer_tokens, per_prompt=False)
    if verbose:
        print('corrupted prob diff', corrupted_prob_diff)
    return {
        "all_prompts": all_prompts,
        "answer_tokens": answer_tokens,
        "clean_tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
        "clean_logits": clean_logits,
        "clean_cache": clean_cache,
        "clean_logit_diff": clean_logit_diff,
        "clean_prob_diff": clean_prob_diff,
        "corrupted_logits": corrupted_logits,
        "corrupted_cache": corrupted_cache,
        "corrupted_logit_diff": corrupted_logit_diff,
        "corrupted_prob_diff": corrupted_prob_diff,
    }
#%%
def logit_diff_denoising_base(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_diff: float,
    clean_diff: float,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return ((patched_logit_diff - corrupted_diff) / (clean_diff  - corrupted_diff)).item()
#%%
def prob_diff_denoising_base(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    corrupted_diff: float,
    clean_diff: float,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    
    patched_logit_diff = get_prob_diff(logits, answer_tokens)
    return ((patched_logit_diff - corrupted_diff) / (clean_diff  - corrupted_diff)).item()
#%%
def logit_flip_metric_base(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch pair 2"] ,
    corrupted_diff: float,
    clean_diff: float,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens, per_prompt=True)
    clean_distances = (clean_diff - patched_logit_diff).abs()
    corrupt_distances = (corrupted_diff - patched_logit_diff).abs()
    return (clean_distances < corrupt_distances).float().mean().item()
#%%
def display_cosine_sims(direction_labels: List[str], directions: List[Float[Tensor, "d_model"]]):
    cosine_similarities = []
    for i, (label_i, direction_i) in enumerate(zip(direction_labels, directions)):
        for j, (label_j, direction_j) in enumerate(zip(direction_labels, directions)):
            similarity = einsum(
                "d_model, d_model -> ", 
                direction_i / direction_i.norm(), 
                direction_j / direction_j.norm()
            )
            cosine_similarities.append([label_i, label_j, similarity.cpu().detach().item()])

    sim_df = pd.DataFrame(cosine_similarities, columns=['direction1', 'direction2', 'cosine_similarity'])
    sim_pt = sim_df.pivot(index='direction1', columns='direction2', values='cosine_similarity')
    styled = sim_pt.style.background_gradient(cmap='Reds', axis=0).format("{:.2f}")
    display(styled)
    save_html(styled, "cosine_similarities", model)

#%%
def format_layer_string(s: str) -> str:
    # Find numbers that directly follow the text "layer"
    match = re.search(r'(?<=layer)\d+', s)
    if match:
        number = match.group()
        # Replace the original number with the zero-padded version
        s = s.replace('layer' + number, 'layer' + number.zfill(2))
    return s

#%% # Direction loading
def get_directions(model: HookedTransformer) -> Tuple[List[np.ndarray], List[str]]:
    direction_labels = (
        [f'kmeans_simple_train_ADJ_layer{l}' for l in range(model.cfg.n_layers)] +
        [f'pca2_simple_train_ADJ_layer{l}' for l in range(model.cfg.n_layers)] +
        [f'logistic_regression_simple_train_ADJ_layer{l}' for l in range(model.cfg.n_layers)]
    )
    directions = [
        load_array(label, model) for label in direction_labels
    ]
    for i, direction in enumerate(directions):
        if direction.ndim == 2:
            direction = direction.squeeze(0)
        directions[i] = torch.tensor(direction).to(device, dtype=torch.float32)
    direction_labels = [format_layer_string(label) for label in direction_labels]
    display_cosine_sims(direction_labels, directions)
    return directions, direction_labels
#%%
# ============================================================================ #
# Directional activation patching
#%% # Create new cache
def create_cache_for_dir_patching(
    clean_cache: ActivationCache, 
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"]
) -> ActivationCache:
    '''
    We patch the sentiment direction from corrupt to clean
    '''
    cache_dict = dict()
    for act_name, clean_value in clean_cache.items():
        is_result = act_name.endswith('result')
        is_resid = (
            act_name.endswith('resid_pre') or
            act_name.endswith('resid_post') or
            act_name.endswith('attn_out') or
            act_name.endswith('mlp_out')
        )
        if is_resid:
            clean_value = clean_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s d, d -> b s'
            )
            clean_proj = einops.einsum(
                clean_value, sentiment_dir, 'b s d, d -> b s'
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
            clean_value = clean_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
            )
            clean_proj = einops.einsum(
                clean_value, sentiment_dir, 'b s h d, d -> b s h'
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
            cache_dict[act_name] = clean_value

    return ActivationCache(cache_dict, model)
#%%
def run_head_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    label: str
) -> Tuple[Float[Tensor, "layer head"], go.Figure]:
    head_results: Float[Tensor, "layer head"] = act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        patching_nodes=IterNode(["result"]),
        patching_metric=patching_metric,
        verbose=False,
    )['result'] * 100
    fig = px.imshow(
        head_results,
        title=f"Patching {label} component of attention heads (corrupted -> clean)",
        labels={"x": "Head", "y": "Layer", "color": patching_metric.__name__},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        width=600,
    )
    fig.update_layout(dict(
        coloraxis=dict(colorbar_ticksuffix = "%"),
        # border=True,
        margin={"r": 100, "l": 100}
    ))
    return head_results, fig
#%%
def run_layer_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    label: str
) -> Tuple[Float[Tensor, "layer"], go.Figure]:
    layer_results: Float[Tensor, "layer"] = act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        patching_nodes=IterNode(["resid_post"]),
        patching_metric=patching_metric,
        verbose=False,
        disable=True,
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
    return layer_results, fig
#%%
def run_attn_mlp_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    label: str
) -> Tuple[Float[Tensor, "attn_mlp layer pos"], go.Figure]:
    attn_mlp_results = act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        patching_nodes=IterNode(["resid_pre", "attn_out", "mlp_out"], seq_pos="each"),
        patching_metric=patching_metric,
        verbose=False,
    )
    result_data = torch.stack([r.T for r in attn_mlp_results.values()]) * 100
    assert attn_mlp_results.keys() == {"resid_pre", "attn_out", "mlp_out"}
    labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(orig_input[0]))]
    fig = imshow_p(
        result_data,
        facet_col=0,
        facet_labels=["resid_pre", "attn_out", "mlp_out"],
        title=f"Patching {label} at resid stream & layer outputs (corrupted -> clean)",
        labels={"x": "Sequence position", "y": "Layer", "color": patching_metric.__name__},
        x=labels,
        xaxis_tickangle=45,
        coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1300,
        margin={"r": 100, "l": 100}
    )
    return result_data, fig
#%%
def get_results_for_metric(
    prompt_type: PromptType, patching_metric_base: Callable,
    direction_labels: List[str], directions: List[Float[Tensor, "d_model"]]
):
    model.reset_hooks()
    prompt_metric_label = (
        "prompt_" + 
        prompt_type.value + 
        "_" + 
        "metric_" + 
        patching_metric_base.__name__.replace("_base", "").replace("_denoising", "")
    )
    data_dict = load_data(prompt_type)
    if "logit" in patching_metric_base.__name__:
        clean_diff = data_dict["clean_logit_diff"]
        corrupt_diff = data_dict["corrupted_logit_diff"]
    else:
        clean_diff = data_dict["clean_prob_diff"]
        corrupt_diff = data_dict["corrupted_prob_diff"]
    patching_metric = partial(
        patching_metric_base, 
        answer_tokens=data_dict["answer_tokens"],
        corrupted_diff=corrupt_diff,
        clean_diff=clean_diff,
    )

    bar = tqdm(zip(direction_labels, directions), total=len(direction_labels))
    figures = []
    data = []
    for label, direction in bar:
        direction = direction / direction.norm()
        new_cache = create_cache_for_dir_patching(
            data_dict["clean_cache"], data_dict["corrupted_cache"], direction
        )
        results, fig = run_layer_patching(model, data_dict["corrupted_tokens"], new_cache, patching_metric, label)
        data.append(results.numpy())
        figures.append(fig)
    fig = make_subplots(
        rows=len(direction_labels), cols=1,
        subplot_titles=direction_labels,
        shared_yaxes=False,
        shared_xaxes=False,
    )
    for i, (label, direction) in enumerate(zip(direction_labels, directions)):
        fig.add_trace(figures[i].data[0], row=i + 1, col=1)
        if i == len(direction_labels) - 1:
            fig.update_xaxes(title_text="Layer", row=i + 1, col=1)
    fig.update_layout(
        title=f"Patching residual stream by layer ({prompt_metric_label})",
        height=6600,
        width=1000,
        showlegend=False,
        margin={"r": 100, "l": 100}
    )
    fig.update_yaxes(title_text=patching_metric_base.__name__)
    save_html(fig, f"layer_direction_patching_{prompt_metric_label}", model)
    fig.show()
    layers_df = pd.DataFrame(data)
    layers_df.columns.name = "layer"
    layers_df.index = direction_labels
    layers_style = (
        layers_df
        .style
        .background_gradient(cmap="RdBu", axis=None)
        .format("{:.1f}%")
        .set_caption(f"Layer direction patching ({prompt_metric_label})")
    )
    save_html(layers_style, f"layer_direction_patching_{prompt_metric_label}", model)
    display(layers_style)

    max_layer_df = layers_df.max(axis=1).sort_values(ascending=False).reset_index()
    max_layer_df.columns = ["direction", "max_layer"]
    max_layer_style = (
        max_layer_df
        .style
        .background_gradient(cmap="RdBu", axis=None)
        .format({"max_layer": "{:.1f}%"})
        .hide()
        .set_caption(f"Layer direction patching max ({prompt_metric_label})")
    )
    save_html(max_layer_style, f"layer_direction_patching_{prompt_metric_label}_max_layer", model)
    display(max_layer_style)
#%%
DIRECTIONS, DIRECTION_LABELS = get_directions(model)
# %%
PROMPT_TYPES = [
    PromptType.SIMPLE_MOOD,
    PromptType.COMPLETION,
    PromptType.SIMPLE,
]
METRICS = [
    logit_diff_denoising_base,
    logit_flip_metric_base,
    prob_diff_denoising_base,
]
for metric in METRICS:
    for prompt_type in PROMPT_TYPES:
        get_results_for_metric(prompt_type, metric, DIRECTION_LABELS, DIRECTIONS)
        break
    break
#%%
# FIXME: extract the layer from the direction and only patch that layer rather than taking the max
