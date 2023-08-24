#%%
import itertools
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
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils
from typing import Dict, Iterable, Tuple, Union, List, Optional, Callable
from functools import partial
from IPython.display import display, HTML
from tqdm.notebook import tqdm
from path_patching import act_patch, Node, IterNode
from utils.prompts import get_dataset, PromptType
from utils.circuit_analysis import get_logit_diff, get_prob_diff, create_cache_for_dir_patching, logit_diff_denoising, prob_diff_denoising, logit_flip_denoising
from utils.store import save_array, load_array, save_html, to_csv
from utils.residual_stream import get_resid_name
#%%
torch.set_grad_enabled(False)
pio.renderers.default = "notebook"
#%% # Model loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS = [
    # 'gpt2-small',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
]
#%%
def get_model(name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    model.name = name
    model.set_use_attn_result(True)
    return model
#%% # Data loading
def load_data(
    prompt_type: str, model: HookedTransformer, verbose: bool = False,
    names_filter = None,
) -> dict:
    model.reset_hooks()
    all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device, prompt_type=prompt_type)
    if verbose:
        print(all_prompts[:5])
        print(clean_tokens.shape)
    
    # Run model with cache
    # N.B. corrupt -> clean
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens, 
        names_filter=names_filter,
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
def display_cosine_sims(
    direction_labels: List[str], directions: List[Float[Tensor, "d_model"]],
    model: Union[str, HookedTransformer]
):
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
    styled = (
        sim_pt
        .style
        .background_gradient(cmap='Reds', axis=0)
        .format("{:.1%}")
        .set_caption("Cosine similarities")
        .set_properties(**{'text-align': 'center'})
    )
    display(styled)
    save_html(styled, "cosine_similarities", model)

#%%
def extract_layer_from_string(s: str) -> int:
    # Find numbers that directly follow the text "layer"
    match = re.search(r'(?<=layer)\d+', s)
    if match:
        number = match.group()
        return int(number)
    else:
        return None

#%%
def zero_pad_layer_string(s: str) -> str:
    # Find numbers that directly follow the text "layer"
    number = extract_layer_from_string(s)
    if number is not None:
        # Replace the original number with the zero-padded version
        s = s.replace(f'layer{number}', f'layer{number:02d}')
    return s

#%% # Direction loading
def get_directions(model: HookedTransformer, display: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    n_layers = model.cfg.n_layers + 1
    direction_labels = (
        # [f'kmeans_simple_train_ADJ_layer{l}' for l in range(n_layers)] +
        # [f'pca2_simple_train_ADJ_layer{l}' for l in range(n_layers)] +
        # [f'svd_simple_train_ADJ_layer{l}' for l in range(n_layers)] +
        [f'mean_diff_simple_train_ADJ_layer{l}' for l in range(n_layers)] +
        [f'logistic_regression_simple_train_ADJ_layer{l}' for l in range(n_layers)] +
        [f'das_simple_train_ADJ_layer{l}' for l in range(n_layers)]
    )
    directions = [
        load_array(label, model) for label in direction_labels
    ]
    for i, direction in enumerate(directions):
        if direction.ndim == 2:
            direction = direction.squeeze(0)
        directions[i] = torch.tensor(direction).to(device, dtype=torch.float32)
    direction_labels = [zero_pad_layer_string(label) for label in direction_labels]
    if display:
        display_cosine_sims(direction_labels, directions)
    return directions, direction_labels
#%%
# ============================================================================ #
# Directional activation patching
#%%
def run_position_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    seq_pos: Union[None, int],
    direction_label: str
) -> Tuple[Float[Tensor, ""], go.Figure]:
    """
    Runs patching experiment for a given position and layer.
    seq_pos=None means all positions.
    """
    model.reset_hooks()
    layer = extract_layer_from_string(direction_label)
    act_name, hook_layer = get_resid_name(layer, model)
    node_name = act_name.split('hook_')[-1]
    return act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        patching_nodes=Node(node_name, layer=hook_layer, seq_pos=seq_pos),
        patching_metric=patching_metric,
        verbose=False,
        disable=True,
    ) * 100
#%%
def run_head_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    seq_pos: Union[None, int],
    heads: List[Tuple[int]],
) -> Tuple[Float[Tensor, ""], go.Figure]:
    """
    Runs patching experiment for given heads and position.
    seq_pos=None means all positions.
    """
    model.reset_hooks()
    nodes = [
        Node('result', layer, head, seq_pos=seq_pos) for layer, head in heads
    ]
    return act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        patching_nodes=nodes,
        patching_metric=patching_metric,
        verbose=False,
        disable=True,
    ) * 100
#%%
def get_results_for_direction_and_position(
    patching_metric_base: Callable, 
    prompt_type: PromptType,
    position: str,
    direction_label: str, 
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
    heads: List[Tuple[int]] = None,
) -> float:
    if heads is None:
        names_filter = lambda name: 'resid' in name
    else:
        names_filter = lambda name: 'result' in name
    data_dict = load_data(prompt_type, model, names_filter=names_filter)
    example_prompt = model.to_str_tokens(data_dict["all_prompts"][0])
    if position == 'ALL':
        seq_pos = None
    else:
        seq_pos = prompt_type.get_placeholder_positions(example_prompt)[position][-1]
    if "logit" in patching_metric_base.__name__:
        clean_diff = data_dict["clean_logit_diff"]
        corrupt_diff = data_dict["corrupted_logit_diff"]
    else:
        clean_diff = data_dict["clean_prob_diff"]
        corrupt_diff = data_dict["corrupted_prob_diff"]
    patching_metric = partial(
        patching_metric_base, 
        answer_tokens=data_dict["answer_tokens"],
        corrupted_value=corrupt_diff,
        clean_value=clean_diff,
        return_tensor=True,
    )

    direction = direction / direction.norm()
    new_cache = create_cache_for_dir_patching(
        data_dict["clean_cache"], data_dict["corrupted_cache"], direction, model
    )
    if heads is None:
        return run_position_patching(
            model, data_dict["corrupted_tokens"], new_cache, patching_metric, seq_pos, direction_label
        )
    return run_head_patching(
        model, data_dict["corrupted_tokens"], new_cache, patching_metric, seq_pos, heads
    )

#%%
def get_results_for_metric(
    patching_metric_base: Callable, prompt_types: Iterable[PromptType], 
    direction_labels: List[str], directions: List[Float[Tensor, "d_model"]],
    model: HookedTransformer,
    heads: List[Tuple[int]] = None,
    disable_tqdm: bool = False,
) -> Float[pd.DataFrame, "direction prompt"]:
    use_heads_label = "resid" if heads is None else "attn_result"
    metric_label = patching_metric_base.__name__.replace('_base', '').replace('_denoising', '')
    bar = tqdm(
        itertools.product(prompt_types, zip(direction_labels, directions)), 
        total=len(prompt_types) * len(direction_labels),
        disable=disable_tqdm,
    )
    results = pd.DataFrame(index=direction_labels, dtype=float)
    for prompt_type, (direction_label, direction) in bar:
        bar.set_description(f"{prompt_type.value} {direction_label}")
        placeholders = prompt_type.get_placeholders() + ['ALL']
        for position in placeholders:
            column = pd.MultiIndex.from_tuples([(prompt_type.value, position)], names=['prompt', 'position'])
            result = get_results_for_direction_and_position(
                patching_metric_base, prompt_type, position, direction_label, direction, model, heads
            )
            # Ensure the column exists
            if (prompt_type.value, position) not in results.columns:
                results[column] = np.nan
            results.loc[direction_label, column] = result
    results.columns = pd.MultiIndex.from_tuples(results.columns, names=['prompt', 'position'])
    to_csv(results, f"direction_patching_{metric_label}", model)
    layers_style = (
        results
        .style
        .background_gradient(cmap="Reds", axis=None, low=0, high=1)
        .format("{:.1f}%")
        .set_caption(f"Direction patching ({metric_label}, {use_heads_label}) in {model.name}")
    )
    save_html(layers_style, f"direction_patching_{metric_label}_{use_heads_label}", model)
    display(layers_style)
    return results
# %%
HEADS = {
    "EleutherAI/pythia-2.8b": [
        (17, 19), (22, 5), (14,4), (20, 10), (12, 2), (10, 26), 
        (12, 4), (12, 17), (14, 2), (13, 20), (9, 29), (11, 16) 
    ]
}
PROMPT_TYPES = [
    PromptType.SIMPLE_TRAIN,
    PromptType.SIMPLE_TEST,
    # PromptType.COMPLETION,
    # PromptType.SIMPLE_ADVERB,
    PromptType.SIMPLE_MOOD,
    # PromptType.SIMPLE_FRENCH,
]
METRICS = [
    logit_diff_denoising,
    # logit_flip_metric,
    # prob_diff_denoising,
]
USE_HEADS = [True, ]
model_metric_bar = tqdm(
    itertools.product(MODELS, METRICS, USE_HEADS), total=len(MODELS) * len(METRICS) * len(USE_HEADS)
)
model = None
for model_name, metric, use_heads in model_metric_bar:
    if use_heads and model_name not in HEADS:
        continue
    elif use_heads:
        heads = HEADS[model_name]
    else:
        heads = None
    patch_label = "attn_result" if use_heads else "resid" 
    model_metric_bar.set_description(f"{model_name} {metric.__name__} {patch_label}")
    if model is None or model.name != model_name:
        model = get_model(model_name)
    DIRECTIONS, DIRECTION_LABELS = get_directions(model, display=False)
    results = get_results_for_metric(
        metric, PROMPT_TYPES, DIRECTION_LABELS, DIRECTIONS, model, heads
    )
# %%
