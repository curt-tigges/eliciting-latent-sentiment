#%%
import glob
import itertools
import os
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
from utils.prompts import CleanCorruptedCacheResults, get_dataset, PromptType, ReviewScaffold
from utils.circuit_analysis import create_cache_for_dir_patching, logit_diff_denoising, prob_diff_denoising, logit_flip_denoising
from utils.store import save_array, load_array, save_html, save_pdf, to_csv, get_model_name, extract_layer_from_string, zero_pad_layer_string, DIRECTION_PATTERN, is_file, get_csv, get_csv_path, flatten_multiindex
from utils.residual_stream import get_resid_name
#%%
torch.set_grad_enabled(False)
pio.renderers.default = "notebook"
#%% # Model loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS = [
    'gpt2-small',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    # 'EleutherAI/pythia-160m',
    # 'EleutherAI/pythia-410m',
    # 'EleutherAI/pythia-1.4b',
    # 'EleutherAI/pythia-2.8b',
]
DIRECTION_GLOBS = [
    'mean_diff_simple_train_ADJ*.npy',
    'pca_simple_train_ADJ*.npy',
    'kmeans_simple_train_ADJ*.npy',
    'logistic_regression_simple_train_ADJ*.npy',
    'das_simple_train_ADJ*.npy',
    'das2d_simple_train_ADJ*.npy',
    'das3d_simple_train_ADJ*.npy',
    # 'das_treebank*.npy',
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

#%% # Direction loading
def get_directions(model: HookedTransformer, display: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    model_name = get_model_name(model)
    
    direction_paths = [
        path
        for glob_str in DIRECTION_GLOBS
        for path in glob.glob(os.path.join('data', model_name, glob_str))
        if "None" not in path and "_all_" not in path
    ]
    direction_labels = [os.path.split(path)[-1] for path in direction_paths]
    del direction_paths
    if "2.8b" in model.cfg.model_name:
        layers_to_keep = [0, 1, 7, 14, 16, 18, 24, 31, 32]
        direction_labels = [
            label for label in direction_labels 
            if any([f"layer{l}" in label for l in layers_to_keep])
        ]
    directions = [
        load_array(label, model) for label in direction_labels
    ]
    for i, direction in enumerate(directions):
        if direction.ndim == 2 and direction.shape[1] == 1:
            direction = direction.squeeze(1)
        elif direction.ndim == 2 and direction.shape[0] == 1:
            direction = direction.squeeze(0)
        assert direction.ndim <= 3, f"Direction {direction_labels[i]} has shape {direction.shape}"
        directions[i] = torch.tensor(direction).to(device, dtype=torch.float32)
    direction_labels = [zero_pad_layer_string(label) for label in direction_labels]
    sorted_indices = sorted(
        range(len(direction_labels)), key=lambda i: direction_labels[i]
    )
    direction_labels = [direction_labels[i] for i in sorted_indices]
    directions = [directions[i] for i in sorted_indices]

    if display:
        display_cosine_sims(direction_labels, directions)
    return directions, direction_labels
#%%
# ============================================================================ #
# Directional activation patching
#%%
def batched_act_patch(
    model: HookedTransformer,
    orig_input: Union[str, List[str], Int[Tensor, "batch seq_len"]],
    patching_nodes: Union[IterNode, Node, List[Node]],
    patching_metric: Callable,
    answer_tokens: Int[Tensor, "batch pair correct"],
    new_cache: ActivationCache,
    batch_size: int,
    apply_metric_to_cache: bool = False,
    verbose: bool = False,
    leave: bool = True,
    disable: bool = False,
) -> Float[Tensor, ""]:
    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    device = model.cfg.device
    result = 0
    for batch_idx, start_idx in enumerate(range(0, len(orig_input), batch_size)):
        end_idx = min(start_idx + batch_size, len(orig_input))
        batch_orig_input = orig_input[start_idx:end_idx].to(device=device)
        batch_new_cache = ActivationCache({
            k: v[start_idx:end_idx].to(device) for k, v in new_cache.items()
        }, model=model)
        batch_answer_tokens = answer_tokens[start_idx:end_idx].to(device)
        batch_metric = partial(
            patching_metric,
            answer_tokens=batch_answer_tokens,
        )
        batch_result = act_patch(
            model=model,
            orig_input=batch_orig_input,
            new_cache=batch_new_cache,
            patching_nodes=patching_nodes,
            patching_metric=batch_metric,
            apply_metric_to_cache=apply_metric_to_cache,
            verbose=verbose,
            leave=leave,
            disable=disable,
        )
        result += batch_result
    torch.set_grad_enabled(was_grad_enabled)
    return result / (batch_idx + 1)


def run_position_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    answer_tokens: Int[Tensor, "batch pair correct"],
    seq_pos: Union[None, int],
    direction_label: str,
    batch_size: int,
) -> float:
    """
    Runs patching experiment for a given position and layer.
    seq_pos=None means all positions.
    """
    model.reset_hooks()
    layer = extract_layer_from_string(direction_label)
    act_name, hook_layer = get_resid_name(layer, model)
    node_name = act_name.split('hook_')[-1]
    return batched_act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        batch_size=batch_size,
        patching_nodes=Node(node_name, layer=hook_layer, seq_pos=seq_pos),
        patching_metric=patching_metric,
        answer_tokens=answer_tokens,
        verbose=True,
        disable=True,
    ).item() * 100
#%%
def run_head_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: Callable,
    answer_tokens: Int[Tensor, "batch pair correct"],
    seq_pos: Union[None, int],
    heads: List[Tuple[int]],
    batch_size: int,
) -> float:
    """
    Runs patching experiment for given heads and position.
    seq_pos=None means all positions.
    """
    model.reset_hooks()
    nodes = [
        Node('result', layer, head, seq_pos=seq_pos) for layer, head in heads
    ]
    return batched_act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        batch_size=batch_size,
        patching_nodes=nodes,
        patching_metric=patching_metric,
        answer_tokens=answer_tokens,
        verbose=True,
        disable=True,
    ).item() * 100
#%%
def get_results_for_direction_and_position(
    patching_metric_base: Callable, 
    prompt_type: PromptType,
    position: str,
    direction_label: str, 
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
    device: torch.device = None,
    batch_size: int = 16,
    heads: List[Tuple[int]] = None,
    scaffold: ReviewScaffold = ReviewScaffold.PLAIN,
) -> float:
    if heads is None:
        layer = extract_layer_from_string(direction_label)
        resid_name = get_resid_name(layer, model)[0]
        names_filter = lambda name: name == resid_name
    else:
        names_filter = lambda name: 'result' in name
    model.reset_hooks()
    clean_corrupt_data = get_dataset(model, device, prompt_type=prompt_type, scaffold=scaffold)
    patching_dataset: CleanCorruptedCacheResults = clean_corrupt_data.run_with_cache(
        model, 
        names_filter=names_filter,
        batch_size=batch_size,
        device=device,
        disable_tqdm=True,
    )
    example_prompt = model.to_str_tokens(clean_corrupt_data.all_prompts[0])
    if position == 'ALL':
        seq_pos = None
    else:
        seq_pos = prompt_type.get_placeholder_positions(example_prompt)[position][-1]
    if "logit" in patching_metric_base.__name__:
        clean_diff = patching_dataset.clean_logit_diff
        corrupt_diff = patching_dataset.corrupted_logit_diff
    else:
        clean_diff = patching_dataset.clean_prob_diff
        corrupt_diff = patching_dataset.corrupted_prob_diff
    patching_metric = partial(
        patching_metric_base, 
        flipped_value=corrupt_diff,
        clean_value=clean_diff,
        return_tensor=True,
    )

    new_cache = create_cache_for_dir_patching(
        patching_dataset.clean_cache, patching_dataset.corrupted_cache, direction, model
    )
    if heads is None:
        return run_position_patching(
            model=model, 
            orig_input=clean_corrupt_data.corrupted_tokens, 
            new_cache=new_cache, 
            batch_size=batch_size,
            patching_metric=patching_metric, 
            answer_tokens=clean_corrupt_data.answer_tokens,
            seq_pos=seq_pos, 
            direction_label=direction_label,
        )
    return run_head_patching(
        model=model, 
        orig_input=clean_corrupt_data.corrupted_tokens, 
        new_cache=new_cache, 
        batch_size=batch_size,
        patching_metric=patching_metric, 
        answer_tokens=clean_corrupt_data.answer_tokens,
        seq_pos=seq_pos, 
        heads=heads,
    )

#%%
def get_results_for_metric(
    patching_metric_base: Callable, prompt_types: Iterable[PromptType], 
    direction_labels: List[str], directions: List[Float[Tensor, "d_model"]],
    model: HookedTransformer,
    device: torch.device = None,
    heads: List[Tuple[int]] = None,
    disable_tqdm: bool = False,
    scaffold: ReviewScaffold = ReviewScaffold.PLAIN,
    batch_size: int = 16,
    use_cache: bool = True,
) -> Float[pd.DataFrame, "direction prompt"]:
    use_heads_label = "resid" if heads is None else "attn_result"
    metric_label = patching_metric_base.__name__.replace('_base', '').replace('_denoising', '')
    csv_path = f"direction_patching_{metric_label}_{use_heads_label}.csv"
    if use_cache and is_file(csv_path, model):
        return get_csv(csv_path, model, index_col=0, header=[0, 1])
    bar = tqdm(
        itertools.product(prompt_types, zip(direction_labels, directions)), 
        total=len(prompt_types) * len(direction_labels),
        disable=disable_tqdm,
    )
    results = pd.DataFrame(index=direction_labels, dtype=float)
    for prompt_type, (direction_label, direction) in bar:
        bar.set_description(f"{prompt_type.value} {direction_label} batch_size={batch_size}")
        placeholders = prompt_type.get_placeholders() + ['ALL']
        for position in placeholders:
            column = pd.MultiIndex.from_tuples([(prompt_type.value, position)], names=['prompt', 'position'])
            result = get_results_for_direction_and_position(
                patching_metric_base=patching_metric_base, 
                prompt_type=prompt_type,
                position=position,
                direction_label=direction_label,
                direction=direction,
                model=model,
                device=device,
                heads=heads,
                scaffold=scaffold,
                batch_size=batch_size,
            )
            # Ensure the column exists
            if (prompt_type.value, position) not in results.columns:
                results[column] = np.nan
            results.loc[direction_label, column] = result
    results.columns = pd.MultiIndex.from_tuples(
        results.columns,
        names=['prompt', 'position']
    )
    to_csv(results, csv_path.replace(".csv", ""), model, index=True)
    return results
#%%
def export_results(
    results: pd.DataFrame, metric_label: str, use_heads_label: str
) -> None:
    all_layers = pd.Series([extract_layer_from_string(label) for label in results.index])
    das_treebank_layers = all_layers[results.index.str.contains("das_treebank")]
    if len(das_treebank_layers) > 0:
        mask = ~results.index.str.contains("das") | all_layers.isin(das_treebank_layers)
        mask.index = results.index
        results = results.loc[mask]

    layers_style = (
        flatten_multiindex(results)
        .style
        .background_gradient(cmap="Reds", axis=None, low=0, high=1)
        .format("{:.1f}%")
        .set_caption(f"Direction patching ({metric_label}, {use_heads_label}) in {model.name}")
    )
    save_html(layers_style, f"direction_patching_{metric_label}_{use_heads_label}", model)
    display(layers_style)

    if not results.columns.get_level_values(0).str.contains("treebank").any():
        return

    s_df = results[~results.index.str.contains("treebank")].copy()
    matches = s_df.index.str.extract(DIRECTION_PATTERN)
    multiindex = pd.MultiIndex.from_arrays(matches.values.T, names=['method', 'dataset', 'position', 'layer'])
    s_df.index = multiindex
    s_df = s_df.reset_index().groupby(['method', 'dataset', 'position']).max().drop('layer', axis=1, level=0)
    s_df = flatten_multiindex(s_df)
    s_df = s_df[["simple_test_ADJ", "simple_test_VRB", "simple_test_ALL", "treebank_test_ALL"]]
    s_df.columns = s_df.columns.str.replace("test_", "").str.replace("treebank_ALL", "treebank")
    s_df.index = s_df.index.str.replace("_simple_train_ADJ", "")
    s_style = (
        s_df
        .style
        .background_gradient(cmap="Reds")
        .format("{:.1f}%")
        .set_caption(f"Direction patching ({metric_label}, {use_heads_label}) in {model.name}")
    )
    to_csv(s_df, f"direction_patching_{metric_label}_simple", model, index=True)
    save_html(
        s_style, f"direction_patching_{metric_label}_{use_heads_label}_simple", model,
        font_size=40,
        )
    display(s_style)
    
    t_df = results[results.index.str.contains("das_treebank") & ~results.index.str.contains("None")].copy()
    t_df = t_df.loc[:, t_df.columns.get_level_values(0).str.contains("treebank")]
    matches = t_df.index.str.extract(DIRECTION_PATTERN)
    multiindex = pd.MultiIndex.from_arrays(matches.values.T, names=['method', 'dataset', 'position', 'layer'])
    t_df.index = multiindex
    t_df = t_df.loc[t_df.index.get_level_values(-1).astype(int) < t_df.index.get_level_values(-1).astype(int).max() - 1]
    t_df.sort_index(level=3)
    t_df = flatten_multiindex(t_df)
    t_df.index = t_df.index.str.replace("das_treebank_train_ALL_0", "")
    t_df.columns = ["logit_diff"]
    t_df = t_df.T
    t_style = t_df.style.background_gradient(cmap="Reds").format("{:.1f}%")
    to_csv(t_df, f"direction_patching_{metric_label}_treebank", model, index=True)
    save_html(t_style, f"direction_patching_{metric_label}_{use_heads_label}_treebank", model)
    display(t_style)

    p_df = results[~results.index.str.contains("treebank")].copy()
    matches = p_df.index.str.extract(DIRECTION_PATTERN)
    multiindex = pd.MultiIndex.from_arrays(
        matches.values.T, names=['method', 'dataset', 'position', 'layer']
    )
    p_df.index = multiindex
    p_df = p_df[("treebank_test", "ALL")]
    p_df = p_df.reset_index()
    p_df.columns = p_df.columns.get_level_values(0)
    p_df.layer = p_df.layer.astype(int)
    fig = px.line(x="layer", y="treebank_test", color="method", data_frame=p_df)
    fig.update_layout(
        title="Out-of-distribution directional patching performance by method and layer"
    )
    fig.show()
    p_df = flatten_multiindex(p_df)
    if use_heads_label == "resid":
        to_csv(p_df, f"direction_patching_{metric_label}_layers", model, index=True) # FIXME: add {heads_label}
    save_html(fig, f"direction_patching_{metric_label}_{use_heads_label}_plot", model)
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_plot", model)
# %%
USE_CACHE = False
HEADS = {
    "gpt2-small": [
        (0, 4),
        (7, 1),
        (9, 2),
        (10, 1),
        (10, 4),
        (11, 9),
        (8, 5),
        (9, 2),
        (9, 10),
        (6, 4),
        (7, 1),
        (7, 5),
    ],
    "EleutherAI/pythia-2.8b": [
        (17, 19), (22, 5), (14,4), (20, 10), (12, 2), (10, 26), 
        (12, 4), (12, 17), (14, 2), (13, 20), (9, 29), (11, 16) 
    ]
}
PROMPT_TYPES = [
    PromptType.TREEBANK_TEST,
    # PromptType.SIMPLE_TRAIN,
    PromptType.SIMPLE_TEST,
    # PromptType.COMPLETION,
    # PromptType.SIMPLE_ADVERB,
    # PromptType.SIMPLE_MOOD,
    # PromptType.SIMPLE_FRENCH,
]
METRICS = [
    # logit_diff_denoising,
    logit_flip_denoising,
    # prob_diff_denoising,
]
USE_HEADS = [False, ]
SCAFFOLD = ReviewScaffold.CONTINUATION
model_metric_bar = tqdm(
    itertools.product(MODELS, METRICS, USE_HEADS), total=len(MODELS) * len(METRICS) * len(USE_HEADS)
)
BATCH_SIZES = {
    "gpt2-small": 512,
    "gpt2-medium": 512,
    "gpt2-large": 256,
    "gpt2-xl": 256,
    "EleutherAI/pythia-160m": 512,
    "EleutherAI/pythia-410m": 512,
    "EleutherAI/pythia-1.4b": 256,
    "EleutherAI/pythia-2.8b": 256,
}
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for model_name, metric, use_heads in model_metric_bar:
    # if "flip" in metric.__name__:
    #     batch_size = 32
    # else:
    batch_size = BATCH_SIZES[model_name]
    if use_heads and model_name not in HEADS:
        continue
    elif use_heads:
        heads = HEADS[model_name]
    else:
        heads = None
    patch_label = "attn_result" if use_heads else "resid" 
    model_metric_bar.set_description(f"{model_name} {metric.__name__} {patch_label} batch_size={batch_size}")
    if model is None or model_name not in model.name:
        model = get_model(model_name)
    DIRECTIONS, DIRECTION_LABELS = get_directions(model, display=False)
    results = get_results_for_metric(
        metric, PROMPT_TYPES, DIRECTION_LABELS, DIRECTIONS, model, device, heads, 
        scaffold=SCAFFOLD, batch_size=batch_size,
        use_cache=USE_CACHE,
    )
    use_heads_label = "attn_result" if use_heads else "resid"
    metric_label = metric.__name__.replace('_base', '').replace('_denoising', '')
    export_results(results, metric_label, use_heads_label)
#%%
def concat_layer_data(models: Iterable[str], metric_label: str, use_heads_label: str):
    layer_data = []
    for model in models:
        model_df = get_csv(
            f"direction_patching_{metric_label}_layers", model, index_col=0
        ) # FIXME: add {heads_label}
        if 'layer' not in model_df.columns:
            print(model, metric_label, use_heads_label, model_df, f"direction_patching_{metric_label}_layers")
        model_df['model'] = model
        model_df['max_layer'] = model_df.layer.max()
        layer_data.append(model_df)
    layer_df = pd.concat(layer_data)
    layer_df['model_family'] = np.where(
        layer_df.model.str.contains("pythia"),
        "pythia",
        "gpt2",
    )
    layer_df['model_size'] = layer_df.model.replace({
        "gpt2-small": "small/140m",
        "gpt2-medium": "medium/410m",
        "gpt2-large": "large/1.4b",
        "gpt2-xl": "xl/2.8b",
        "EleutherAI/pythia-160m": "small/140m",
        "EleutherAI/pythia-410m": "medium/410m",
        "EleutherAI/pythia-1.4b": "large/1.4b",
        "EleutherAI/pythia-2.8b": "xl/2.8b",
    })
    fig = px.line(
        x="layer", 
        y="treebank_test", 
        color="method", 
        facet_col="model_size", 
        facet_row="model_family", 
        facet_col_wrap=4, 
        data_frame=layer_df,
        labels={
            "treebank_test": "Directional patching performance (%)",
        }
    )
    fig.update_layout(
        title="Out-of-distribution directional patching performance by method and layer",
        width=1600,
        height=800,
        title_x=0.5,
    )
    for axis in fig.layout:
        if "xaxis" in axis:
            fig.layout[axis].matches = None
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_facet_plot", model)
    save_html(fig, f"direction_patching_{metric_label}_{use_heads_label}_facet_plot", model)
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_facet_plot", model)
    fig.show()
# %%
concat_layer_data(
    MODELS, "logit_diff", "resid"
)
#%%
concat_layer_data(
    MODELS, "logit_flip", "resid"
)