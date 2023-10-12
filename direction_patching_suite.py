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
from typing import Dict, Iterable, Literal, Tuple, Union, List, Optional, Callable
from functools import partial
from IPython.display import display, HTML
from tqdm.notebook import tqdm
from path_patching import act_patch, Node, IterNode
from utils.prompts import CleanCorruptedCacheResults, get_dataset, PromptType, ReviewScaffold
from utils.circuit_analysis import create_cache_for_dir_patching, logit_diff_denoising, prob_diff_denoising, logit_flip_denoising, PatchingMetric
from utils.store import save_array, load_array, save_html, save_pdf, to_csv, get_model_name, extract_layer_from_string, zero_pad_layer_string, DIRECTION_PATTERN, is_file, get_csv, get_csv_path, flatten_multiindex, save_text, load_text
from utils.residual_stream import get_resid_name
#%%
torch.set_grad_enabled(False)
pio.renderers.default = "notebook"
#%% # Global Settings
USE_CACHE = False
ALL_LAYERS = False
PROJ = "ortho"
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
    'mean_diff_simple_*all_but_one*.npy',
    'logistic_regression_simple_*all_but_one*.npy',
    'das_simple_*all_but_one*.npy',
    # 'pca_simple_train_ADJ*.npy',
    # 'kmeans_simple_train_ADJ*.npy',
    # 'das2d_simple_train_ADJ*.npy',
    # 'das3d_simple_train_ADJ*.npy',
    # 'random_direction_layer*.npy',
    # 'das_treebank*.npy',
]
PROMPT_TYPES = [
    # PromptType.SIMPLE_TEST,
    # PromptType.TREEBANK_TEST,
    # PromptType.SIMPLE_ADVERB,
    PromptType.SIMPLE_BOOK,
    PromptType.SIMPLE_PRODUCT,
    PromptType.SIMPLE_RES,
    PromptType.SIMPLE_TRAIN,
    # PromptType.COMPLETION,
    # PromptType.SIMPLE_FRENCH,
]
SCAFFOLD = ReviewScaffold.CONTINUATION
METRICS = [
    PatchingMetric.LOGIT_DIFF_DENOISING,
    # PatchingMetric.LOGIT_FLIP_DENOISING,
    # PatchingMetric.PROB_DIFF_DENOISING,
]
USE_HEADS = [False, ]
#%%
def get_model(name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    model.set_use_attn_result(True)
    return model
#%% # Direction loading
def get_direction(label: str, model: HookedTransformer) -> Float[Tensor, "d_model"]:
    if "_layer0.npy" not in label:
        # reverse zero-padding
        label = label.replace("_layer0", "_layer")
    if "*" in label:
        paths = [
            path for path in glob.glob(os.path.join('data', model_name, label))
            if "all_but_one" not in path and "None" not in path and
            "_ALL_" not in path
        ]
        assert len(paths) == 1, f"Found {len(paths)} paths for {label}: {paths}"
        label = os.path.split(paths[0])[-1]
    direction = load_array(label, model)
    if direction.ndim == 2 and direction.shape[1] == 1:
            direction = direction.squeeze(1)
    elif direction.ndim == 2 and direction.shape[0] == 1:
        direction = direction.squeeze(0)
    assert direction.ndim <= 3, f"Direction {label} has shape {direction.shape}"
    direction = torch.tensor(direction).to(device, dtype=torch.float32)
    return direction
#%%
def get_directions(model: HookedTransformer) -> Tuple[List[np.ndarray], List[str]]:
    model_name = get_model_name(model)
    
    direction_paths = [
        path
        for glob_str in DIRECTION_GLOBS
        for path in glob.glob(os.path.join('data', model_name, glob_str))
        if "None" not in path and "_activations" not in path and '_ALL_' not in path
        and 'french' not in path # FIXME: remove this
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
        get_direction(label, model) for label in direction_labels
    ]
    direction_labels = [zero_pad_layer_string(label) for label in direction_labels]
    sorted_indices = sorted(
        range(len(direction_labels)), key=lambda i: direction_labels[i]
    )
    direction_labels = [direction_labels[i] for i in sorted_indices]
    directions = [directions[i] for i in sorted_indices]

    # direction_labels.append('zero')
    # directions.append(torch.zeros_like(directions[0]))

    return directions, direction_labels
#%%
# ============================================================================ #
# Directional activation patching
FN_OF_LOGITS = Callable[
    [Float[Tensor, "batch seq_len d_model"]],
    Float[Tensor, ""]
]
FN_OF_ANSWERS = Callable[
    [Float[Tensor, "batch seq_len d_model"], Int[Tensor, "batch pair correct"]],
    Float[Tensor, ""]
]
#%%
def batched_act_patch(
    model: HookedTransformer,
    orig_input: Int[Tensor, "batch seq_len"],
    patching_nodes: Union[IterNode, Node, List[Node]],
    patching_metric: FN_OF_ANSWERS,
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
    result: Float[Tensor, ""] = torch.tensor([0], device=device, dtype=torch.float32)
    bar = tqdm(
        enumerate(range(0, len(orig_input), batch_size)),
        total=len(orig_input) // batch_size,
        disable=disable,
    )
    batch_idx = 0
    for batch_idx, start_idx in bar:
        end_idx = min(start_idx + batch_size, len(orig_input))
        batch_orig_input = orig_input[start_idx:end_idx].to(device=device)
        batch_new_cache = ActivationCache({
            k: v[start_idx:end_idx].to(device) for k, v in new_cache.items()
        }, model=model)
        batch_answer_tokens = answer_tokens[start_idx:end_idx].to(device)
        batch_metric: FN_OF_LOGITS = partial(
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


def run_resid_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: FN_OF_ANSWERS,
    answer_tokens: Int[Tensor, "batch pair correct"],
    seq_pos: Union[None, int],
    direction_label: str,
    batch_size: int,
    all_layers: bool = True,
) -> float:
    """
    Runs patching experiment for a given position and layer.
    seq_pos=None means all positions.
    """
    model.reset_hooks()
    if all_layers:
        patching_nodes = [
            Node('resid_pre', layer=layer, seq_pos=seq_pos) 
            for layer in range(model.cfg.n_layers)
        ]
    else:
        layer = extract_layer_from_string(direction_label)
        act_name, hook_layer = get_resid_name(layer, model)
        node_name = act_name.split('hook_')[-1]
        patching_nodes = Node(node_name, layer=hook_layer, seq_pos=seq_pos)
    result =  batched_act_patch(
        model=model,
        orig_input=orig_input,
        new_cache=new_cache,
        batch_size=batch_size,
        patching_nodes=patching_nodes,
        patching_metric=patching_metric,
        answer_tokens=answer_tokens,
        verbose=True,
        disable=True,
    ).item() * 100
    return result
#%%
def run_head_patching(
    model: HookedTransformer,
    orig_input: Float[Tensor, "batch seq"],
    new_cache: ActivationCache,
    patching_metric: FN_OF_ANSWERS,
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
dataset_cache = dict()


def get_dataset_cached(
    model: HookedTransformer,
    prompt_type: PromptType,
    scaffold: ReviewScaffold,
    min_tokens: int = 0,
    max_tokens: int = 100,
    center: bool = True,
):
    key = (
        model.cfg.model_name,
        prompt_type.value,
        scaffold.value,
        min_tokens,
        max_tokens,
        center,
    )
    if key in dataset_cache:
        return dataset_cache[key]
    clean_corrupt_data = get_dataset(
        model, torch.device("cpu"), prompt_type=prompt_type, scaffold=scaffold
    )

    # FIXME: need to uncomment if using max_tokens
    # # Filter by padding
    # clean_corrupt_data = clean_corrupt_data.restrict_by_padding(
    #     min_tokens=min_tokens, max_tokens=max_tokens
    # )

    dataset_cache[key] = clean_corrupt_data
    return dataset_cache[key]


#%%
def get_result_cached(
    patching_metric_base: PatchingMetric, 
    prompt_type: PromptType,
    position: str,
    direction_label: str, 
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    heads: Optional[List[Tuple[int]]] = None,
    scaffold: ReviewScaffold = ReviewScaffold.PLAIN,
    center: bool = True,
    all_layers: bool = True,
    min_tokens: int = 0,
    max_tokens: int = 25,
    disable_tqdm: bool = True,
):
    use_csv = USE_CACHE and heads is None and not all_layers
    txt_name = (
        patching_metric_base.__name__.replace('_denoising', '') +
        f"_{prompt_type.value}_{scaffold}_{min_tokens}_{max_tokens}_{position}_"
        f"{direction_label}.txt"
    )
    if use_csv and is_file(txt_name, model):
        return float(load_text(txt_name, model))
    result = get_results_for_direction_and_position(
        patching_metric_base=patching_metric_base, 
        prompt_type=prompt_type,
        position=position,
        direction_label=direction_label,
        direction=direction,
        model=model,
        device=device,
        batch_size=batch_size,
        heads=heads,
        scaffold=scaffold,
        center=center,
        all_layers=all_layers,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        disable_tqdm=disable_tqdm,
    )
    if use_csv:
        save_text(str(result), txt_name, model)
    return result

    
#%%
def get_results_for_direction_and_position(
    patching_metric_base: PatchingMetric, 
    prompt_type: PromptType,
    position: str,
    direction_label: str, 
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    heads: Optional[List[Tuple[int]]] = None,
    scaffold: ReviewScaffold = ReviewScaffold.PLAIN,
    center: bool = True,
    all_layers: bool = True,
    min_tokens: int = 0,
    max_tokens: int = 25,
    disable_tqdm: bool = True,
) -> float:
    if heads is None and all_layers:
        names_filter = lambda name: 'resid_pre' in name
    elif heads is None:
        layer = extract_layer_from_string(direction_label)
        resid_name = get_resid_name(layer, model)[0]
        names_filter = lambda name: name == resid_name
    else:
        names_filter = lambda name: 'result' in name
    model.reset_hooks()
    clean_corrupt_data = get_dataset_cached(
        model=model,
        prompt_type=prompt_type,
        scaffold=scaffold,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        center=center,
    )
    patching_dataset: CleanCorruptedCacheResults = clean_corrupt_data.run_with_cache(
        model, 
        names_filter=names_filter,
        batch_size=batch_size,
        device=device,
        disable_tqdm=disable_tqdm,
        center=center,
    )
    # print(patching_dataset.clean_logit_diff, patching_dataset.corrupted_logit_diff)
    example_prompt = model.to_str_tokens(clean_corrupt_data.all_prompts[0])
    if position == 'ALL':
        seq_pos = None
    else:
        seq_pos = prompt_type.get_placeholder_positions(example_prompt)[position][-1]
    if patching_metric_base == PatchingMetric.LOGIT_DIFF_DENOISING:
        clean_value = patching_dataset.clean_logit_diff
        corrupt_value = patching_dataset.corrupted_logit_diff
    elif patching_metric_base == PatchingMetric.PROB_DIFF_DENOISING:
        clean_value = patching_dataset.clean_prob_diff
        corrupt_value = patching_dataset.corrupted_prob_diff
    elif patching_metric_base == PatchingMetric.LOGIT_FLIP_DENOISING:
        clean_value = patching_dataset.clean_accuracy
        corrupt_value = patching_dataset.corrupted_accuracy
    else:
        raise ValueError(f"Unknown patching metric {patching_metric_base}")
    patching_metric = partial(
        patching_metric_base, 
        flipped_value=corrupt_value,
        clean_value=clean_value,
        return_tensor=True,
    )
    new_cache = create_cache_for_dir_patching(
        patching_dataset.clean_cache, 
        patching_dataset.corrupted_cache, 
        direction, 
        model,
    )
    if heads is None:
        return run_resid_patching(
            model=model, 
            orig_input=clean_corrupt_data.corrupted_tokens, 
            new_cache=new_cache, 
            batch_size=batch_size,
            patching_metric=patching_metric, 
            answer_tokens=clean_corrupt_data.answer_tokens,
            seq_pos=seq_pos, 
            direction_label=direction_label,
            all_layers=all_layers,
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
    patching_metric_base: PatchingMetric, 
    prompt_types: Iterable[PromptType], 
    direction_labels: List[str], 
    directions: List[Float[Tensor, "d_model"]],
    model: HookedTransformer,
    device: Optional[torch.device] = None,
    heads: Optional[List[Tuple[int]]] = None,
    disable_tqdm: bool = False,
    scaffold: ReviewScaffold = ReviewScaffold.PLAIN,
    batch_size: int = 16,
    all_layers: bool = True,
    proj: Optional[Literal["para", "ortho"]] = None,
) -> Float[pd.DataFrame, "direction prompt"]:
    use_heads_label = "resid" if heads is None else "attn_result"
    metric_label = patching_metric_base.__name__.replace('_base', '').replace('_denoising', '')
    proj_label = "" if proj is None else f"_{proj}"
    all_but_one_label = "_all_but_one" if "all_but_one" in direction_labels[0] else ""
    csv_path = (
        f"direction_patching_{metric_label}_{use_heads_label}_"
        f"{scaffold.value}{proj_label}{all_but_one_label}.csv"
    )
    # if use_cache and is_file(csv_path, model):
    #     return get_csv(csv_path, model, index_col=0, header=[0, 1])
    bar = tqdm(
        itertools.product(prompt_types, zip(direction_labels, directions)), 
        total=len(prompt_types) * len(direction_labels),
        disable=disable_tqdm,
    )
    results = pd.DataFrame(index=direction_labels, dtype=float)
    results.index = results.index.str.replace(".npy", f"{proj_label}.npy")
    for prompt_type, (direction_label, raw_direction) in bar:
        direction = raw_direction.clone()
        bar.set_description(f"{prompt_type.value} {direction_label}, batch_size={batch_size}, proj={PROJ}")
        match = re.match(DIRECTION_PATTERN, direction_label)
        assert match is not None, (
            f"Direction label {direction_label} does not match pattern {DIRECTION_PATTERN}"
        )
        method, _, _, layer = match.groups()
        if proj is not None:
            base_direction_label = [
                label for label in direction_labels
                if prompt_type.value in label and method in label and layer in label
            ]
            assert len(base_direction_label) == 1, (
                f"Could not find base direction for {direction_label}. "
                f"Found {base_direction_label}. "
                f"Prompt type {prompt_type.value}, method {method}, layer {layer}."

            )
            base_direction_label = base_direction_label[0]
            base_direction_label = base_direction_label.replace("_all_but_one", "*")
            insample = base_direction_label == direction_label 
            will_zero = insample and (proj == "ortho")
            base_direction = get_direction(base_direction_label, model)
            base_direction /= base_direction.norm(keepdim=True)
            direction /= direction.norm(keepdim=True)
            assert np.isclose(direction.norm().item(), 1.0)
            assert np.isclose(base_direction.norm().item(), 1.0)
            if proj == "para":
                direction = base_direction * (base_direction @ direction)
            elif will_zero:
                direction = torch.zeros_like(direction)
            elif proj == "ortho":
                direction -= base_direction * (base_direction @ direction)
            else:
                raise ValueError(f"Unknown projection {proj}")
            direction_label = direction_label.replace(".npy", f"{proj_label}.npy")
            if direction.norm() > 0:
                direction /= direction.norm(keepdim=True)
                assert np.isclose(
                    direction.norm().item(), 1.0
                ), (
                    f"Projected direction {direction_label} has norm {direction.norm()}"
                )
            assert not torch.isnan(direction).any()
            
        assert direction_label in results.index, (
            f"Direction label {direction_label} not in results index"
        )
        # placeholders = prompt_type.get_placeholders() + ['ALL']
        placeholders = ['ALL']
        for position in placeholders:
            column = pd.MultiIndex.from_tuples([(prompt_type.value, position)], names=['prompt', 'position'])
            if (direction != 0).any():
                result = get_result_cached(
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
                    all_layers=all_layers,
                )
            else:
                result = 0
            assert not np.isnan(result), (
                f"Result is nan for {prompt_type.value}, {position}, "
                f"direction_label={direction_label}, direction={direction}"
            )
            # Ensure the column exists
            if (prompt_type.value, position) not in results.columns:
                results[column] = np.nan
            results.loc[direction_label, column] = result
            torch.cuda.empty_cache()
    results.columns = pd.MultiIndex.from_tuples(
        results.columns,
        names=['prompt', 'position']
    )
    to_csv(results, csv_path.replace(".csv", ""), model, index=True)
    return results
# %%
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
    "EleutherAI/pythia-2.8b": 64,
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
    DIRECTIONS, DIRECTION_LABELS = get_directions(model)
    results = get_results_for_metric(
        metric, PROMPT_TYPES, DIRECTION_LABELS, DIRECTIONS, model, device, heads, 
        scaffold=SCAFFOLD, batch_size=batch_size,
        all_layers=ALL_LAYERS,
        proj=PROJ,
    )
    print(results)
#%%

