# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import einops
from tqdm.auto import tqdm
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Iterable, List, Tuple, Union, Optional, Dict
from jaxtyping import Float, Int
from torch import Tensor
from functools import partial
from enum import Enum
import itertools
from IPython.display import display, HTML
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import test_prompt
import transformer_lens.evals as evals
import wandb
from utils.store import (
    save_array,
    save_html,
    update_csv,
    get_csv,
    eval_csv,
    is_file,
    load_pickle,
    save_pdf,
)
from utils.prompts import PromptType, get_dataset
from utils.residual_stream import ResidualStreamDataset
from utils.classification import train_classifying_direction, ClassificationMethod
from utils.das import GradientMethod, train_das_subspace
from utils.classifier import HookedClassifier
from utils.treebank import ReviewScaffold
from utils.methods import FittingMethod

# %%
# ============================================================================ #
# model loading
SKIP_IF_EXISTS = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS = [
    "mistralai/Mistral-7B-v0.1",
    # "stablelm-base-alpha-3b",
    # "stablelm-base-alpha-7b",
    # 'gpt2-small',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    # 'EleutherAI/pythia-160m',
    # 'EleutherAI/pythia-410m',
    # 'EleutherAI/pythia-1.4b',
    # "EleutherAI/pythia-6.9b",
]
METHODS = [
    # ClassificationMethod.KMEANS,
    # ClassificationMethod.PCA,
    # ClassificationMethod.SVD,
    # ClassificationMethod.MEAN_DIFF,
    # ClassificationMethod.LOGISTIC_REGRESSION,
    GradientMethod.DAS,
    # GradientMethod.DAS2D,
    # GradientMethod.DAS3D,
]
TRAIN_TYPES = [
    PromptType.SIMPLE_TRAIN,
    # PromptType.SIMPLE_ADVERB,
    # PromptType.SIMPLE_BOOK,
    # PromptType.SIMPLE_RES,
    # PromptType.SIMPLE_PRODUCT,
    # PromptType.CLASSIFICATION_4,
    # PromptType.SIMPLE_ADVERB,
    # PromptType.SIMPLE_MOOD,
    # PromptType.SIMPLE_FRENCH,
    # PromptType.PROPER_NOUNS,
    # PromptType.MEDICAL,
    # PromptType.TREEBANK_TRAIN,
]
TEST_TYPES = [
    # PromptType.SIMPLE_TEST,
    # PromptType.SIMPLE_ADVERB,
    PromptType.NONE,
]
SCAFFOLD = ReviewScaffold.CONTINUATION
BATCH_SIZES = {
    "gpt2-small": 128,
    "gpt2-medium": 64,
    "gpt2-large": 32,
    "gpt2-xl": 8,
    "EleutherAI/pythia-160m": 128,
    "EleutherAI/pythia-410m": 64,
    "EleutherAI/pythia-1.4b": 32,
    "EleutherAI/pythia-2.8b": 8,
    "EleutherAI/pythia-6.9b": 4,
    "stablelm-base-alpha-3b": 8,
    "stablelm-base-alpha-7b": 8,
    "mistralai/Mistral-7B-v0.1": 16,
}


# %%
def get_model(name: str):
    model = HookedTransformer.from_pretrained(
        name,
        torch_dtype=torch.float16,
        dtype="float16",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        move_to_device=True,
    ).requires_grad_(False)
    return model


# %%
def select_layers(
    n_layers: int,
):
    return list(range(n_layers + 1))  # FIXME: UNCOMMENT!!!
    # if n_layers <= 16:
    #     return [0] + list(range(1, n_layers + 1, 4))
    # elif n_layers <= 24:
    #     return [0] + list(range(1, n_layers + 1, 4))
    # elif n_layers <= 36:
    #     return [0] + list(range(1, n_layers + 1, 4))
    # elif n_layers <= 48:
    #     return [0] + list(range(1, n_layers + 1, 8))


# %%
# sweep_model = HookedClassifier.from_pretrained(
#     "data/gpt2-small/gpt2_imdb_classifier",
#     "gpt2_imdb_classifier_classification_head_weights",
#     "gpt2",
#     center_unembed=True,
#     center_writing_weights=True,
#     fold_ln=True,
#     refactor_factored_attn_matrices=True,
# )
# sweep_model.set_requires_grad(False)
# %%
# ============================================================================ #
# DAS sweep with wandb
# %%
# for sweep_model in MODELS:
#     sweep_model = get_model(sweep_model)
#     sweep_layer = sweep_model.cfg.n_layers // 2
#     sweep_func = partial(
#         train_das_subspace,
#         model=sweep_model, device=device,
#         train_type=PromptType.SIMPLE_TRAIN, train_pos="ADJ", train_layer=sweep_layer,
#         test_type=PromptType.SIMPLE_MOOD, test_pos=None, test_layer=sweep_layer,
#         wandb_enabled=True,
#         downcast=False,
#         # scaffold=ReviewScaffold.CONTINUATION,
#         data_requires_grad=False,
#     )
#     sweep_config = {
#         "name": f"DAS subspace dimension ({sweep_model.cfg.model_name})",
#         "method": "grid",
#         "metric": {"name": "loss", "goal": "minimize"},
#         "parameters": {
#             "lr": {"values": [1e-3]},
#             "epochs": {"values": [64]},
#             "weight_decay": {"values": [0.0]},
#             "betas": {"values": [[0.9, 0.999]]},
#             "d_das": {"values": [1, 2, 4, 8, 16]},
#             "batch_size": {"values": [64]},
#         },
#     }
#     sweep_id = wandb.sweep(sweep_config, project=train_das_subspace.__name__)
#     wandb.agent(sweep_id, function=sweep_func)
#     break
# %%
# model = get_model('gpt2-small')
# das_dirs, das_path = train_das_subspace(
#     model, device,
#     train_type=PromptType.TREEBANK_TRAIN, train_pos=None, train_layer=0,
#     test_type=PromptType.TREEBANK_DEV, test_pos=None, test_layer=0,
#     wandb_enabled=False,
#     downcast=False,
#     scaffold=ReviewScaffold.CONTINUATION,
#     data_requires_grad=False,
#     batch_size=8,
#     epochs=1,
# )
# %%
# ============================================================================ #
# Training loop
BAR = tqdm(
    itertools.product(MODELS, TRAIN_TYPES, TEST_TYPES, METHODS),
    total=len(TRAIN_TYPES) * len(TEST_TYPES) * len(MODELS) * len(METHODS),
)
model = None
for model_name, train_type, test_type, method in BAR:
    BAR.set_description(
        f"model:{model_name},"
        f"trainset:{train_type.value},"
        f"testset:{test_type.value},"
        f"method:{method.value}"
    )
    if model is None or model.cfg.model_name not in model_name:
        del model
        model = get_model(model_name)
    if "test" in train_type.value:
        # Don't train on test sets
        continue
    train_placeholders = train_type.get_placeholders()
    test_placeholders = test_type.get_placeholders()
    if len(train_placeholders) == 0:
        train_placeholders = ["ALL"]
    if len(test_placeholders) == 0:
        test_placeholders = ["ALL"]
    train_placeholders = train_placeholders[:1]
    test_placeholders = test_placeholders[:1]
    layers = select_layers(model.cfg.n_layers)
    if layers is None:
        continue
    placeholders_layers = list(
        itertools.product(train_placeholders, test_placeholders, layers)
    )
    assert len(placeholders_layers) > 0
    kwargs = dict()
    if method in (ClassificationMethod.PCA, ClassificationMethod.SVD):
        kwargs["n_components"] = 1
    layers_bar = tqdm(placeholders_layers, leave=False)
    for train_pos, test_pos, train_layer in layers_bar:
        test_layer = train_layer  # Don't train/eval on different layers
        layers_bar.set_description(
            f"train_pos:{train_pos},"
            f"test_pos:{test_pos},"
            f"train_layer:{train_layer},"
            f"test_layer:{test_layer}"
        )
        if train_pos == "VRB":
            # Don't train on verbs as sample size is too small
            print("Skipping because train_pos is VRB")
            continue
        save_path = (
            f"{method.value}_{train_type.value}_{train_pos}_layer{train_layer}.npy"
        )
        if SKIP_IF_EXISTS and is_file(save_path, model):
            print(f"Skipping because file already exists: {save_path}")
            continue
        if train_pos == "ALL":
            train_pos = None
        if test_pos == "ALL":
            test_pos = None
        if isinstance(method, GradientMethod):
            train_test_discrepancy = (
                test_type != PromptType.NONE
                and (train_type != test_type or train_layer != test_layer)
                and "treebank" in train_type.value
            )
            if train_test_discrepancy:
                print("Skipping due to train/test discrepancy")
                continue
            print("Calling train_das_subspace...")
            _, das_path = train_das_subspace(
                model,
                device,
                train_type,
                train_pos,
                train_layer,
                test_type,
                test_pos,
                test_layer,
                scaffold=SCAFFOLD,
                wandb_enabled=False,
                epochs=1 if "treebank" in train_type.value else 64,
                batch_size=BATCH_SIZES[model_name],
                d_das=method.get_dimension(),
            )
            print(f"Saving DAS direction to {das_path}")
            torch.cuda.empty_cache()
            print("Emptied CUDA cache")
        else:
            trainset = ResidualStreamDataset.get_dataset(
                model, device, prompt_type=train_type, scaffold=SCAFFOLD
            )
            testset = ResidualStreamDataset.get_dataset(
                model, device, prompt_type=test_type, scaffold=SCAFFOLD
            )
            assert trainset is not None
            cls_path = train_classifying_direction(
                trainset,
                train_pos,
                train_layer,
                testset,
                test_pos,
                test_layer,
                method,
                **kwargs,
            )
            print(f"Saving classification direction to {cls_path}")
# %%
# ============================================================================ #
# # END OF ACTUAL DIRECTION FITTING

# %%
# ============================================================================ #
# Summary stats


# %%
def hide_nan(val):
    return "" if pd.isna(val) else f"{val:.1%}"


# %%
def plot_accuracy_similarity(
    df,
    label: str,
    model: HookedTransformer,
    train_sets: Optional[Iterable[PromptType]] = None,
    train_positions: Optional[Iterable[str]] = None,
    test_sets: Optional[Iterable[PromptType]] = None,
):
    if train_sets is None:
        train_sets = [PromptType.SIMPLE_TRAIN]
    if train_positions is None:
        train_positions = ["ADJ"]
    if test_sets is None:
        test_sets = [
            PromptType.SIMPLE_TRAIN,
            PromptType.SIMPLE_TEST,
            PromptType.SIMPLE_ADVERB,
        ]
    df = df.loc[
        df.train_set.isin(train_sets)
        & df.train_pos.isin(train_positions)
        & df.test_set.isin(test_sets)
    ].copy()
    df.train_set = pd.Categorical(df.train_set, categories=train_sets, ordered=True)
    df.train_pos = pd.Categorical(
        df.train_pos, categories=train_positions, ordered=True
    )
    df.test_set = pd.Categorical(df.test_set, categories=test_sets, ordered=True)
    if df.empty:
        print(f"No data to plot for {label}")
        return
    accuracy_styler = (
        df.pivot(
            index=["train_set", "train_pos", "train_layer"],
            columns=["test_set", "test_pos"],
            values="accuracy",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
        .style.background_gradient(cmap="Reds")
        .format(hide_nan)
        .set_caption(f"{label} accuracy ({model})")
    )
    save_html(accuracy_styler, f"{label}_accuracy", model, static=True)
    display(accuracy_styler)
    similarity_styler = (
        df.pivot(
            index=[
                "train_set",
                "train_pos",
                "train_layer",
            ],
            columns=["test_set", "test_pos"],
            values="similarity",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
        .style.background_gradient(cmap="Reds")
        .format(hide_nan)
        .set_caption(f"{label} cosine similarities ({model})")
    )
    save_html(similarity_styler, f"{label}_similarity", model, static=True)
    display(similarity_styler)


# %%
for model in MODELS:
    fitting_stats = get_csv("direction_fitting_stats", model)
    for method in METHODS:
        plot_accuracy_similarity(
            fitting_stats.loc[fitting_stats.method.eq(method.value)].copy(),
            method.value,
            model,
            train_sets=["simple_train", "treebank_train"],
            train_positions=["ADJ", "ALL"],
            test_sets=["simple_test", "simple_adverb", "treebank_test"],
        )
# %%
# ============================================================================ #
# PCA/SVD plots


# %%
# hacky functions for reading from nested CSV
def tensor_from_str(
    string: str,
    dim: int = 2,
) -> Float[Tensor, "batch dim"]:
    if dim == 2:
        split_list = string.split("\n")
        split_list = [
            el.replace("[", "").replace("]", "").strip().split() for el in split_list
        ]
        split_list = [[float(el) for el in row] for row in split_list]
        return torch.tensor(split_list)
    elif dim == 1:
        s = string.replace("tensor([", "").replace("])", "")
        # Splitting by comma and stripping spaces to get boolean values
        bool_vals = [val.strip() == "True" for val in s.split(",")]
        # Creating a tensor from the list of booleans
        tensor_val = torch.tensor(bool_vals)
        return tensor_val


def list_from_str(
    string: str,
) -> List[str]:
    split_list = string.split(",")
    split_list = [
        el.replace("[", "").replace("]", "").replace("'", "").strip().replace(" ", "_")
        for el in split_list
    ]
    return split_list


# %%
def plot_pca_svd_2d(
    model: HookedTransformer,
    method: ClassificationMethod,
    train_pcs: Float[Tensor, "batch d_model"],
    train_str_labels: List[str],
    train_true_labels: Float[Tensor, "batch d_model"],
    test_pcs: Float[Tensor, "batch d_model"],
    test_str_labels: List[str],
    test_true_labels: Float[Tensor, "batch d_model"],
    pca_centroids: Float[Tensor, "centroid d_model"],
    train_label: str = "train",
    test_label: str = "test",
    colorscale=[[0, "darkred"], [1, "darkblue"]],
    opacity=0.8,
    marker_size=12,
):
    train_label = train_label.replace("simple_", "").replace("_layer0", "")
    test_label = test_label.replace("simple_", "").replace("_layer0", "")
    method_label = method.value.upper()

    if "ADJ" in train_label:
        train_str_labels = [label.split("_")[0] for label in train_str_labels]
    else:
        train_str_labels = [label.split("_")[1] for label in train_str_labels]
    if "ADJ" in test_label:
        test_str_labels = [label.split("_")[0] for label in test_str_labels]
    else:
        test_str_labels = [label.split("_")[1] for label in test_str_labels]

    if isinstance(model, HookedTransformer):
        model = model.cfg.model_name
    fig = go.Figure()

    # in-sample dots
    fig.add_trace(
        go.Scatter(
            x=train_pcs[:, 0],
            y=train_pcs[:, 1],
            text=train_str_labels,
            textposition="bottom center",
            textfont=dict(size=16),
            mode="markers+text",
            marker=dict(
                color=train_true_labels.to(dtype=torch.int32),
                colorscale=colorscale,
                opacity=opacity,
                maxdisplayed=10,
                size=marker_size,
            ),
            name=f"{method_label} IS ({train_label})",
        )
    )

    # out-of-sample squares
    fig.add_trace(
        go.Scatter(
            x=test_pcs[:, 0],
            y=test_pcs[:, 1],
            text=test_str_labels,
            textposition="bottom center",
            textfont=dict(size=16),
            mode="markers+text",
            marker=dict(
                color=test_true_labels.to(dtype=torch.int32),
                colorscale=colorscale,
                opacity=opacity,
                symbol="square",
                size=marker_size,
                maxdisplayed=5,
            ),
            name=f"{method_label} OOS ({test_label})",
        )
    )

    # centroids (Xs)
    fig.add_trace(
        go.Scatter(
            x=pca_centroids[:, 0],
            y=pca_centroids[:, 1],
            mode="markers",
            marker=dict(color="green", symbol="x", size=10),
            name="Centroids",
        )
    )
    oos_label = (
        "sample" if "ADJ" in train_label and "ADJ" in test_label else "distribution"
    )
    fig.update_layout(
        title={
            "text": f"{method_label} in and out of {oos_label} ({model})",
            "font": {"size": 24},  # Adjust the size as needed
        },
        xaxis_title="PC1",
        yaxis_title="PC2",
        title_x=0.5,
        legend=dict(
            x=0,  # Adjust this value as needed
            y=-0.2,  # Adjust this value to position the legend below the plot
            orientation="h",  # Set the orientation to horizontal
            font=dict(size=16),  # Adjust the size as needed
        ),
        xaxis=dict(
            title_font=dict(size=18),  # Adjust the size as needed
            tickfont=dict(size=16),  # Adjust the size as needed
        ),
        yaxis=dict(
            title_font=dict(size=18),  # Adjust the size as needed
            tickfont=dict(size=16),  # Adjust the size as needed
        ),
    )
    save_html(fig, f"{method.value}_{train_label}_{test_label}", model)
    save_pdf(fig, f"{method.value}_{train_label}_{test_label}", model)
    return fig


# %%
def plot_components_from_cache(
    model: HookedTransformer,
    method: ClassificationMethod,
    train_set: PromptType,
    train_pos: str,
    train_layer: int,
    test_set: PromptType,
    test_pos: str,
    test_layer: int,
):
    plot_df = get_csv("pca_svd_plot", model, index_col=None)
    assert not plot_df.empty, "PCA data completely missing"
    plot_df = plot_df.loc[
        (plot_df.method == method.value)
        & (plot_df.train_set == train_set.value)
        & (plot_df.train_pos == train_pos)
        & (plot_df.train_layer == train_layer)
        & (plot_df.test_set == test_set.value)
        & (plot_df.test_pos == test_pos)
        & (plot_df.test_layer == test_layer)
    ]
    assert not plot_df.empty, (
        f"Missing PCA data for query: "
        f"method={method.value}, "
        f"train_set={train_set.value}, "
        f"train_pos={train_pos}, "
        f"train_layer={train_layer}, "
        f"test_set={test_set.value}, "
        f"test_pos={test_pos}, "
        f"test_layer={test_layer}"
    )
    assert len(plot_df) == 1, f"Found {len(plot_df)} rows for query"
    plot_df = plot_df.iloc[0]
    train_pcs: Float[Tensor, "batch d_model"] = tensor_from_str(plot_df.train_pcs)
    train_str_labels: List[str] = list_from_str(plot_df.train_str_labels)
    train_true_labels: Float[Tensor, "batch d_model"] = tensor_from_str(
        plot_df.train_true_labels, dim=1
    )
    test_pcs: Float[Tensor, "batch d_model"] = tensor_from_str(plot_df.test_pcs)
    test_str_labels: List[str] = list_from_str(plot_df.test_str_labels)
    test_true_labels: Float[Tensor, "batch d_model"] = tensor_from_str(
        plot_df.test_true_labels, dim=1
    )
    pca_centroids: Float[Tensor, "centroid d_model"] = tensor_from_str(
        plot_df.pca_centroids
    )
    train_label = f"{train_set.value}_{train_pos}_layer{train_layer}"
    test_label = f"{test_set.value}_{test_pos}_layer{test_layer}"
    fig = plot_pca_svd_2d(
        model,
        method,
        train_pcs,
        train_str_labels,
        train_true_labels,
        test_pcs,
        test_str_labels,
        test_true_labels,
        pca_centroids,
        train_label=train_label,
        test_label=test_label,
    )
    return fig


# %%
for model in ("gpt2-small",):
    fig = plot_components_from_cache(
        model,
        ClassificationMethod.PCA,
        PromptType.SIMPLE_TRAIN,
        "ADJ",
        0,
        PromptType.SIMPLE_TEST,
        "ADJ",
        0,
    )
    fig.show()
    save_pdf(fig, "pca_train_test_adjectives_layer_0", model)
    save_html(fig, "pca_train_test_adjectives_layer_0", model)
    save_pdf(fig, "pca_train_test_adjectives_layer_0", model)
# %%
for model in ("gpt2-small",):
    fig = plot_components_from_cache(
        model,
        ClassificationMethod.PCA,
        PromptType.SIMPLE_TRAIN,
        "ADJ",
        0,
        PromptType.SIMPLE_TEST,
        "VRB",
        0,
    )
    fig.show()
    save_pdf(fig, "pca_train_adjectives_test_verbs_layer_0", model)
    save_html(fig, "pca_train_adjectives_test_verbs_layer_0", model)
    save_pdf(fig, "pca_train_test_adjectives_layer_0", model)
# %%
