#%%
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import einops
from tqdm.notebook import tqdm
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Iterable, List, Tuple, Union, Optional, Dict
from jaxtyping import Float, Int
from torch import Tensor
from functools import partial
import copy
from enum import Enum
import os
import itertools
from IPython.display import display, HTML
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import test_prompt
import transformer_lens.evals as evals
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import wandb
from utils.store import save_array, save_html, update_csv, get_csv, eval_csv, is_file, load_pickle
from utils.prompts import PromptType
from utils.residual_stream import ResidualStreamDataset
from utils.classification import train_classifying_direction, ClassificationMethod
from utils.das import FittingMethod, train_das_subspace
from utils.classifier import HookedClassifier
from utils.treebank import ReviewScaffold
#%%
# ============================================================================ #
# model loading
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODELS = [
    'gpt2-small',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
]
METHODS = [
    # ClassificationMethod.KMEANS,
    # ClassificationMethod.PCA,
    # ClassificationMethod.SVD,
    # ClassificationMethod.MEAN_DIFF,
    # ClassificationMethod.LOGISTIC_REGRESSION,
    FittingMethod.DAS,
]
PROMPT_TYPES = [
    # PromptType.SIMPLE_TRAIN,
    # PromptType.SIMPLE_TEST,
    # PromptType.CLASSIFICATION_4,
    # PromptType.SIMPLE_ADVERB,
    # PromptType.SIMPLE_MOOD,
    # PromptType.SIMPLE_FRENCH,
    # PromptType.PROPER_NOUNS,
    # PromptType.MEDICAL,
    PromptType.TREEBANK_TRAIN,
]
SCAFFOLD = ReviewScaffold.CONTINUATION
LAYERS = [
    "0", "1", "{n} // 4", "{n} // 3", "{n} // 2", 
    "2 * {n} // 3", "3 * {n} // 4", "{n} - 1", "{n}"
]
BATCH_SIZES = {
    'gpt2-small': 128,
    'gpt2-medium': 64,
    'gpt2-large': 16,
    'gpt2-xl': 8,
    'EleutherAI/pythia-160m': 128,
    'EleutherAI/pythia-410m': 64,
    'EleutherAI/pythia-1.4b': 32,
    'EleutherAI/pythia-2.8b': 16,
}
#%%
def get_model(name: str):
    model = HookedTransformer.from_pretrained(
        name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    ).requires_grad_(False)
    model.name = name
    return model
#%%
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
#%%
# ============================================================================ #
# DAS sweep with wandb
#%%
# for sweep_model in MODELS:
#     sweep_model = get_model(sweep_model)
#     sweep_layer = sweep_model.cfg.n_layers // 2
#     sweep_func = partial(
#         train_das_subspace,
#         model=sweep_model, device=device,
#         train_type=PromptType.TREEBANK_TRAIN, train_pos=None, train_layer=sweep_layer,
#         test_type=PromptType.TREEBANK_DEV, test_pos=None, test_layer=sweep_layer,
#         wandb_enabled=True,
#         downcast=False,
#         scaffold=ReviewScaffold.CONTINUATION,
#         data_requires_grad=False,
# #     )
#     sweep_config = {
#         "name": f"DAS subspace dimension ({sweep_model.cfg.model_name})",
#         "method": "grid",
#         "metric": {"name": "loss", "goal": "minimize"},
#         "parameters": {
#             "lr": {"values": [1e-3]},
#             "epochs": {"values": [1]},
#             "weight_decay": {"values": [0.0]},
#             "betas": {"values": [[0.9, 0.999]]},
#             "d_das": {"values": [1, 2, 4, 8, 16]},
#             "batch_size": {"values": [64]},
#         },
#     }
#     sweep_id = wandb.sweep(sweep_config, project=train_das_subspace.__name__)
#     wandb.agent(sweep_id, function=sweep_func)
#%%
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
#%%
# ============================================================================ #
# Training loop

BAR = tqdm(
    itertools.product(MODELS, PROMPT_TYPES, PROMPT_TYPES, METHODS),
    total=len(PROMPT_TYPES) ** 2 * len(MODELS) * len(METHODS),
)
model = None
for model_name, train_type, test_type, method in BAR:
    BAR.set_description(
        f"model:{model_name},"
        f"trainset:{train_type.value},"
        f"testset:{test_type.value},"
        f"method:{method.value}"
    )
    if model is None or model.name != model_name:
        del model
        model = get_model(model_name)
    if 'test' in train_type.value:
        # Don't train on test sets
        continue
    train_placeholders = train_type.get_placeholders() + ["ALL"]
    test_placeholders = test_type.get_placeholders() + ["ALL"]
    layers = [eval(layer.format(n=model.cfg.n_layers)) for layer in LAYERS]
    placeholders_layers = list(itertools.product(
        train_placeholders, 
        test_placeholders,
        layers
    ))
    assert len(placeholders_layers) > 0
    kwargs = dict()
    if method in (ClassificationMethod.PCA, ClassificationMethod.SVD):
        kwargs['n_components'] = 2
    layers_bar = tqdm(placeholders_layers, leave=False)
    for train_pos, test_pos, train_layer in layers_bar:
        test_layer = train_layer # Don't train/eval on different layers
        layers_bar.set_description(
            f"train_pos:{train_pos},"
            f"test_pos:{test_pos},"
            f"train_layer:{train_layer},"
            f"test_layer:{test_layer}"
        )
        if train_pos == 'VRB':
            # Don't train on verbs as sample size is too small
            continue
        save_path = f"{method.value}_{train_type.value}_{train_pos}_layer{train_layer}.npy"
        if is_file(save_path, model):
            # print(f"Skipping because file already exists: {save_path}")
            continue
        if train_pos == 'ALL':
            train_pos = None
        if test_pos == "ALL":
            test_pos = None
        if method == FittingMethod.DAS:
            if train_type != test_type or train_layer != test_layer:
                continue
            _, das_path = train_das_subspace(
                model, device,
                train_type, train_pos, train_layer,
                test_type, test_pos, test_layer,
                scaffold=SCAFFOLD,
                wandb_enabled=False,
                epochs = 1 if "treebank" in train_type.value else 64,
                batch_size=BATCH_SIZES[model_name],
            )
            print(f"Saving DAS direction to {das_path}")
            
        else:
            trainset = ResidualStreamDataset.get_dataset(
                model, device, prompt_type=train_type, scaffold=SCAFFOLD
            )
            testset = ResidualStreamDataset.get_dataset(
                model, device, prompt_type=test_type, scaffold=SCAFFOLD
            )
            cls_path = train_classifying_direction(
                trainset, train_pos, train_layer,
                testset, test_pos, test_layer,
                method,
                **kwargs
            )
            print(f"Saving classification direction to {cls_path}")
#%%
# ============================================================================ #
# Summary stats

#%%
def hide_nan(val):
    return '' if pd.isna(val) else f"{val:.1%}"
#%%
def plot_accuracy_similarity(df, label: str, model: HookedTransformer):
    df = df.loc[
        df.train_set.isin(['simple_train']) & 
        df.train_pos.isin(['ADJ']) &
        df.test_set.isin(['simple_train', 'simple_test', 'simple_adverb'])
    ]
    if df.empty:
        print(f"No data to plot for {label}")
        return
    accuracy_styler = df.pivot(
        index=["train_set", "train_pos", "train_layer", ],
        columns=["test_set", "test_pos"],
        values="accuracy",
    ).sort_index(axis=0).sort_index(axis=1).style.background_gradient(cmap="Reds").format(hide_nan).set_caption(f"{label} accuracy ({model})")
    save_html(accuracy_styler, f"{label}_accuracy", model)
    display(accuracy_styler)
    similarity_styler = df.pivot(
        index=["train_set",  "train_pos", "train_layer",],
        columns=["test_set", "test_pos"],
        values="similarity",
    ).sort_index(axis=0).sort_index(axis=1).style.background_gradient(cmap="Reds").format(hide_nan).set_caption(f"{label} cosine similarities ({model})")
    save_html(similarity_styler, f"{label}_similarity", model)
    display(similarity_styler)
#%%
for model in MODELS:
    fitting_stats = get_csv("direction_fitting_stats", model)
    for method in METHODS:
        plot_accuracy_similarity(
            fitting_stats.loc[fitting_stats.method.eq(method.value)],
            method.value,
            model,
        )
#%%
# ============================================================================ #
# PCA/SVD plots

#%%
# hacky functions for reading from nested CSV
def tensor_from_str(
    string: str,
    dim: int = 2,
) -> Float[Tensor, "batch dim"]:
    if dim == 2:
        split_list = string.split('\n')
        split_list = [el.replace('[', '').replace(']', '').strip().split() for el in split_list]
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
    split_list = string.split(',')
    split_list = [el.replace('[', '').replace(']', '').replace("'", "").strip().replace(" ", "_") for el in split_list]
    return split_list
#%%
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
    train_label: str = 'train', 
    test_label: str = 'test',
):
    if isinstance(model, HookedTransformer):
        model = model.cfg.model_name
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_pcs[:, 0],
            y=train_pcs[:, 1],
            text=train_str_labels,
            mode="markers",
            marker=dict(
                color=train_true_labels.to(dtype=torch.int32),
                colorscale="RdBu",
                opacity=0.8,
            ),
            name=f"{method.value} in-sample ({train_label})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_pcs[:, 0],
            y=test_pcs[:, 1],
            text=test_str_labels,
            mode="markers",
            marker=dict(
                color=test_true_labels.to(dtype=torch.int32),
                colorscale="RdBu",
                opacity=0.8,
                symbol="square",
            ),
            name=f"{method.value} out-of-sample ({test_label})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pca_centroids[:, 0],
            y=pca_centroids[:, 1],
            mode="markers",
            marker=dict(color='green', symbol='x', size=10),
            name="Centroids",
        )
    )
    fig.update_layout(
        title=(
            f"{method.value} in and out of sample "
            f"({model})"
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    save_html(
        fig, f"{method.value}_{train_label}_{test_label}", model
    )
    return fig

# %%
def plot_components_from_cache(
    model: HookedTransformer,
    method: ClassificationMethod, 
    train_set: PromptType, train_pos: str, train_layer: int,
    test_set: PromptType, test_pos: str, test_layer: int,
):
    plot_df = get_csv("pca_svd_plot", model)
    plot_df = plot_df.loc[
        (plot_df.method == method.value) &
        (plot_df.train_set == train_set.value) &
        (plot_df.train_pos == train_pos) &
        (plot_df.train_layer == train_layer) &
        (plot_df.test_set == test_set.value) &
        (plot_df.test_pos == test_pos) &
        (plot_df.test_layer == test_layer)
    ]
    assert len(plot_df) == 1, f"Found {len(plot_df)} rows for query"
    plot_df = plot_df.iloc[0]
    train_pcs: Float[Tensor, "batch d_model"] = tensor_from_str(plot_df.train_pcs)
    train_str_labels: List[str] = list_from_str(plot_df.train_str_labels)
    train_true_labels: Float[Tensor, "batch d_model"] = tensor_from_str(plot_df.train_true_labels, dim=1)
    test_pcs: Float[Tensor, "batch d_model"] = tensor_from_str(plot_df.test_pcs)
    test_str_labels: List[str] = list_from_str(plot_df.test_str_labels)
    test_true_labels: Float[Tensor, "batch d_model"] = tensor_from_str(plot_df.test_true_labels, dim=1)
    pca_centroids: Float[Tensor, "centroid d_model"] = tensor_from_str(plot_df.pca_centroids)
    train_label = f"{train_set.value}_{train_pos}_layer{train_layer}"
    test_label = f"{test_set.value}_{test_pos}_layer{test_layer}"
    fig = plot_pca_svd_2d(
        model,
        method, 
        train_pcs, train_str_labels, train_true_labels,
        test_pcs, test_str_labels, test_true_labels,
        pca_centroids, 
        train_label=train_label, test_label=test_label,
    )
    return fig
#%%
for model in MODELS:
    for method in (ClassificationMethod.PCA, ClassificationMethod.SVD):
        fig = plot_components_from_cache(
            model,
            ClassificationMethod.PCA,
            PromptType.SIMPLE_TRAIN, 'ADJ', 0,
            PromptType.SIMPLE_TEST, 'ADJ', 0,
        )
        fig.show()
#%%
