#%%
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import einops
from tqdm.notebook import tqdm
import random
import pandas as pd
from pathlib import Path
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
import circuitsvis as cv
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformer_lens.utils import test_prompt
import transformer_lens.evals as evals
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import wandb
from utils.store import save_array, save_html, update_csv, get_csv, eval_csv
from utils.prompts import get_dataset, filter_words_by_length, PromptType
from utils.residual_stream import ResidualStreamDataset
from utils.classification import train_kmeans, CSV_COLS
#%%
# ============================================================================ #
# model loading

#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'gpt2-small'
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.name = MODEL_NAME
#%%
PROMPT_TYPES = [
    PromptType.SIMPLE_TRAIN,
    PromptType.SIMPLE_TEST,
    PromptType.SIMPLE_ADVERB,
    PromptType.SIMPLE_FRENCH,
    PromptType.PROPER_NOUNS,
    PromptType.MEDICAL,
]
LAYERS = list(range(model.cfg.n_layers + 1))
BAR = tqdm(
    itertools.product(PROMPT_TYPES, LAYERS, PROMPT_TYPES, LAYERS),
    total=len(PROMPT_TYPES) ** 2 * len(LAYERS) ** 2,
)
for train_type, train_layer, test_type, test_layer in BAR:
    BAR.set_description(f"trainset:{train_type.value}, train_layer:{train_layer}, testset:{test_type.value}, test_layer:{test_layer}")
    if train_layer != test_layer or 'test' in train_type.value:
        # Don't train/eval on different layers
        # Don't train on test sets
        continue
    placeholders = itertools.product(
        train_type.get_placeholders(), test_type.get_placeholders()
    )
    for train_pos, test_pos in placeholders:
        if train_pos == 'VRB':
            # Don't train on verbs as sample size is too small
            continue
        query = (
            f"(train_set == '{train_type.value}') & "
            f"(test_set == '{test_type.value}') & "
            f"(train_layer == {train_layer}) & "
            f"(test_layer == {test_layer}) &"
            f"(train_pos == '{train_pos}') & "
            f"(test_pos == '{test_pos}')"
        )
        # if eval_csv(query, "km_stats", model):
        #     continue

        trainset = ResidualStreamDataset.get_dataset(model, device, prompt_type=train_type)
        testset = ResidualStreamDataset.get_dataset(model, device, prompt_type=test_type)
        train_kmeans(
            trainset, train_layer,
            testset, test_layer,
        )
        train_kmeans(
            trainset, train_layer,
            testset, test_layer,
            pca_components=2,
        )
#%%
km_stats = get_csv("km_stats", model, key_cols=CSV_COLS)
km_stats = km_stats.loc[
    km_stats.method.eq('kmeans')
    km_stats.train_set.isin(['simple_train']) & 
    km_stats.train_pos.isin(['ADJ']) &
    km_stats.test_set.isin(['simple_train', 'simple_test', 'simple_adverb'])
]
km_stats
#%%
km_stats.train_pos.value_counts()
#%%
def hide_nan(val):
    return '' if pd.isna(val) else f"{val:.1%}"
#%%
accuracy_styler = km_stats.pivot(
    index=["train_set", "train_pos", "train_layer", ],
    columns=["test_set", "test_pos"],
    values="accuracy",
).sort_index(axis=0).sort_index(axis=1).style.background_gradient(cmap="Reds").format(hide_nan).set_caption(f"K-means accuracy ({model.name})")
save_html(accuracy_styler, "km_accuracy", model)
display(accuracy_styler)
#%%
similarity_styler = km_stats.pivot(
    index=["train_set",  "train_pos", "train_layer",],
    columns=["test_set", "test_pos"],
    values="similarity",
).sort_index(axis=0).sort_index(axis=1).style.background_gradient(cmap="Reds").format(hide_nan).set_caption(f"K-means cosine similarities ({model.name})")
save_html(similarity_styler, "km_similarity", model)
display(similarity_styler)
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
def plot_pca_2d(
    train_pcs, train_str_labels, train_true_labels, 
    test_pcs, test_str_labels, test_true_labels,
    pca_centroids, 
    train_label: str = 'train', 
    test_label: str = 'test',
):
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
            name=f"PCA in-sample ({train_label})",
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
            name=f"PCA out-of-sample ({test_label})",
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
            f"PCA in and out of sample "
            f"({model.name})"
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    save_html(
        fig, f"pca_{train_label}_{test_label}", model
    )
    return fig

# %%
def plot_pca_from_cache(
    train_set: PromptType, train_pos: str, train_layer: int,
    test_set: PromptType, test_pos: str, test_layer: int,
):
    plot_df = get_csv("pca_plot", model, key_cols=CSV_COLS)
    plot_df = plot_df.loc[
        (plot_df.train_set == train_set.value) &
        (plot_df.train_pos == train_pos) &
        (plot_df.train_layer == train_layer) &
        (plot_df.test_set == test_set.value) &
        (plot_df.test_pos == test_pos) &
        (plot_df.test_layer == test_layer)
    ]
    assert len(plot_df) == 1, f"Found {len(plot_df)} rows for query"
    plot_df = plot_df.iloc[0]
    train_pcs = tensor_from_str(plot_df.train_pcs)
    train_str_labels = list_from_str(plot_df.train_str_labels)
    train_true_labels = tensor_from_str(plot_df.train_true_labels, dim=1)
    test_pcs = tensor_from_str(plot_df.test_pcs)
    test_str_labels = list_from_str(plot_df.test_str_labels)
    test_true_labels = tensor_from_str(plot_df.test_true_labels, dim=1)
    pca_centroids = tensor_from_str(plot_df.pca_centroids)
    train_label = f"{train_set.value}_{train_pos}_layer{train_layer}"
    test_label = f"{test_set.value}_{test_pos}_layer{test_layer}"
    fig = plot_pca_2d(
        train_pcs, train_str_labels, train_true_labels,
        test_pcs, test_str_labels, test_true_labels,
        pca_centroids, 
        train_label=train_label, test_label=test_label,
    )
    return fig
#%%
fig = plot_pca_from_cache(
    PromptType.SIMPLE_TRAIN, 'ADJ', 0,
    PromptType.SIMPLE_TEST, 'ADJ', 0,
)
fig.show()
#%%
