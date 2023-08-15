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
import tqdm.auto as tqdm
import random
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Iterable, List, Tuple, Union, Optional
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
from utils.store import save_array, save_html
from utils.prompts import get_dataset, filter_words_by_length
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
# ============================================================================ #
# Data loading
POSITION_MARKERS = [
    ' perfect', ' amazing', ' impressive', 'rily', ' fabulous',
    'antly', 'ant', ' Mandela', ' diagnostic',
]
#%%
class KMeansDataset:

    def __init__(
        self, 
        prompt_strings: List[str], 
        prompt_tokens: Int[Tensor, "batch pos"], 
        binary_labels: Int[Tensor, "batch"],
        str_labels: List[str],
        embed_position: int,
        example: List[str],
        model: HookedTransformer,
        name: str,
    ) -> None:
        self.prompt_strings = prompt_strings
        self.prompt_tokens = prompt_tokens
        self.binary_labels = binary_labels
        self.str_labels = str_labels
        self.embed_position = embed_position
        self.example = example
        self.model = model
        self.name = name

    def __len__(self) -> int:
        return len(self.prompt_strings)
    
    def __eq__(self, other: object) -> bool:
        return set(self.prompt_strings) == set(other.prompt_strings)
    
    def embed(self, layer: int) -> Float[Tensor, "batch d_model"]:
        assert 0 <= layer <= self.model.cfg.n_layers
        hook = 'resid_pre'
        if layer == self.model.cfg.n_layers:
            hook = 'resid_post'
            layer -= 1
        _, cache = self.model.run_with_cache(
            self.prompt_tokens, return_type=None, names_filter = lambda name: hook in name
        )
        out: Float[Tensor, "batch pos d_model"] = cache[hook, layer]
        return out[:, self.embed_position, :].cpu().detach()

#%%
def get_km_dataset(
    model: HookedTransformer,
    device: torch.device,
    prompt_type: str = "simple_train",
    position_markers: List[str] = POSITION_MARKERS,
) -> KMeansDataset:
    all_prompts, answer_tokens, clean_tokens, _ = get_dataset(model, device, prompt_type=prompt_type)
    example_str_tokens = model.to_str_tokens(all_prompts[0])
    for marker in position_markers:
        if marker in example_str_tokens:
            embed_position = example_str_tokens.index(marker)
            break
    else:
        raise ValueError(f'Could not find position marker in example tokens {example_str_tokens}')
    example_prompt_indexed = [f"{i}:{tok}" for i, tok in enumerate(example_str_tokens)]
    clean_labels = answer_tokens[:, 0, 0] == answer_tokens[0, 0, 0]
    str_labels = [model.to_str_tokens(p)[embed_position] for p in all_prompts]
    return KMeansDataset(
        all_prompts,
        clean_tokens,
        clean_labels,
        str_labels,
        embed_position,
        example_prompt_indexed,
        model,
        prompt_type,
    )
#%% # helper functions
def split_by_label(
    adjectives: List[str], labels: Int[np.ndarray, "batch"]
) -> Tuple[List[str], List[str]]:
    first_cluster = [
        adj[1:]
        for i, adj in enumerate(adjectives) 
        if labels[i] == 0
    ]
    second_cluster = [
        adj[1:]
        for i, adj in enumerate(adjectives) 
        if labels[i] == 1
    ]
    return first_cluster, second_cluster


def get_accuracy(
    predicted_positive: Iterable[str],
    predicted_negative: Iterable[str],
    actual_positive: Iterable[str],
    actual_negative: Iterable[str],
) -> Tuple[str, int, int, float]:
    correct = (
        len(set(predicted_positive) & set(actual_positive)) +
        len(set(predicted_negative) & set(actual_negative))
    )
    total = len(actual_positive) + len(actual_negative)
    accuracy = correct / total
    return correct, total, accuracy


def plot_pca_2d(
    train_pcs, train_adjectives, train_true_labels, 
    test_pcs, test_adjectives, test_true_labels,
    verb_pcs, all_verbs, verb_true_labels,
    pca_centroids, label
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_pcs[:, 0],
            y=train_pcs[:, 1],
            text=train_adjectives,
            mode="markers",
            marker=dict(
                color=train_true_labels,
                colorscale="RdBu",
                opacity=0.8,
            ),
            name="PCA in-sample",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_pcs[:, 0],
            y=test_pcs[:, 1],
            text=test_adjectives,
            mode="markers",
            marker=dict(
                color=test_true_labels,
                colorscale="RdBu",
                opacity=0.8,
                symbol="square",
            ),
            name="PCA out-of-sample",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=verb_pcs[:, 0],
            y=verb_pcs[:, 1],
            text=all_verbs,
            mode="markers",
            marker=dict(
                color=verb_true_labels,
                colorscale="RdBu",
                opacity=0.8,
                symbol="star",
            ),
            name="PCA verbs",
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
            f"PCA on {label} of single-token adjectives "
            f"({model.name})"
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    return fig
#%%
def clean_label(label: str) -> str:
    label = label.replace('.npy', '')
    label = label.replace('.html', '')
    label = label.replace('data/', '')
    assert "/" not in label, "Label must not contain slashes"
    return label


def get_model_name(model: Union[HookedTransformer, str]) -> str:
    if isinstance(model, HookedTransformer):
        assert len(model.name) > 0, "Model must have a name"
        model = model.name
    model = model.replace('EleutherAI/', '')
    return model


def update_csv(
    data: pd.DataFrame,
    label: str, 
    model: Union[HookedTransformer, str], 
    key_cols: Iterable[str] = ('train_set', 'train_layer', 'test_set', 'test_layer'),
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.csv')
    curr = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    curr = pd.concat([curr, data], axis=0)
    curr = curr.drop_duplicates(subset=key_cols)
    curr.to_csv(path, index=False)
    return path


def get_csv(
    label: str,
    model: Union[HookedTransformer, str],
    key_cols: Iterable[str] = ('train_set', 'train_layer', 'test_set', 'test_layer'),
) -> pd.DataFrame:
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    path = os.path.join(model_path, label + '.csv')
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=key_cols)
    return df


def eval_csv(
    query: str,
    label: str,
    model: Union[HookedTransformer, str],
):
    df = get_csv(label, model)
    return df.eval(query).any()
#%%
# ============================================================================ #
# K-means
#%%
def train_kmeans(
    train_data: KMeansDataset, train_layer: int,
    test_data: KMeansDataset, test_layer: int,
    n_init: int = 10,
    n_clusters: int = 2,
):
    train_embeddings = train_data.embed(train_layer)
    test_embeddings = test_data.embed(test_layer)
    train_positive_adjectives = [
        adj.strip() for adj, label in zip(train_data.str_labels, train_data.binary_labels) if label == 1
    ]
    train_negative_adjectives = [
        adj.strip() for adj, label in zip(train_data.str_labels, train_data.binary_labels) if label == 0
    ]
    test_positive_adjectives = [
        adj.strip() for adj, label in zip(test_data.str_labels, test_data.binary_labels) if label == 1
    ]
    test_negative_adjectives = [
        adj.strip() for adj, label in zip(test_data.str_labels, test_data.binary_labels) if label == 0
    ]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(train_embeddings)
    train_km_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    test_km_labels = kmeans.predict(test_embeddings)
    km_centroids: Float[np.ndarray, "cluster d_model"] = kmeans.cluster_centers_

    km_first_cluster, km_second_cluster = split_by_label(
        train_data.str_labels, train_km_labels
    )
    pos_first = (
        len(set(km_first_cluster) & set(train_positive_adjectives)) >
        len(set(km_second_cluster) & set(train_positive_adjectives))
    )
    if pos_first:
        train_positive_cluster = km_first_cluster
        train_negative_cluster = km_second_cluster
        km_positive_centroid = km_centroids[0, :]
        km_negative_centroid = km_centroids[1, :]
        test_positive_cluster, test_negative_cluster = split_by_label(
            test_data.str_labels, test_km_labels
        )
    else:
        train_positive_cluster = km_second_cluster
        train_negative_cluster = km_first_cluster
        km_positive_centroid = km_centroids[1, :]
        km_negative_centroid = km_centroids[0, :]
        test_negative_cluster, test_positive_cluster = split_by_label(
            test_data.str_labels, test_km_labels
        )
        train_km_labels = 1 - train_km_labels
        test_km_labels = 1 - test_km_labels
    km_line: Float[np.ndarray, "d_model"] = (
        km_positive_centroid - km_negative_centroid
    )
    # write k means line to file
    save_array(km_line, f"km_{train_data.name}_layer{train_layer}", model)
    # compute cosine sim
    test_kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    test_kmeans.fit(test_embeddings)
    test_line: Float[np.ndarray, "d_model"] = (
        test_kmeans.cluster_centers_[0, :] - test_kmeans.cluster_centers_[1, :]
    )
    cosine_sim = np.dot(km_line, test_line) / (
        np.linalg.norm(km_line) * np.linalg.norm(test_line)
    )
    # get accuracy
    correct, total, accuracy = get_accuracy(
        test_positive_cluster,
        test_negative_cluster,
        test_positive_adjectives,
        test_negative_adjectives,
    )
    columns = [
        'train_set', 'train_layer', 'test_set', 'test_layer', 
        'correct', 'total', 'accuracy', 'similarity',
    ]
    data = [[
        train_data.name, train_layer, test_data.name, test_layer,
        correct, total, accuracy, cosine_sim,
    ]]
    accuracy_df = pd.DataFrame(data, columns=columns)
    update_csv(
        accuracy_df, "km_stats", model
    )

# %%
# ============================================================================ #
# PCA
def train_pca(
    train_data: KMeansDataset, train_layer: int,
    test_data: KMeansDataset, test_layer: int,
    return_figure: bool = False,
    n_init: int = 10,
    n_clusters: int = 2,
) -> Tuple[pd.DataFrame, go.Figure]:
    train_embeddings = train_data.embed(train_layer)
    test_embeddings = test_data.embed(test_layer)
    train_positive_adjectives = [
        adj.strip() for adj, label in zip(train_data.str_labels, train_data.binary_labels) if label == 1
    ]
    train_negative_adjectives = [
        adj.strip() for adj, label in zip(train_data.str_labels, train_data.binary_labels) if label == 0
    ]
    test_positive_adjectives = [
        adj.strip() for adj, label in zip(test_data.str_labels, test_data.binary_labels) if label == 1
    ]
    test_negative_adjectives = [
        adj.strip() for adj, label in zip(test_data.str_labels, test_data.binary_labels) if label == 0
    ]
    pca = PCA(n_components=2)
    train_pcs = pca.fit_transform(train_embeddings.numpy())
    test_pcs = pca.transform(test_embeddings.numpy())
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(train_pcs)
    train_pca_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    test_pca_labels = kmeans.predict(test_pcs)
    pca_centroids: Float[np.ndarray, "cluster pca"] = kmeans.cluster_centers_
    # PCA components should already be normalised, but just in case
    comp_unit = pca.components_[0, :] / np.linalg.norm(pca.components_[0, :])
    save_array(comp_unit, f"pca_0_{train_data.name}_layer{train_layer}", model)
    pca_first_cluster, pca_second_cluster = split_by_label(
        train_data.str_labels, train_pca_labels
    )
    pca_pos_first = (
        len(set(pca_first_cluster) & set(train_positive_adjectives)) >
        len(set(pca_second_cluster) & set(train_positive_adjectives))
    )
    if pca_pos_first:
        # positive first
        train_pca_positive_cluster = pca_first_cluster
        train_pca_negative_cluster = pca_second_cluster
        test_pca_positive_cluster, test_pca_negative_cluster = split_by_label(
            test_data.str_labels, test_pca_labels
        )
        pca_positive_centroid = pca_centroids[0, :]
        pca_negative_centroid = pca_centroids[1, :]
    else:
        # negative first
        train_pca_positive_cluster = pca_second_cluster
        train_pca_negative_cluster = pca_first_cluster
        test_pca_negative_cluster, test_pca_positive_cluster = split_by_label(
            test_data.str_labels, test_pca_labels
        )
        pca_negative_centroid = pca_centroids[0, :]
        pca_positive_centroid = pca_centroids[1, :]
        train_pca_labels = 1 - train_pca_labels
        test_pca_labels = 1 - test_pca_labels
    
    test_pca = PCA(n_components=2)
    test_pca.fit_transform(train_embeddings.numpy())
    test_comp_unit = test_pca.components_[0, :] / np.linalg.norm(test_pca.components_[0, :])
    # compute cosine sim
    cosine_sim = np.dot(comp_unit, test_comp_unit) / (
        np.linalg.norm(comp_unit) * np.linalg.norm(test_comp_unit)
    )
    correct, total, accuracy = get_accuracy(
        test_pca_positive_cluster,
        test_pca_negative_cluster,
        test_positive_adjectives,
        test_negative_adjectives,
    )
    columns = [
        'train_set', 'train_layer', 'test_set', 'test_layer', 
        'correct', 'total', 'accuracy', 'similarity',
    ]
    data = [[
        train_data.name, train_layer, test_data.name, test_layer,
        correct, total, accuracy, cosine_sim,
    ]]
    accuracy_df = pd.DataFrame(data, columns=columns)
    update_csv(
        accuracy_df, "pc_stats", model
    )
    if return_figure and pca.n_components_ == 2:
        fig = plot_pca_2d(
            train_pcs, train_data.str_labels, train_data.binary_labels, 
            test_pcs, test_data.str_labels, test_data.binary_labels,
            pca_centroids
        )
    else:
        fig = None
    return fig
#%%
PROMPT_TYPES = [
    "simple_train",
    "simple_test",
    "simple_adverb",
    "simple_french",
    "proper_nouns",
    "medical",
]
LAYERS = list(range(model.cfg.n_layers + 1))
ITERATOR = tqdm(
    itertools.product(PROMPT_TYPES, LAYERS, PROMPT_TYPES, LAYERS),
    total=len(PROMPT_TYPES) ** 2 * len(LAYERS) ** 2,
)
for train_type, train_layer, test_type, test_layer in ITERATOR:
    if train_layer != test_layer:
        continue
    query = (
        "(train_set == @train_type) & "
        "(test_set == @test_type) & "
        "(train_layer == @train_layer) & "
        "(test_layer == @test_layer)"
    )
    if eval_csv(query, "km_stats", model):
        continue
    print(f"Training {train_type} layer {train_layer} on {test_type} layer {test_layer}")
    trainset = get_km_dataset(model, device, prompt_type=train_type)
    testset = get_km_dataset(model, device, prompt_type=test_type)
    train_kmeans(
        trainset, train_layer,
        testset, test_layer,
    )
    train_pca(
        trainset, train_layer,
        testset, test_layer,
    )
#%%
km_stats = get_csv("km_stats", model)
km_stats
#%%
def hide_nan(val):
    return '' if pd.isna(val) else f"{val:.1%}"
#%%
km_stats.pivot(
    index=["train_set", "train_layer"],
    columns=["test_set"],
    values="accuracy",
).style.background_gradient(cmap="Reds").format(hide_nan).set_caption("K-means accuracy")
#%%
km_stats.pivot(
    index=["train_set", "train_layer"],
    columns=["test_set"],
    values="similarity",
).style.background_gradient(cmap="Reds").format(hide_nan).set_caption("K-means cosine similarities")
# %%
