#%%
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
import transformer_lens.utils as utils
import transformer_lens.evals as evals
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import wandb
from utils.store import save_array, save_html
from utils.prompts import get_dataset
# ============================================================================ #
# model loading

#%%
device = torch.device('cpu')
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
#%%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
all_prompts[0]
#%%
def check_for_duplicates(
    str_tokens: List[str],
) -> None:
    if len(str_tokens) != len(set(str_tokens)):
        raise AssertionError('Duplicate tokens found: ' + str_tokens)

def check_single_tokens(
    str_tokens: List[str],
    transformer: HookedTransformer = model,
) -> None:
    good_tokens = []
    error = False
    for token in str_tokens:
        if token.startswith(' '):
            check_token = token
        else:
            check_token = ' ' + token
        try:
            transformer.to_single_token(check_token)
            good_tokens.append(token)
        except AssertionError as e:
            error = True
            continue
    if error:
        raise AssertionError(
            'Non-single tokens found, reduced list: ' + str(good_tokens)
        )
    
def check_overlap(trainset: Iterable[str], testset: Iterable[str]) -> None:
    trainset = set(trainset)
    testset = set(testset)
    overlap = trainset.intersection(testset)
    if len(overlap) > 0:
        raise AssertionError(
            'Overlap between train and test sets.\n'
            f'Reduced test set: {testset - overlap}'
        )
    
def prepend_space(tokens: Iterable[str]) -> List[str]:
    return [' ' + token for token in tokens]


def strip_space(tokens: Iterable[str]) -> List[str]:
    return [token.strip() for token in tokens]
#%%
def get_adjectives_and_verbs():
    train_positive_adjectives = [
        'perfect',
        'fantastic',
        'delightful',
        'cheerful',
        'good',
        'remarkable',
        'satisfactory',
        'wonderful',
        'nice',
        'fabulous',
        'outstanding',
        'satisfying',
        'awesome',
        'exceptional',
        'adequate',
        'incredible',
        'extraordinary',
        'amazing',
        'decent',
        'lovely',
        'brilliant',
        'charming',
        'terrific',
        'superb',
        'spectacular',
        'great',
        'splendid',
        'beautiful',
        'positive',
        'excellent',
        'pleasant'
    ]
    train_negative_adjectives = [
        'dreadful',
        'bad',
        'dull',
        'depressing',
        'miserable',
        'tragic',
        'nasty',
        'inferior',
        'horrific',
        'terrible',
        'ugly',
        'disgusting',
        'disastrous',
        'annoying',
        'boring',
        'offensive',
        'frustrating',
        'wretched',
        'inadequate',
        'dire',
        'unpleasant',
        'horrible',
        'disappointing',
        'awful'
    ]
    train_adjectives = prepend_space(
        train_positive_adjectives + train_negative_adjectives
    )
    train_true_labels = (
        [1] * len(train_positive_adjectives) + 
        [0] * len(train_negative_adjectives)
    )
    check_for_duplicates(train_adjectives)
    check_single_tokens(train_adjectives)
    test_positive_adjectives = [
        'stunning', 'impressive', 'admirable', 'phenomenal', 
        'radiant', 
        'glorious', 'magical', 
        'pleasing', 'movie'
    ]

    test_negative_adjectives = [
        'foul', 
        'vile', 'appalling', 
        'rotten', 'grim', 
        'dismal'
    ]
    test_adjectives = prepend_space(
        test_positive_adjectives + test_negative_adjectives
    )
    test_true_labels = (
        [1] * len(test_positive_adjectives) +
        [0] * len(test_negative_adjectives)
    )
    check_overlap(train_positive_adjectives, test_positive_adjectives)
    check_overlap(train_negative_adjectives, test_negative_adjectives)
    check_single_tokens(test_positive_adjectives)
    check_single_tokens(test_negative_adjectives)
    check_for_duplicates(test_adjectives)
    positive_verbs = [
        'enjoyed', 'loved', 'liked', 'appreciated', 'admired', 
    ]
    negative_verbs = [
        'hated', 'despised', 'disliked',
    ]
    all_verbs = prepend_space(positive_verbs + negative_verbs)
    verb_true_labels = (
        [1] * len(positive_verbs) +
        [0] * len(negative_verbs)
    )
    check_for_duplicates(all_verbs)
    check_single_tokens(positive_verbs)
    check_single_tokens(negative_verbs)
    return (
        train_adjectives, train_true_labels,
        test_adjectives, test_true_labels,
        all_verbs, verb_true_labels
    )

# %%
# ============================================================================ #
# Embed
#%%
class EmbedType(Enum):
    EMBED = 'embed_only'
    UNEMBED = 'unembed_transpose'
    MLP = 'embed_and_mlp0'
    ATTN = 'embed_and_attn0'
    RESID = 'resid_post'
    CONTEXT = 'context'
#%%
def embed_and_mlp0(
    tokens: Int[Tensor, "batch 1"],
    transformer: HookedTransformer = model
) -> Float[Tensor, "batch 1 d_model"]:
    block0 = transformer.blocks[0]
    resid_mid = transformer.embed(tokens)
    mlp_out = block0.mlp((resid_mid))
    resid_post = resid_mid + mlp_out
    return block0.ln2(resid_post)
#%%
def embed_and_attn0(
    tokens: Int[Tensor, "batch 2"],
    transformer: HookedTransformer = model
) -> Float[Tensor, "batch 1 d_model"]:
    assert tokens.shape[1] == 2
    _, cache = transformer.run_with_cache(
        tokens, return_type=None, names_filter = lambda name: 'attn_out' in name
    )
    out: Float[Tensor, "batch pos d_model"] = cache['attn_out', 0]
    return out[:, 1, :]
#%%
def resid_layer(
    tokens: Int[Tensor, "batch 2"],
    layer: int,
    transformer: HookedTransformer = model
) -> Float[Tensor, "batch 1 d_model"]:
    assert tokens.shape[1] == 2
    _, cache = transformer.run_with_cache(
        tokens, return_type=None, names_filter = lambda name: 'resid_post' in name
    )
    out: Float[Tensor, "batch pos d_model"] = cache['resid_post', layer]
    return out[:, 1, :]
#%%
def embed_str_tokens(
    str_tokens: List[str],
    embed_type: EmbedType,
    layer: int = None,
    transformer: HookedTransformer = model,
) -> Float[Tensor, "batch d_model"]:
    tokens: Int[Tensor, "batch 2"] = transformer.to_tokens(
        str_tokens, prepend_bos=True
    )
    non_bos_tokens: Int[Tensor, "batch 1"] = tokens[:, 1:]
    embeddings: Float[Tensor, "batch 1 d_model"]
    if embed_type == EmbedType.EMBED:
        embeddings = transformer.embed(non_bos_tokens)
    elif embed_type == EmbedType.UNEMBED:
        # one-hot encode tokens
        oh_tokens: Int[Tensor, "batch 1 vocab"] = F.one_hot(
            non_bos_tokens, num_classes=transformer.cfg.d_vocab
        ).to(torch.float32)
        wU: Float[Tensor, "model vocab"] = transformer.W_U
        embeddings = oh_tokens @ wU.T
    elif embed_type == EmbedType.MLP:
        embeddings = embed_and_mlp0(non_bos_tokens)
    elif embed_type == EmbedType.ATTN:
        embeddings = embed_and_attn0(tokens)
    elif embed_type == EmbedType.RESID:
        embeddings = resid_layer(tokens, layer)
    else:
        raise ValueError(f'Unrecognised embed type: {embed_type}')
    embeddings: Float[Tensor, "batch d_model"] = embeddings.squeeze(1)
    assert len(embeddings.shape) == 2, (
        f"Expected embeddings to be 2D, got {embeddings.shape}"
    )
    return embeddings.detach()
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
    label: str,
) -> Tuple[str, int, int, float]:
    print('get_accuracy')
    print(predicted_positive, actual_positive)
    print(predicted_negative, actual_negative)
    correct = (
        len(set(predicted_positive) & set(actual_positive)) +
        len(set(predicted_negative) & set(actual_negative))
    )
    total = len(actual_positive) + len(actual_negative)
    accuracy = correct / total
    return label, correct, total, accuracy


def plot_pca_1d(train_pcs, train_adjectives, train_pca_labels, pca_centroids):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_pcs[:, 0],
            y=np.zeros_like(train_pcs[:, 0]),
            text=train_adjectives,
            mode="markers",
            marker=dict(
                color=train_pca_labels,
                colorscale="RdBu",
                opacity=0.8,
            ),
            name="PCA in-sample",
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=test_pcs[:, 0],
    #         y=np.zeros_like(test_pcs[:, 0]),
    #         mode="markers",
    #         marker=dict(
    #             color=test_labels,
    #             colorscale="oryel",
    #             opacity=0.8,
    #         ),
    #         name="PCA out-of-sample",
    #     )
    # )
    fig.add_trace(
        go.Scatter(
            x=pca_centroids[:, 0],
            y=np.zeros_like(pca_centroids[:, 0]),
            mode="markers",
            marker=dict(
                color=["red", "blue"],
                size=10,
                opacity=1,
            ),
            name="PCA centroids",
        )
    )
    fig.update_layout(
        xaxis=dict(
            title="PC1",
            # showgrid=False,
            # zeroline=False,
            # showline=False,
            # showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        showlegend=True,
    )
    return fig


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
    save_html(fig, f"PCA_{embedding_type.value}", model)
    return fig
#%%
def get_embed_label(embedding_type: EmbedType, layer: int = None):
    layer_suffix = f"_layer_{layer}" if embedding_type in (EmbedType.RESID, EmbedType.CONTEXT) else ""
    return f"{embedding_type.value}{layer_suffix}"
#%%
# ============================================================================ #
# K-means
#%%
def train_kmeans(
    train_embeddings, train_adjectives, train_labels,
    test_embeddings, test_adjectives, test_labels,
    verb_embeddings, all_verbs, verb_labels,
    embedding_type, layer
) -> Tuple[KMeans, pd.DataFrame]:
    train_positive_adjectives = [
        adj.strip() for adj, label in zip(train_adjectives, train_labels) if label == 1
    ]
    train_negative_adjectives = [
        adj.strip() for adj, label in zip(train_adjectives, train_labels) if label == 0
    ]
    test_positive_adjectives = [
        adj.strip() for adj, label in zip(test_adjectives, test_labels) if label == 1
    ]
    test_negative_adjectives = [
        adj.strip() for adj, label in zip(test_adjectives, test_labels) if label == 0
    ]
    positive_verbs = [
        verb.strip() for verb, label in zip(all_verbs, verb_labels) if label == 1
    ]
    negative_verbs = [
        verb.strip() for verb, label in zip(all_verbs, verb_labels) if label == 0
    ]
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(train_embeddings)
    train_km_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    test_km_labels = kmeans.predict(test_embeddings)
    verb_km_labels = kmeans.predict(verb_embeddings)
    km_centroids: Float[np.ndarray, "cluster d_model"] = kmeans.cluster_centers_

    print('train adjectives and labels', train_adjectives, train_km_labels, train_labels)
    km_first_cluster, km_second_cluster = split_by_label(
        train_adjectives, train_km_labels
    )
    print('KM clusters', km_first_cluster, km_second_cluster)
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
            test_adjectives, test_km_labels
        )
        verb_positive_cluster, verb_negative_cluster = split_by_label(
            all_verbs, verb_km_labels
        )
    else:
        train_positive_cluster = km_second_cluster
        train_negative_cluster = km_first_cluster
        km_positive_centroid = km_centroids[1, :]
        km_negative_centroid = km_centroids[0, :]
        test_negative_cluster, test_positive_cluster = split_by_label(
            test_adjectives, test_km_labels
        )
        verb_negative_cluster, verb_positive_cluster = split_by_label(
            all_verbs, verb_km_labels
        )
        train_km_labels = 1 - train_km_labels
        test_km_labels = 1 - test_km_labels
        verb_km_labels = 1 - verb_km_labels
    km_line: Float[np.ndarray, "d_model"] = (
        km_positive_centroid - km_negative_centroid
    )
    # write k means line to file
    embed_label = get_embed_label(embedding_type, layer)
    save_array(km_line, f"km_2c_line_{embed_label}", model)
    # get accuracy
    print('pos/neg clusters', train_positive_cluster, train_positive_adjectives, train_negative_cluster, train_negative_adjectives)
    accuracy_data = []
    accuracy_data.append(get_accuracy(
        train_positive_cluster,
        train_negative_cluster,
        train_positive_adjectives,
        train_negative_adjectives,
        "In-sample KMeans",
    ))
    accuracy_data.append(get_accuracy(
        test_positive_cluster,
        test_negative_cluster,
        test_positive_adjectives,
        test_negative_adjectives,
        "Out-of-sample KMeans",
    ))
    accuracy_data.append(get_accuracy(
        verb_positive_cluster,
        verb_negative_cluster,
        positive_verbs,
        negative_verbs,
        "Out-of-sample KMeans verbs",
    ))
    accuracy_df = pd.DataFrame(accuracy_data, columns=['partition', 'correct', 'total', 'accuracy'])
    accuracy_df['embedding_type'] = embed_label
    return kmeans, accuracy_df

# %%
# ============================================================================ #
# PCA
def train_pca(
    train_embeddings, train_adjectives, train_labels,
    test_embeddings, test_adjectives, test_labels,
    verb_embeddings, all_verbs, verb_labels,
    embedding_type, layer, 
    kmeans: KMeans
) -> Tuple[pd.DataFrame, go.Figure]:
    train_positive_adjectives = [
        adj.strip() for adj, label in zip(train_adjectives, train_labels) if label == 1
    ]
    train_negative_adjectives = [
        adj.strip() for adj, label in zip(train_adjectives, train_labels) if label == 0
    ]
    test_positive_adjectives = [
        adj.strip() for adj, label in zip(test_adjectives, test_labels) if label == 1
    ]
    test_negative_adjectives = [
        adj.strip() for adj, label in zip(test_adjectives, test_labels) if label == 0
    ]
    positive_verbs = [
        verb.strip() for verb, label in zip(all_verbs, verb_labels) if label == 1
    ]
    negative_verbs = [
        verb.strip() for verb, label in zip(all_verbs, verb_labels) if label == 0
    ]
    embed_label = get_embed_label(embedding_type, layer)
    pca = PCA(n_components=2)
    train_pcs = pca.fit_transform(train_embeddings.numpy())
    test_pcs = pca.transform(test_embeddings.numpy())
    verb_pcs = pca.transform(verb_embeddings.numpy())
    kmeans.fit(train_pcs)
    train_pca_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    test_pca_labels = kmeans.predict(test_pcs)
    verb_pca_labels = kmeans.predict(verb_pcs)
    pca_centroids: Float[np.ndarray, "cluster pca"] = kmeans.cluster_centers_
    for comp in range(pca.n_components_):
        # PCA components should already be normalised, but just in case
        comp_unit = pca.components_[comp, :] / np.linalg.norm(pca.components_[comp, :])
        save_array(comp_unit, f"pca_{comp}_{embed_label}", model)
    pca_first_cluster, pca_second_cluster = split_by_label(
        train_adjectives, train_pca_labels
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
            test_adjectives, test_pca_labels
        )
        verb_pca_positive_cluster, verb_pca_negative_cluster = split_by_label(
            all_verbs, verb_pca_labels
        )
        pca_positive_centroid = pca_centroids[0, :]
        pca_negative_centroid = pca_centroids[1, :]
    else:
        # negative first
        train_pca_positive_cluster = pca_second_cluster
        train_pca_negative_cluster = pca_first_cluster
        test_pca_negative_cluster, test_pca_positive_cluster = split_by_label(
            test_adjectives, test_pca_labels
        )
        verb_pca_negative_cluster, verb_pca_positive_cluster = split_by_label(
            all_verbs, verb_pca_labels
        )
        pca_negative_centroid = pca_centroids[0, :]
        pca_positive_centroid = pca_centroids[1, :]
        train_pca_labels = 1 - train_pca_labels
        test_pca_labels = 1 - test_pca_labels
        verb_pca_labels = 1 - verb_pca_labels
    accuracy_data = []
    accuracy_data.append(get_accuracy(
        train_pca_positive_cluster,
        train_pca_negative_cluster,
        train_positive_adjectives,
        train_negative_adjectives,
        "In-sample PCA",
    ))
    accuracy_data.append(get_accuracy(
        test_pca_positive_cluster,
        test_pca_negative_cluster,
        test_positive_adjectives,
        test_negative_adjectives,
        "Out-of-sample PCA",
    ))
    accuracy_data.append(get_accuracy(
        verb_pca_positive_cluster,
        verb_pca_negative_cluster,
        positive_verbs,
        negative_verbs,
        "Out-of-sample PCA verbs",
    ))
    accuracy_df = pd.DataFrame(accuracy_data, columns=['partition', 'correct', 'total', 'accuracy'])
    accuracy_df['embedding_type'] = embed_label

    if pca.n_components_== 1:
        fig = plot_pca_1d()
    elif pca.n_components_ == 2:
        fig = plot_pca_2d(
            train_pcs, train_adjectives, train_labels, 
            test_pcs, test_adjectives, test_labels,
            verb_pcs, all_verbs, verb_labels,
            pca_centroids, embed_label
        )
    else:
        fig = None
    return accuracy_df, fig
#%%
def run_for_embedding(embedding_type: EmbedType, layer: int = None) -> Tuple[pd.DataFrame, go.Figure]:
    (
        train_adjectives, train_labels, 
        test_adjectives, test_labels,
        all_verbs, verb_labels
    ) = get_adjectives_and_verbs()
    train_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
        train_adjectives, embedding_type, layer=layer
    )
    test_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
        test_adjectives, embedding_type, layer=layer
    )
    verb_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
        all_verbs, embedding_type, layer=layer
    )
    train_embeddings.shape, test_embeddings.shape, verb_embeddings.shape
    kmeans, km_df = train_kmeans(
        train_embeddings, train_adjectives, train_labels,
        test_embeddings, test_adjectives, test_labels,
        verb_embeddings, all_verbs, verb_labels,
        embedding_type, layer
    )
    pca_df, fig = train_pca(
        train_embeddings, train_adjectives, train_labels,
        test_embeddings, test_adjectives, test_labels,
        verb_embeddings, all_verbs, verb_labels,
        embedding_type, layer, kmeans
    )
    return pd.concat([km_df, pca_df], axis=0), fig

#%%
def run_for_activation(embedding_type: EmbedType, layer: int = None) -> Tuple[pd.DataFrame, go.Figure]:
    assert embedding_type == EmbedType.CONTEXT
    train_size = len(clean_tokens) // 2
    _, train_cache = model.run_with_cache(
        clean_tokens[:train_size], return_type=None,
    )
    _, test_cache = model.run_with_cache(
        clean_tokens[train_size:], return_type=None,
    )
    train_embeddings: Float[Tensor, "batch d_model"] = train_cache["resid_post", layer][:, -1, :]
    test_embeddings: Float[Tensor, "batch d_model"] = test_cache["resid_post", layer][:, -1, :]
    verb_embeddings: Float[Tensor, "batch d_model"] = test_cache["resid_post", layer][:, -1, :]
    adjective_position = 6
    train_adjectives = [model.to_str_tokens(tokens)[adjective_position] for tokens in clean_tokens[:train_size]]
    test_adjectives = [model.to_str_tokens(tokens)[adjective_position] for tokens in clean_tokens[train_size:]]
    all_verbs = test_adjectives
    train_labels = [int((token == answer_tokens[0, 0, 0]).item()) for token in answer_tokens[:train_size, 0, 0]]
    test_labels = [int((token == answer_tokens[0, 0, 0]).item()) for token in answer_tokens[train_size:, 0, 0]]
    verb_labels = test_labels
    print(train_adjectives)
    kmeans, km_df = train_kmeans(
        train_embeddings, train_adjectives, train_labels,
        test_embeddings, test_adjectives, test_labels,
        verb_embeddings, all_verbs, verb_labels,
        embedding_type, layer
    )
    pca_df, fig = train_pca(
        train_embeddings, train_adjectives, train_labels,
        test_embeddings, test_adjectives, test_labels,
        verb_embeddings, all_verbs, verb_labels,
        embedding_type, layer, kmeans
    )
    return pd.concat([km_df, pca_df], axis=0), fig
#%% # train, test embeddings
embedding_type = EmbedType.CONTEXT # CHANGEME!!!!!!!!!!!!!
layer_data = []
layer_figs = []
for layer in range(12):
    layer_df, layer_fig = run_for_activation(embedding_type=embedding_type, layer=layer)
    layer_data.append(layer_df)
    layer_figs.append(layer_fig)
accuracy_df = pd.concat(layer_data).reset_index(drop=True)
accuracy_df
#%%
accuracy_df['layer'] = accuracy_df.embedding_type.str.extract(r'layer_(\d+)').astype(int)
accuracy_df.pivot(
    index="layer",
    columns="partition",
    values="accuracy",
).style.background_gradient(cmap="Reds").format("{:.1%}")
#%%
# concatenate the list of figures
fig = make_subplots(rows=len(layer_figs), cols=1, shared_xaxes=False, shared_yaxes=False)

for i, layer_fig in enumerate(layer_figs):
    if layer_fig is not None:
        for trace in layer_fig.data:
            fig.add_trace(trace, row=i+1, col=1)
    fig.update_yaxes(title_text=f"Layer {i}", row=i+1, col=1)
    

# Add legend to the first subplot
fig.update_traces(showlegend=True, row=1, col=1)

# Hide legends in other subplots
for i in range(2, len(layer_figs)+1):
    fig.update_traces(showlegend=False, row=i, col=1)

fig.update_layout(
    title=f"PCA for {embedding_type.value} embeddings in each layer",
    xaxis_title="PC1",
    height=3000,
    width=1000,
)
save_html(fig, f"pca_{embedding_type.value}_by_layer.html", model)
fig.show()
# %%
