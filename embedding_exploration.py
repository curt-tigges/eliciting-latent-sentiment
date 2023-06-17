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
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Iterable, List, Tuple, Union, Optional
from jaxtyping import Float, Int
from torch import Tensor
from functools import partial
import copy
import os
import itertools
from IPython.display import display, HTML
import circuitsvis as cv
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
import wandb
# ============================================================================ #
# model loading

#%%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
#%%
# ============================================================================ #
# Data loading

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
        except AssertionError:
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

#%%
# ============================================================================ #
# Training set

train_positive_adjectives = [
    'perfect',
    'fantastic',
    'delightful',
    'cheerful',
    'marvelous',
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
    'joyful',
    'positive',
    'excellent',
    'pleasant'
]
train_negative_adjectives = [
    'dreadful',
    'bad',
    'lousy',
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
    'horrendous',
    'annoying',
    'boring',
    'offensive',
    'frustrating',
    'wretched',
    'inadequate',
    'dire',
    'unpleasant',
    'horrible',
    'mediocre',
    'disappointing',
    'awful'
]
train_adjectives = prepend_space(
    train_positive_adjectives + train_negative_adjectives
)
check_single_tokens(train_adjectives)
#%%
# ============================================================================ #
# Testing set
test_positive_adjectives = [
    'breathtaking', 'stunning', 'impressive', 'admirable', 'phenomenal', 
    'radiant', 'sublime', 'glorious', 'magical', 'sensational', 'pleasing'
]

test_negative_adjectives = [
    'foul', 'vile', 'appalling', 'rotten', 'grim', 'dismal'
]
test_adjectives = prepend_space(
    test_positive_adjectives + test_negative_adjectives
)
check_overlap(train_positive_adjectives, test_positive_adjectives)
check_overlap(train_negative_adjectives, test_negative_adjectives)
check_single_tokens(test_positive_adjectives)
check_single_tokens(test_negative_adjectives)
#%% # verb set
positive_verbs = [
    'enjoyed', 'loved', 'liked', 'appreciated', 'admired', 'cherished'
]
negative_verbs = [
    'hated', 'despised', 'disliked', 'despised'
]
all_verbs = prepend_space(positive_verbs + negative_verbs)
check_single_tokens(positive_verbs)
check_single_tokens(negative_verbs)

# %%
# ============================================================================ #
# Embed
def embed_str_tokens(
    str_tokens: List[str],
    transformer: HookedTransformer = model,
) -> Float[Tensor, "batch d_model"]:
    tokens: Int[Tensor, "batch pos"] = transformer.to_tokens(
        str_tokens, prepend_bos=False
    )
    embeddings: Float[Tensor, "batch pos d_model"] = transformer.embed(
        tokens
    )
    embeddings: Float[Tensor, "batch d_model"] = embeddings.squeeze(1)
    assert len(embeddings.shape) == 2, (
        f"Expected embeddings to be 2D, got {embeddings.shape}"
    )
    return embeddings.cpu().detach()
#%% # train, test embeddings
train_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    train_adjectives
)
test_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    test_adjectives
)
verb_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    all_verbs
)
train_embeddings_normalised: Float[Tensor, "batch d_model"] = F.normalize(
    train_embeddings, dim=-1
)
# %%
# ============================================================================ #
# Cosine similarity
# compute cosine similarity between all pairs of embeddings
cosine_similarities = einops.einsum(
    train_embeddings_normalised, train_embeddings_normalised, "b m, c m->b c"
)
# %%
# plot cosine similarity matrix
fig = px.imshow(
    cosine_similarities.numpy(),
    x=train_adjectives,
    y=train_adjectives,
    color_continuous_scale="RdBu",
    title="Cosine similarity between single-token adjectives",
)
fig.show()
# %%
# ============================================================================ #
# PCA and kmeans
pca = PCA(n_components=2)
train_pcs = pca.fit_transform(train_embeddings.numpy())
test_pcs = pca.transform(test_embeddings.numpy())
verb_pcs = pca.transform(verb_embeddings.numpy())
kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(train_pcs)
train_labels: Int[np.ndarray, "batch"] = kmeans.labels_
test_labels = kmeans.predict(test_pcs)
verb_labels = kmeans.predict(verb_pcs)
centroids: Float[np.ndarray, "cluster pca"] = kmeans.cluster_centers_
#%%
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
#%%
first_cluster = [
    adj[1:]
    for i, adj in enumerate(train_adjectives) 
    if train_labels[i] == 0
]
second_cluster = [
    adj[1:]
    for i, adj in enumerate(train_adjectives) 
    if train_labels[i] == 1
]
# %%
pos_first = (
    (set(first_cluster) == set(train_positive_adjectives)) and
    (set(second_cluster) == set(train_negative_adjectives))
)
neg_first = (
    (set(first_cluster) == set(train_negative_adjectives)) and
    (set(second_cluster) == set(train_positive_adjectives))
)
assert pos_first or neg_first
if pos_first:
    train_positive_cluster = first_cluster
    train_negative_cluster = second_cluster
    positive_centroid = centroids[0, :]
    negative_centroid = centroids[1, :]
    test_positive_cluster, test_negative_cluster = split_by_label(
        test_adjectives, test_labels
    )
    verb_positive_cluster, verb_negative_cluster = split_by_label(
        all_verbs, verb_labels
    )
else:
    train_positive_cluster = second_cluster
    train_negative_cluster = first_cluster
    positive_centroid = centroids[1, :]
    negative_centroid = centroids[0, :]
    test_negative_cluster, test_positive_cluster = split_by_label(
        test_adjectives, test_labels
    )
    verb_positive_cluster, verb_negative_cluster = split_by_label(
        all_verbs, verb_labels
    )
#%% # out-of-sample test
assert set(test_positive_cluster) == set(test_positive_adjectives)
assert set(test_negative_cluster) == set(test_negative_adjectives)
#%% # out-of-sample verbs
assert set(verb_positive_cluster) == set(positive_verbs)
assert set(verb_negative_cluster) == set(negative_verbs)
#%% # compute euclidean distance between centroids and each point
positive_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    train_pcs - positive_centroid, axis=-1
)
negative_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    train_pcs - negative_centroid, axis=-1
)
# print the adjectives closest to each centroid
positive_closest = np.array(train_adjectives)[
    np.argsort(positive_distances)[:5]
]
negative_closest = np.array(train_adjectives)[
    np.argsort(negative_distances)[:5]
]
print(f"Positive centroid nearest adjectives: {positive_closest}")
print(f"Negative centroid nearest adjectives: {negative_closest}")
# %%
# plot the PCA
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=train_pcs[:, 0],
        y=train_pcs[:, 1],
        mode="markers",
        marker=dict(
            color=train_labels,
            colorscale="RdBu",
            opacity=0.8,
        ),
        name="PCA in-sample",
    )
)
# fig.add_trace(
#     go.Scatter(
#         x=test_pcs[:, 0],
#         y=test_pcs[:, 1],
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
        x=verb_pcs[:, 0],
        y=verb_pcs[:, 1],
        mode="markers",
        marker=dict(
            color=verb_labels,
            colorscale="oryel",
            opacity=0.8,
        ),
        name="PCA verbs",
    )
)
fig.add_trace(
    go.Scatter(
        x=train_pcs[np.argsort(positive_distances)[:5], 0],
        y=train_pcs[np.argsort(positive_distances)[:5], 1],
        mode="markers",
        marker=dict(
            color="pink",
            opacity=0.8,
        ),
        name="Positive centroid nearest adjectives",
    )
)
fig.add_trace(
    go.Scatter(
        x=train_pcs[np.argsort(negative_distances)[:5], 0],
        y=train_pcs[np.argsort(negative_distances)[:5], 1],
        mode="markers",
        marker=dict(
            color="violet",
            opacity=0.8,
        ),
        name="Negative centroid nearest adjectives",
    )
)
fig.add_trace(
    go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="markers",
        marker=dict(color='green', symbol='x', size=10),
        name="Centroids",
    )
)
fig.update_layout(
    title="PCA of single-token adjectives",
    xaxis_title="PC1",
    yaxis_title="PC2",
)
fig.show()
# %%
