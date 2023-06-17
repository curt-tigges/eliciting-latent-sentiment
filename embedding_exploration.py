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
train_adjectives = [
    ' ' + adj for adj in train_positive_adjectives + train_negative_adjectives
]
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
test_adjectives = [
    ' ' + adj for adj in test_positive_adjectives + test_negative_adjectives
]
check_overlap(train_positive_adjectives, test_positive_adjectives)
check_overlap(train_negative_adjectives, test_negative_adjectives)
check_single_tokens(test_positive_adjectives)
check_single_tokens(test_negative_adjectives)


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
    embeddings: Float[Tensor, "batch pos d_model"] = transformer.embed(tokens)
    embeddings: Float[Tensor, "batch d_model"] = embeddings.squeeze(1)
    return embeddings.cpu().detach()

train_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    train_adjectives
)
embeddings_normalised: Float[Tensor, "batch d_model"] = F.normalize(
    train_embeddings, dim=-1
)
# %%
# ============================================================================ #
# Cosine similarity
# compute cosine similarity between all pairs of embeddings
cosine_similarities = einops.einsum(
    embeddings_normalised, embeddings_normalised, "b m, c m->b c"
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
principal_components = pca.fit_transform(train_embeddings.numpy())
kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(principal_components)
cluster_labels = kmeans.labels_
centroids: Float[np.ndarray, "cluster pca"] = kmeans.cluster_centers_
#%%
first_cluster = [
    adj[1:]
    for i, adj in enumerate(train_adjectives) 
    if cluster_labels[i] == 0
]
second_cluster = [
    adj[1:]
    for i, adj in enumerate(train_adjectives) 
    if cluster_labels[i] == 1
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
    positive_cluster = first_cluster
    negative_cluster = second_cluster
    positive_centroid = centroids[0, :]
    negative_centroid = centroids[1, :]
#%% # compute euclidean distance between centroids and each point
positive_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    principal_components - positive_centroid, axis=-1
)
negative_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    principal_components - negative_centroid, axis=-1
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
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        mode="markers",
        marker=dict(
            color=cluster_labels,
            colorscale="RdBu",
            opacity=0.8,
        ),
        name="PCA",
    )
)
fig.add_trace(
    go.Scatter(
        x=principal_components[np.argsort(positive_distances)[:5], 0],
        y=principal_components[np.argsort(positive_distances)[:5], 1],
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
        x=principal_components[np.argsort(negative_distances)[:5], 0],
        y=principal_components[np.argsort(negative_distances)[:5], 1],
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
