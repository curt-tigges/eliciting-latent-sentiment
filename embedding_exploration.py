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
from typing import List, Tuple, Union, Optional
from jaxtyping import Float, Int
from typeguard import typechecked
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
    # center_unembed=True,
    # center_writing_weights=True,
    # fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
#%%
# ============================================================================ #
# Data loading

single_token_adjectives = [
    'perfect',
    'fantastic',
    'delightful',
    'dreadful',
    'bad',
    'lousy',
    'cheerful',
    'dull',
    'marvelous',
    'depressing',
    'good',
    'remarkable',
    'satisfactory',
    'miserable',
    'tragic',
    'wonderful',
    'nice',
    'fabulous',
    'outstanding',
    'nasty',
    'inferior',
    'horrific',
    'terrible',
    'satisfying',
    'ugly',
    'disgusting',
    'awesome',
    'disastrous',
    'horrendous',
    'annoying',
    'exceptional',
    'boring',
    'adequate',
    'offensive',
    'incredible',
    'extraordinary',
    'amazing',
    'frustrating',
    'wretched',
    'inadequate',
    'decent',
    'lovely',
    'brilliant',
    'charming',
    'dire',
    'terrific',
    'unpleasant',
    'superb',
    'spectacular',
    'great',
    'splendid',
    'horrible',
    'beautiful',
    'mediocre',
    'joyful',
    'positive',
    'disappointing',
    'awful',
    'excellent',
    'pleasant'
]

single_token_adjectives = [' ' + adj for adj in set(single_token_adjectives)]
for adj in single_token_adjectives:
    model.to_single_token(adj)
# %%
# ============================================================================ #
# Embed
tokens: Int[Tensor, "batch pos"] = model.to_tokens(
    single_token_adjectives, prepend_bos=False
)
embeddings: Float[Tensor, "batch pos d_model"] = model.embed(tokens).detach().cpu()
assert tokens.shape[1] == embeddings.shape[1] == 1
embeddings: Float[Tensor, "batch d_model"] = embeddings.squeeze(1)
embeddings_normalised: Float[Tensor, "batch d_model"] = F.normalize(embeddings, dim=-1)
# %%
# compute cosine similarity between all pairs of embeddings
cosine_similarities = torch.einsum("bm,cm->bc", embeddings_normalised, embeddings_normalised)
# %%
# plot cosine similarity matrix
fig = px.imshow(
    cosine_similarities.numpy(),
    x=single_token_adjectives,
    y=single_token_adjectives,
    color_continuous_scale="RdBu",
    title="Cosine similarity between single-token adjectives",
)
fig.show()
# %%
# ============================================================================ #
# # Initialize and fit KMeans

kmeans = KMeans(n_clusters=2, n_init=10)  # 'k' is the number of clusters you want to create
kmeans.fit(embeddings.numpy())
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

first_cluster = [
    adj 
    for i, adj in enumerate(single_token_adjectives) 
    if cluster_labels[i] == 0
]
second_cluster = [
    adj 
    for i, adj in enumerate(single_token_adjectives) 
    if cluster_labels[i] == 1
]

# %%
first_cluster
# %%
second_cluster
# %%
# ============================================================================ #
# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(embeddings.numpy())
kmeans.fit(principal_components)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# %%
# plot the PCA
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        mode="markers",
        marker=dict(
            color=labels,
            colorscale="RdBu",
            opacity=0.8,
        ),
        name="PCA",
    )
)
fig.add_trace(
    go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="markers",
        marker=dict(color='red', symbol='x', size=10),
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
# ============================================================================ #
# To-dos

# FIXME: does this direction in embedding space generalise to other tokens?
# FIXME: automate the checking of kmeans labels
# FIXME: avoid performing kmeans twice
# %%
