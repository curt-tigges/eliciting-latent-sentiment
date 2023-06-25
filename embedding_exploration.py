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
    dup_tokens = []
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
            dup_tokens.append(token)
            continue
    if error:
        raise AssertionError(
            'Non-single tokens found: ' + str(dup_tokens)
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
    'wonderful',
    'nice',
    'fabulous',
    'outstanding',
    'satisfying',
    'awesome',
    'exceptional',
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
train_neutral_adjectives = [
    'okay',
    'fine',
    'average',
    'ok',
    'neutral',
    'fair',
    'adequate',
    'alright',
    'OK',
    'reasonable',
    'sufficient',
    'acceptable',
    'satisfactory',
    'balanced',
    'valid',
    'appropriate',
]

train_adjectives = prepend_space(
    train_positive_adjectives + 
    train_neutral_adjectives + 
    train_negative_adjectives
)
train_true_labels = (
    [1] * len(train_positive_adjectives) + 
    [0] * len(train_neutral_adjectives) +
    [-1] * len(train_negative_adjectives)
)
check_for_duplicates(train_adjectives)
check_single_tokens(train_adjectives)
#%%
# ============================================================================ #
# Testing set
test_positive_adjectives = [
    'breathtaking', 'stunning', 'impressive', 'admirable', 'phenomenal', 
    'radiant', 'sublime', 'glorious', 'magical', 'sensational', 'pleasing',
]

test_negative_adjectives = [
    'appalling', 'grim', 'hideous', 'dismal', 'rotten', 
    'vile',  'gruesome', 'foul',
]
test_neutral_adjectives = [
    'decent', 'common', 'standard', 'routine', 'functional',
    'typical', 'ordinary', 'regular', 'usual',
]
test_adjectives = prepend_space(
    test_positive_adjectives + 
    test_neutral_adjectives + 
    test_negative_adjectives
)
test_true_labels = (
    [1] * len(test_positive_adjectives) +
    [0] * len(test_neutral_adjectives) +
    [-1] * len(test_negative_adjectives)
)
check_overlap(train_positive_adjectives, test_positive_adjectives)
check_overlap(train_negative_adjectives, test_negative_adjectives)
check_overlap(train_neutral_adjectives, test_neutral_adjectives)
check_single_tokens(test_positive_adjectives)
check_single_tokens(test_negative_adjectives)
check_single_tokens(test_neutral_adjectives)
check_for_duplicates(test_adjectives)
#%% # verb set
positive_verbs = [
    'enjoyed', 'loved', 'liked', 'appreciated', 'admired', 'cherished'
]
negative_verbs = [
    'hated', 'despised', 'disliked',
]
neutral_verbs = [
    'watched', 'saw', 'viewed', 'witnessed', 'experienced', 'observed', 'noticed',
]
all_verbs = prepend_space(positive_verbs + negative_verbs)
verb_true_labels = (
    [1] * len(positive_verbs) +
    [0] * len(neutral_verbs) + 
    [0] * len(negative_verbs)
)
check_for_duplicates(all_verbs)
check_single_tokens(positive_verbs)
check_single_tokens(neutral_verbs)
check_single_tokens(negative_verbs)

# %%
# ============================================================================ #
# Embed
#%%
class EmbedType(Enum):
    EMBED = 'embed_only'
    UNEMBED = 'unembed_transpose'
    MLP = 'embed_and_mlp0'
#%%
_, train_cache = model.run_with_cache(train_adjectives, return_type=None)
#%%
def embed_and_mlp0(
    tokens: Int[Tensor, "batch 1"],
    transformer: HookedTransformer = model
):
    block0 = transformer.blocks[0]
    resid_mid = transformer.embed(tokens)
    mlp_out = block0.mlp((resid_mid))
    resid_post = resid_mid + mlp_out
    return block0.ln2(resid_post)
#%%
def embed_str_tokens(
    str_tokens: List[str],
    embed_type: EmbedType,
    transformer: HookedTransformer = model,
) -> Float[Tensor, "batch d_model"]:
    tokens: Int[Tensor, "batch 1"] = transformer.to_tokens(
        str_tokens, prepend_bos=False
    )
    embeddings: Float[Tensor, "batch 1 d_model"]
    if embed_type == EmbedType.EMBED:
        embeddings = transformer.embed(tokens)
    elif embed_type == EmbedType.UNEMBED:
        # one-hot encode tokens
        oh_tokens: Int[Tensor, "batch 1 vocab"] = F.one_hot(
            tokens, num_classes=transformer.cfg.d_vocab
        ).to(torch.float32)
        wU: Float[Tensor, "model vocab"] = transformer.W_U
        embeddings = oh_tokens @ wU.T
    elif embed_type == EmbedType.MLP:
        embeddings = embed_and_mlp0(tokens)
    else:
        raise ValueError(f'Unrecognised embed type: {embed_type}')
    embeddings: Float[Tensor, "batch d_model"] = embeddings.squeeze(1)
    assert len(embeddings.shape) == 2, (
        f"Expected embeddings to be 2D, got {embeddings.shape}"
    )
    return embeddings.cpu().detach()
#%% # train, test embeddings
embedding_type = EmbedType.MLP
train_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    train_adjectives, embedding_type
)
test_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    test_adjectives, embedding_type
)
verb_embeddings: Float[Tensor, "batch d_model"] = embed_str_tokens(
    all_verbs, embedding_type
)
train_embeddings_normalised: Float[Tensor, "batch d_model"] = F.normalize(
    train_embeddings, dim=-1
)
train_embeddings_centred: Float[Tensor, "batch d_model"] = ((
    train_embeddings - 
    train_embeddings.mean(dim=0)
).T / train_embeddings.norm(dim=-1)).T
#%%
# ============================================================================ #
# K-means
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(train_embeddings)
train_km_labels: Int[np.ndarray, "batch"] = kmeans.labels_
test_km_labels = kmeans.predict(test_embeddings)
verb_km_labels = kmeans.predict(verb_embeddings)
km_centroids: Float[np.ndarray, "cluster d_model"] = kmeans.cluster_centers_

#%%
def split_by_label(
    adjectives: List[str], labels: Int[np.ndarray, "batch"]
) -> Tuple[List[str], List[str]]:
    return [
        [adj[1:] for i, adj in enumerate(adjectives) if labels[i] == label]
        for label in [0, 1, 2]
    ]
#%%
train_km_clusters = split_by_label(
    train_adjectives, train_km_labels
)
test_km_clusters = split_by_label(
    test_adjectives, test_km_labels
)
verb_km_clusters = split_by_label(
    all_verbs, verb_km_labels
)
#%%
def get_cluster_and_centroid(
    train_relevant_adjectives: List[str],
    train_clusters: List[List[str]],
    train_centroids: Float[np.ndarray, "cluster d"],
):
    if (
        len(set(train_clusters[0]) & set(train_relevant_adjectives)) >
        len(train_clusters[0]) * 0.5
    ):
        label = 0
    elif (
        len(set(train_clusters[1]) & set(train_relevant_adjectives)) >
        len(train_clusters[1]) * 0.5
    ):
        label = 1
    else:
        label = 2
    return (
        train_clusters[label], train_centroids[label, :], label
    )
#%%
train_positive_cluster, km_positive_centroid, km_positive_label = get_cluster_and_centroid(
    train_positive_adjectives, train_km_clusters, km_centroids
)
train_negative_cluster, km_negative_centroid, km_negative_label = get_cluster_and_centroid(
    train_negative_adjectives, train_km_clusters, km_centroids
)
train_neutral_cluster, km_neutral_centroid, km_neutral_label = get_cluster_and_centroid(
    train_neutral_adjectives, train_km_clusters, km_centroids
)
km_label_dict = {
    km_positive_label: 1,
    km_negative_label: -1,
    km_neutral_label: 0,
}
train_km_labels = [
    km_label_dict[label] for label in train_km_labels
]
test_km_labels = [
    km_label_dict[label] for label in test_km_labels
]
verb_km_labels = [
    km_label_dict[label] for label in verb_km_labels
]
test_km_positive_cluster = test_km_clusters[km_positive_label]
test_km_negative_cluster = test_km_clusters[km_negative_label]
test_km_neutral_cluster = test_km_clusters[km_neutral_label]
verb_km_positive_cluster = verb_km_clusters[km_positive_label]
verb_km_negative_cluster = verb_km_clusters[km_negative_label]
verb_km_neutral_cluster = verb_km_clusters[km_neutral_label]


km_pos_neg: Float[np.ndarray, "d_model"] = (
    km_positive_centroid - km_negative_centroid
)
km_pos_neg_normalised: Float[
    Tensor, "d_model"
] = torch.tensor(km_pos_neg / np.linalg.norm(km_pos_neg), dtype=torch.float32)
#%%
print(np.linalg.norm(km_positive_centroid))
print(np.linalg.norm(km_negative_centroid))
print(np.linalg.norm(train_embeddings[0, :]))
#%% # write k means line to file
with open(f"data/km_positive_{embedding_type.value}.npy", "wb") as f:
    np.save(f, km_positive_centroid)
with open(f"data/km_negative_{embedding_type.value}.npy", "wb") as f:
    np.save(f, km_negative_centroid)
with open(f"data/km_neutral_{embedding_type.value}.npy", "wb") as f:
    np.save(f, km_neutral_centroid)
with open(f"data/km_positive_negative_{embedding_type.value}.npy", "wb") as f:
    np.save(f, km_pos_neg)
#%%
# project adjectives onto k-means line
train_km_projected = einops.einsum(
    train_embeddings, km_pos_neg_normalised, "b d, d->b"
).numpy()
# sort adjectives by projection with k-means line
train_km_projected_sorted = sorted(
    zip(train_km_projected, train_adjectives), key=lambda x: x[0]
)
print(
    "Most negative: ", 
    train_km_projected_sorted[:10], 
    "\nMost positive: ",
    train_km_projected_sorted[-10:]
)

# %%
# ============================================================================ #
# PCA
pca = PCA(n_components=2)
train_pcs = pca.fit_transform(train_embeddings.numpy())
test_pcs = pca.transform(test_embeddings.numpy())
verb_pcs = pca.transform(verb_embeddings.numpy())
kmeans.fit(train_pcs)
train_pca_labels: Int[np.ndarray, "batch"] = kmeans.labels_
test_pca_labels = kmeans.predict(test_pcs)
verb_pca_labels = kmeans.predict(verb_pcs)
pca_centroids: Float[np.ndarray, "cluster pca"] = kmeans.cluster_centers_
#%%
train_pca_clusters = split_by_label(
    train_adjectives, train_pca_labels
)
test_pca_clusters = split_by_label(
    test_adjectives, test_pca_labels
)
verb_pca_clusters = split_by_label(
    all_verbs, verb_pca_labels
)
#%%
train_pca_positive_cluster, pca_positive_centroid, pca_positive_label = get_cluster_and_centroid(
    train_positive_adjectives, train_pca_clusters, pca_centroids
)
train_pca_negative_cluster, pca_negative_centroid, pca_negative_label = get_cluster_and_centroid(
    train_negative_adjectives, train_pca_clusters, pca_centroids
)
train_pca_neutral_cluster, pca_neutral_centroid, pca_neutral_label = get_cluster_and_centroid(
    train_neutral_adjectives, train_pca_clusters, pca_centroids
)
pca_label_dict = {
    pca_positive_label: 1,
    pca_negative_label: -1,
    pca_neutral_label: 0,
}
train_pca_labels = [
    pca_label_dict[label] for label in train_pca_labels
]
test_pca_labels = [
    pca_label_dict[label] for label in test_pca_labels
]
verb_pca_labels = [
    pca_label_dict[label] for label in verb_pca_labels
]
test_pca_positive_cluster = test_pca_clusters[pca_positive_label]
test_pca_negative_cluster = test_pca_clusters[pca_negative_label]
test_pca_neutral_cluster = test_pca_clusters[pca_neutral_label]
verb_pca_positive_cluster = verb_pca_clusters[pca_positive_label]
verb_pca_negative_cluster = verb_pca_clusters[pca_negative_label]
verb_pca_neutral_cluster = verb_pca_clusters[pca_neutral_label]

pca_pos_neg: Float[np.ndarray, "d_model"] = (
    pca_positive_centroid - pca_negative_centroid
)
pca_pos_neg_normalised: Float[
    Tensor, "d_model"
] = torch.tensor(pca_pos_neg / np.linalg.norm(pca_pos_neg), dtype=torch.float32)
pca_pos_neg: Float[np.ndarray, "2"] = (
    pca_positive_centroid - pca_negative_centroid
)
pca_pos_neg_normalised: Float[
    Tensor, "2"
] = torch.tensor(pca_pos_neg / np.linalg.norm(pca_pos_neg), dtype=torch.float32)
#%%
# project adjectives onto PCA line
train_pca_projected = einops.einsum(
    train_pcs, pca_pos_neg_normalised, "b d, d->b"
)
# sort adjectives by projection with k-means line
train_pca_projected_sorted = sorted(
    zip(train_pca_projected, train_adjectives), key=lambda x: x[0]
)
train_pca_projected_sorted[:10]

# %%
# ============================================================================ #
# Classification accuracy

def print_accuracy(
    predicted_positive: Iterable[str],
    predicted_neutral: Iterable[str],
    predicted_negative: Iterable[str],
    actual_positive: Iterable[str],
    actual_neutral: Iterable[str],
    actual_negative: Iterable[str],
    label: str,
) -> None:
    correct = (
        len(set(predicted_positive) & set(actual_positive)) +
        len(set(predicted_neutral) & set(actual_neutral)) +
        len(set(predicted_negative) & set(actual_negative))
    )
    total = len(actual_positive) + len(actual_neutral) + len(actual_negative)
    accuracy = correct / total
    print(f"{label} accuracy: {correct}/{total} = {accuracy:.0%}")

# ============================================================================ #
# KMeans accuracy

print_accuracy(
    train_positive_cluster,
    train_neutral_cluster,
    train_negative_cluster,
    train_positive_adjectives,
    train_neutral_adjectives,
    train_negative_adjectives,
    "In-sample KMeans",
)
print_accuracy(
    test_km_positive_cluster,
    test_km_neutral_cluster,
    test_km_negative_cluster,
    test_positive_adjectives,
    test_neutral_adjectives,
    test_negative_adjectives,
    "Out-of-sample KMeans",
)
print_accuracy(
    verb_km_positive_cluster,
    verb_km_neutral_cluster,
    verb_km_negative_cluster,
    positive_verbs,
    neutral_verbs,
    negative_verbs,
    "Out-of-sample KMeans verbs",
)
# ============================================================================ #
# PCA accuracy

print_accuracy(
    train_pca_positive_cluster,
    train_pca_neutral_cluster,
    train_pca_negative_cluster,
    train_positive_adjectives,
    train_neutral_adjectives,
    train_negative_adjectives,
    "In-sample PCA",
)
print_accuracy(
    test_pca_positive_cluster,
    test_pca_neutral_cluster,
    test_pca_negative_cluster,
    test_positive_adjectives,
    test_neutral_adjectives,
    test_negative_adjectives,
    "Out-of-sample PCA",
)
print_accuracy(
    verb_pca_positive_cluster,
    verb_pca_neutral_cluster,
    verb_pca_negative_cluster,
    positive_verbs,
    neutral_verbs,
    negative_verbs,
    "Out-of-sample PCA verbs",
)
#%% # compute euclidean distance between centroids and each point
positive_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    train_pcs - pca_positive_centroid, axis=-1
)
negative_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    train_pcs - pca_negative_centroid, axis=-1
)
neutral_distances: Float[np.ndarray, "batch"] = np.linalg.norm(
    train_pcs - pca_neutral_centroid, axis=-1
)
# print the adjectives closest to each centroid
positive_closest = np.array(train_adjectives)[
    np.argsort(positive_distances)[:5]
]
negative_closest = np.array(train_adjectives)[
    np.argsort(negative_distances)[:10]
]
negative_closest_distances = [
    f"{d:.2f}" for d in np.sort(negative_distances)[:10]
]
neutral_closest = np.array(train_adjectives)[
    np.argsort(neutral_distances)[:10]
]
neutral_closest_distances = [
    f"{d:.2f}" for d in np.sort(neutral_distances)[:10]
]
print(f"Positive centroid nearest adjectives: {positive_closest}")
print(f"Negative centroid nearest adjectives: {negative_closest}")
print(f"Negative centroid distances: {negative_closest_distances}")
print(f"Neutral centroid nearest adjectives: {neutral_closest}")
#%% # 
# ============================================================================ #
# Plotting
#%%
def plot_pca_1d():
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
#%%
if pca.n_components_== 1:
    fig = plot_pca_1d()
    fig.show()
#%%
label_colorscale = [
    [0, 'blue'],       # color for values at the lower end of the scale
    [0.5, 'green'],    # color for values in the middle of the scale
    [1, 'red']         # color for values at the higher end of the scale
]
# %%
# plot the PCA

def plot_pca_2d():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_pcs[:, 0],
            y=train_pcs[:, 1],
            text=train_adjectives,
            mode="markers",
            marker=dict(
                color=train_true_labels,
                colorscale=label_colorscale,
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
                colorscale=label_colorscale,
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
                colorscale=label_colorscale,
                opacity=0.8,
                symbol="star",
            ),
            name="PCA verbs",
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=train_pcs[np.argsort(positive_distances)[:5], 0],
    #         y=train_pcs[np.argsort(positive_distances)[:5], 1],
    #         text=positive_closest,
    #         mode="markers",
    #         marker=dict(
    #             color="red",
    #             opacity=0.8,
    #             symbol="square",
    #         ),
    #         name="Positive centroid nearest adjectives",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=train_pcs[np.argsort(negative_distances)[:5], 0],
    #         y=train_pcs[np.argsort(negative_distances)[:5], 1],
    #         text=negative_closest,
    #         mode="markers",
    #         marker=dict(
    #             color="blue",
    #             opacity=0.8,
    #             symbol="square",
    #         ),
    #         name="Negative centroid nearest adjectives",
    #     )
    # )
    fig.add_trace(
        go.Scatter(
            x=pca_centroids[:, 0],
            y=pca_centroids[:, 1],
            mode="markers",
            marker=dict(color='black', symbol='x', size=10),
            name="Centroids",
        )
    )
    fig.update_layout(
        title=f"PCA on {embedding_type.value} of single-token adjectives",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )
    return fig
# %%
if pca.n_components_ == 2:
    fig = plot_pca_2d()
    fig.show()
#%%
# ============================================================================ #
# Correlation between measures of negativity
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=train_pca_projected,
        y=train_km_projected,
        mode="markers",
        marker=dict(
            color=train_true_labels,
            colorscale=label_colorscale,
            opacity=0.8,
        ),
        name="Kmeans vs PCA scores",
        text=train_adjectives,
    )
)
fig.update_layout(
    title=(
        "Correlation between KM and PCA scores using " + 
        embedding_type.value
    ),
    yaxis_title="Kmeans scores",
    xaxis_title="PCA scores",
)
fig.show()
# %%
# write the results to a csv
df = pd.DataFrame({
    "adjective": train_pca_projected,
    "kmeans": train_km_projected,
    "pca": train_pcs[:, 0],
    "binary_label": train_pca_labels,
})
df.to_csv("data/negativity_scores.csv", index=False)
# %%
# ============================================================================ #
# Histogram of dot product of embeddings and km_line

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=train_embeddings[:len(train_positive_adjectives)].numpy().dot(km_pos_neg),
        marker=dict(
            color="red",
            opacity=0.5,
        ),
        name="positive adj in-sample",
        nbinsx=20,
        showlegend=True,
    )
)
fig.add_trace(
    go.Histogram(
        x=train_embeddings[len(train_positive_adjectives):].numpy().dot(km_pos_neg),
        marker=dict(
            color="blue",
            opacity=0.5,
        ),
        name="negative adj in-sample",
        nbinsx=20,
        showlegend=True,
    )
)
fig.add_trace(
    go.Histogram(
        x=test_embeddings[:len(test_positive_adjectives)].numpy().dot(km_pos_neg),
        marker=dict(
            color="darkred",
            opacity=0.5,
        ),
        name="positive adj out-of-sample",
        nbinsx=20,
        showlegend=True,
    )
)
fig.add_trace(
    go.Histogram(
        x=test_embeddings[len(test_positive_adjectives):].numpy().dot(km_pos_neg),
        marker=dict(
            color="darkblue",
            opacity=0.5,
        ),
        name="negative adj out-of-sample",
        nbinsx=20,
        showlegend=True,
    )
)
fig.add_trace(
    go.Histogram(
        x=test_embeddings[:len(test_positive_adjectives)].numpy().dot(km_pos_neg),
        marker=dict(
            color="pink",
            opacity=0.5,
        ),
        name="positive verb out-of-sample",
        nbinsx=20,
        showlegend=True,
    )
)
fig.add_trace(
    go.Histogram(
        x=test_embeddings[len(test_positive_adjectives):].numpy().dot(km_pos_neg),
        marker=dict(
            color="teal",
            opacity=0.5,
        ),
        name="negative verb out-of-sample",
        nbinsx=20,
        showlegend=True,
    )
)
fig.update_layout(
    title="Histogram of dot product of embeddings and KM line",
    barmode="overlay",
    bargap=0.1,
)
#%%
pile_loader = evals.make_pile_data_loader(model.tokenizer, batch_size=8)
#%%
sample_size = 10_000
dot_data = []
for x in pile_loader:
    batch_embed: Float[Tensor, "batch pos d_model"] = embed_and_mlp0(
        x['tokens']
    ).cpu().detach().numpy()
    batch_dots: Float[np.ndarray, "batch pos"] = einops.einsum(
        batch_embed, 
        km_pos_neg_normalised,
        "batch pos d_model, d_model -> batch pos",
    )
    flattened_dots = batch_dots.flatten()
    list_of_str_tokens = [
        s 
        for i in range(len(x['tokens']))
        for s in model.to_str_tokens(x['tokens'][i]) 
    ]
    dot_df = pd.DataFrame({
        "dot": flattened_dots,
        "token": list_of_str_tokens,
    })
    dot_data.append(dot_df)
    sample_size -= len(flattened_dots)
    if sample_size <= 0:
        break
#%%
dots_df = pd.concat(dot_data)
len(dots_df)
#%%
fig = px.histogram(
    data_frame=dots_df, x='dot', hover_data=['token'], marginal="rug",
    title="Histogram of dot product of embeddings and KM line on pile",
)
fig.write_html("data/pile_embeddings.html")
fig.show()

# %%
