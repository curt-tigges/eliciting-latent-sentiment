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
#%%
def get_special_positions(format_string: str, token_list: List[str]) -> Dict[str, List[int]]:
    # Loop through each token in the token_list
    format_idx = 0
    curr_sub_token = None
    out = dict()
    # print('get_special_positions', format_string, token_list)
    for token_index, token in enumerate(token_list):
        # Check if the token appears in the format_string
        # print(token_index, token, format_idx, curr_sub_token, out)
        if format_string[format_idx] == '{':
            curr_sub_token = format_string[format_idx + 1:format_string.find('}', format_idx)]
        if format_string.find(token, format_idx) >= 0:
            format_idx = format_string.find(token, format_idx) + len(token)
        elif curr_sub_token is not None:
            out[curr_sub_token] = out.get(curr_sub_token, []) + [token_index]
    
    return out

# Test
format_string = "|endoftext|> When I hear the name{NOUN}, I feel very"
token_list = ['0:<|endoftext|>', '1:When', '2: I', '3: hear', '4: the', '5: name', '6: Mandela', '7:,', '8: I', '9: feel', '10: very']
print(get_special_positions(format_string, token_list)) 
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
CSV_COLS = (
    'train_set', 'train_pos', 'train_layer', 'test_set', 'test_pos', 'test_layer'
)
#%%
class KMeansDataset:

    def __init__(
        self, 
        prompt_strings: List[str], 
        prompt_tokens: Int[Tensor, "batch pos"], 
        binary_labels: Int[Tensor, "batch"],
        position_type: str,
        model: HookedTransformer,
        prompt_type: str,
    ) -> None:
        assert len(prompt_strings) == len(prompt_tokens)
        assert len(prompt_strings) == len(binary_labels)
        self.prompt_strings = prompt_strings
        self.prompt_tokens = prompt_tokens
        self.binary_labels = binary_labels
        self.model = model
        self.prompt_type = prompt_type
        example_str_tokens = model.to_str_tokens(prompt_strings[0])
        self.example = [f"{i}:{tok}" for i, tok in enumerate(example_str_tokens)]
        self.special_position_dict = get_special_positions(prompt_type.get_format_string(), example_str_tokens)
        label_positions = [pos for _, positions in self.special_position_dict.items() for pos in positions]
        assert position_type in self.special_position_dict.keys(), (
            f"Position type {position_type} not found in {self.special_position_dict.keys()} "
            f"for prompt type {prompt_type}"
        )
        self.embed_position = self.special_position_dict[position_type][-1]
        self.position_type = position_type
        self.str_labels = [
            ''.join([model.to_str_tokens(prompt)[pos] for pos in label_positions]) 
            for prompt in prompt_strings
        ]

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
    position_type: str = 'ADJ',
) -> KMeansDataset:
    all_prompts, answer_tokens, clean_tokens, _ = get_dataset(model, device, prompt_type=prompt_type)
    clean_labels = answer_tokens[:, 0, 0] == answer_tokens[0, 0, 0]
    
    assert len(all_prompts) == len(answer_tokens)
    assert len(all_prompts) == len(clean_tokens)
    return KMeansDataset(
        all_prompts,
        clean_tokens,
        clean_labels,
        position_type,
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

#%%
# ============================================================================ #
# K-means
def _fit_kmeans(
    train_data: KMeansDataset, train_layer: int,
    test_data: KMeansDataset, test_layer: int,
    n_init: int = 10,
    n_clusters: int = 2,
    random_state: int = 0,
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
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(train_embeddings)
    train_km_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    test_km_labels = kmeans.predict(test_embeddings)
    km_centroids: Float[np.ndarray, "cluster d_model"] = kmeans.cluster_centers_

    km_first_cluster, km_second_cluster = split_by_label(
        train_data.str_labels, train_km_labels
    )
    one_pos = len(set(km_first_cluster) & set(train_positive_adjectives))
    one_neg = len(set(km_first_cluster) & set(train_negative_adjectives))
    two_pos = len(set(km_second_cluster) & set(train_positive_adjectives))
    two_neg = len(set(km_second_cluster) & set(train_negative_adjectives))
    pos_first = one_pos + two_neg > one_neg + two_pos
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
    # get accuracy
    _, _, insample_accuracy = get_accuracy(
        train_positive_cluster,
        train_negative_cluster,
        train_positive_adjectives,
        train_negative_adjectives,
    )
    correct, total, accuracy = get_accuracy(
        test_positive_cluster,
        test_negative_cluster,
        test_positive_adjectives,
        test_negative_adjectives,
    )
    is_insample = (
        train_data.prompt_type == test_data.prompt_type and
        train_data.position_type == test_data.position_type and
        train_layer == test_layer
    )
    if is_insample:
        assert accuracy >= 0.5, (
            f"Accuracy should be at least 50%, got {accuracy:.1%}, "
            f"direct calc:{insample_accuracy:.1%}, "
            f"train:{train_data.prompt_type.value}, layer:{train_layer}, \n"
            f"positive cluster: {sorted(test_positive_cluster)}\n"
            f"negative cluster: {sorted(test_negative_cluster)}\n"
            f"positive adjectives: {sorted(test_positive_adjectives)}\n"
            f"negative adjectives: {sorted(test_negative_adjectives)}\n"
        )
    
    return km_line, correct, total, accuracy
#%%
def safe_cosine_sim(
    line1: Float[np.ndarray, "d_model"], line2: Float[np.ndarray, "d_model"],
    tol: float = 1e-6, min_value: float = -0.9,
):
    if np.linalg.norm(line1) < tol or np.linalg.norm(line2) < tol:
        cosine_sim = 0
    else:
        cosine_sim = np.dot(line1, line2) / (
            np.linalg.norm(line1) * np.linalg.norm(line2)
        )
    assert cosine_sim >= min_value, (
        f"cosine sim very negative ({cosine_sim:.2f}), looks like a flipped sign"
    )
    return cosine_sim
#%%
def train_kmeans(
    train_data: KMeansDataset, train_layer: int,
    test_data: KMeansDataset, test_layer: int,
    n_init: int = 10,
    n_clusters: int = 2,
    random_state: int = 0,
):
    km_line, correct, total, accuracy = _fit_kmeans(
        train_data, train_layer,
        test_data, test_layer,
        n_init=n_init,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    test_line, _, _, _ = _fit_kmeans(
        test_data, test_layer,
        test_data, test_layer,
        n_init=n_init,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    # write k means line to file
    save_array(km_line, f"km_{train_data.prompt_type.value}_layer{train_layer}", model)

    cosine_sim = safe_cosine_sim(km_line, test_line)
    columns = [
        'train_set', 'train_layer', 'train_pos',
        'test_set', 'test_layer',  'test_pos',
        'correct', 'total', 'accuracy', 'similarity',
    ]
    data = [[
        train_data.prompt_type.value, train_layer, train_data.position_type,
        test_data.prompt_type.value, test_layer, test_data.position_type,
        correct, total, accuracy, cosine_sim,
    ]]
    stats_df = pd.DataFrame(data, columns=columns)
    
    update_csv(
        stats_df, "km_stats", model, key_cols=CSV_COLS
    )

# %%
# ============================================================================ #
# PCA
def train_pca(
    train_data: KMeansDataset, train_layer: int,
    test_data: KMeansDataset, test_layer: int,
    n_init: int = 10,
    n_clusters: int = 2,
    random_state: int = 0,
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
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(train_pcs)
    train_pca_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    test_pca_labels = kmeans.predict(test_pcs)
    pca_centroids: Float[np.ndarray, "cluster pca"] = kmeans.cluster_centers_
    # PCA components should already be normalised, but just in case
    comp_unit = pca.components_[0, :] / np.linalg.norm(pca.components_[0, :])
    save_array(comp_unit, f"pca_0_{train_data.prompt_type.value}_layer{train_layer}", model)
    pca_first_cluster, pca_second_cluster = split_by_label(
        train_data.str_labels, train_pca_labels
    )
    one_pos = len(set(pca_first_cluster) & set(train_positive_adjectives))
    one_neg = len(set(pca_first_cluster) & set(train_negative_adjectives))
    two_pos = len(set(pca_second_cluster) & set(train_positive_adjectives))
    two_neg = len(set(pca_second_cluster) & set(train_negative_adjectives))
    pca_pos_first = one_pos + two_neg > one_neg + two_pos
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
    if np.linalg.norm(test_comp_unit) < 1e-6:
        cosine_sim = 0
    else:
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
        'train_set', 'train_layer', 'train_pos',
        'test_set', 'test_layer',  'test_pos',
        'correct', 'total', 'accuracy', 'similarity',
    ]
    data = [[
        train_data.prompt_type.value, train_layer, train_data.position_type,
        test_data.prompt_type.value, test_layer, test_data.position_type,
        correct, total, accuracy, cosine_sim,
    ]]
    stats_df = pd.DataFrame(data, columns=columns)
    update_csv(
        stats_df, "pca_stats", model, key_cols=CSV_COLS
    )
    plot_data = [[
         train_data.prompt_type.value, train_layer, train_data.position_type,
        test_data.prompt_type.value, test_layer, test_data.position_type,
        train_pcs, train_data.str_labels, train_data.binary_labels, 
        test_pcs, test_data.str_labels, test_data.binary_labels,
        pca_centroids
    ]]
    plot_columns = [
        'train_set', 'train_layer', 'train_pos',
        'test_set', 'test_layer',  'test_pos',
        'train_pcs', 'train_str_labels', 'train_true_labels',
        'test_pcs', 'test_str_labels', 'test_true_labels',
        'pca_centroids',
    ]
    plot_df = pd.DataFrame(plot_data, columns=plot_columns)
    update_csv(
        plot_df, "pca_plot", model, key_cols=CSV_COLS
    )
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

        trainset = get_km_dataset(model, device, prompt_type=train_type, position_type=train_pos)
        testset = get_km_dataset(model, device, prompt_type=test_type, position_type=test_pos)
        train_kmeans(
            trainset, train_layer,
            testset, test_layer,
        )
        train_pca(
            trainset, train_layer,
            testset, test_layer,
        )
#%%
km_stats = get_csv("km_stats", model, key_cols=CSV_COLS)
km_stats = km_stats.loc[
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
