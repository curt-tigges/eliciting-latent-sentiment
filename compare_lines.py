#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
import re
import os
import pandas as pd
from utils.store import load_array, save_html, extract_layer_from_string, zero_pad_layer_string, flatten_multiindex
# %%
MODEL_NAME = "gpt2-small"
PATTERN = (
    r'^(kmeans_simple_train_ADJ|pca_simple_train_ADJ|das_simple_train_ADJ|logistic_regression_simple_train_ADJ|mean_diff_simple_train_ADJ|random_direction)_'
    r'layer(\d*)'
    r'\.npy$'
)
#%%
def get_directions():
    dir_path = os.path.join('data', MODEL_NAME)
    matching_files = [filename for filename in os.listdir(dir_path) if re.match(PATTERN, filename)]
    return sorted(matching_files)
#%%
def clean_labels(labels: pd.Series):
    return (
        labels
        .str.replace("simple_train_", "")
        .str.replace("logistic_regression", "LR")
        .str.replace("das", "DAS")
        .str.replace("kmeans", "K_means")
        .str.replace("pca", "PCA")
        .str.replace("treebank_train_ALL", "treebank")
        .str.replace("mean_diff", "Mean_diff")
        .str.replace("random_direction", "Random")
        .str.replace("_ADJ", "")
        .str.replace(".npy", "")
        .str.replace("000", "00")
    )
#%%
direction_labels = get_directions()
direction_layers = [extract_layer_from_string(label) for label in direction_labels]
directions = [load_array(filename, MODEL_NAME).squeeze() for filename in direction_labels]
direction_labels = [zero_pad_layer_string(label) for label in direction_labels]
stacked = np.stack(directions)
stacked = (stacked.T / np.linalg.norm(stacked, axis=1)).T
similarities: Float[np.ndarray, "direction d_model"] = einops.einsum(
    stacked, stacked, 'm d, n d -> m n'
)
#%%
def move_col_to_end(col, df):
    cols = list(df.columns)
    cols.remove(col)
    cols.append(col)
    return df[cols]
#%%
def move_row_to_end(row, df):
    rows = list(df.index)
    rows.remove(row)
    rows.append(row)
    return df.loc[rows]
# %%
df = pd.DataFrame(
    similarities,
    columns=direction_labels,
    index=direction_labels,
).sort_index(axis=0).sort_index(axis=1)
df.columns = clean_labels(df.columns)
df.index = clean_labels(df.index)
df = df.loc[
    df.index.str.contains("_layer00"),
    df.columns.str.contains("_layer00")
]
df.columns = df.columns.str.replace("_layer00", "")
df.index = df.index.str.replace("_layer00", "")
df = df.abs()
df
#%%
# df = df.drop(columns=['das_treebank_train_ALL_0'], index=['das_treebank_train_ALL_0'])
styled = (
    df.style
    .background_gradient(cmap='Reds', vmin=0, vmax=1)
    .format("{:.1%}")
)
save_html(
    styled, 'direction_similarities', MODEL_NAME, static=True
)
styled
# %%
