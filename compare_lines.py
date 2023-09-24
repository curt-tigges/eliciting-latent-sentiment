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
    r'^(kmeans|pca|das|logistic_regression|mean_diff)_'
    r'(simple_train|treebank_train)_'
    r'(ADJ|ALL)_'
    r'layer(\d*)'
    r'\.npy$'
)
#%%
def get_directions():
    dir_path = os.path.join('data', MODEL_NAME)
    matching_files = [filename for filename in os.listdir(dir_path) if re.match(PATTERN, filename)]
    return sorted(matching_files)
#%%
def parse_filename(filename):
    match = re.search(PATTERN, filename)
    method, data, position, layer = match.groups()
    layer = int(layer)
    return method, data, position, layer
#%%
direction_labels = get_directions()
direction_layers = [extract_layer_from_string(label) for label in direction_labels]
directions = [load_array(filename, MODEL_NAME).squeeze() for filename in direction_labels]
direction_labels = [zero_pad_layer_string(label) for label in direction_labels]
label_tuples = [parse_filename(label) for label in direction_labels]
treebank_layers = [layer for i, layer in enumerate(direction_layers) if label_tuples[i][1] == 'treebank_train']
label_multiindex = pd.MultiIndex.from_tuples(label_tuples, names=['method', 'data', 'position', 'layer'])
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
        .str.replace("_0", "")
        .str.replace("mean_diff", "Mean_diff")
        .str.replace("_ADJ", "")
    )
# %%
df = pd.DataFrame(
    similarities,
    columns=label_multiindex,
    index=label_multiindex,
).sort_index(axis=0).sort_index(axis=1)
df = df.loc[
    df.index.get_level_values(-1).isin([0]),
    df.columns.get_level_values(-1).isin([0]),
]
# df = df.loc[
#     (df.index.get_level_values(1) != 'treebank_train'),
#     (df.columns.get_level_values(1) != 'treebank_train')
# ]
df = df.loc[
    (df.index.get_level_values(1) == 'treebank_train') | (df.index.get_level_values(2) == 'ADJ'),
    (df.columns.get_level_values(1) == 'treebank_train') | (df.columns.get_level_values(2) == 'ADJ')
]
df = df.loc[
    (df.index.get_level_values(1) != 'treebank_train') | (df.index.get_level_values(0) == 'das'),
    (df.columns.get_level_values(1) != 'treebank_train') | (df.columns.get_level_values(0) == 'das')
]
df = flatten_multiindex(df)
df = df.abs()
# df = df.drop(columns=['das_treebank_train_ALL_0'], index=['das_treebank_train_ALL_0'])
df.columns = clean_labels(df.columns)
df.index = clean_labels(df.index)
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
