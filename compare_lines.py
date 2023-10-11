#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
import re
import os
import pandas as pd
from utils.store import load_array, save_html, extract_layer_from_string, zero_pad_layer_string, flatten_multiindex, DIRECTION_PATTERN
# %%
MODEL_NAME = "gpt2-small"

#%%
def get_directions():
    dir_path = os.path.join('data', MODEL_NAME)
    matching_files = [
        filename 
        for filename in os.listdir(dir_path) 
        if re.match(DIRECTION_PATTERN, filename) 
        and "2d" not in filename and "3d" not in filename and "treebank" not in filename
        and "_ALL" not in filename
    ]
    return sorted(matching_files)
#%%
def clean_labels(labels: pd.Series):
    return (
        labels
        # .str.replace("simple_train_", "")
        .str.replace("simple_train_", "simple_movie_")
        .str.replace("logistic_regression", "LR")
        .str.replace("das", "DAS")
        .str.replace("kmeans", "K_means")
        .str.replace("pca", "PCA")
        .str.replace("treebank_train_ALL", "treebank")
        .str.replace("mean_diff", "Mean_diff")
        .str.replace("random_direction", "Random")
        # .str.replace("_ADJ", "")
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
df = df.loc[
    df.index.str.contains("_layer00"),
    df.columns.str.contains("_layer00")
]

df = df.abs()
extracted = df.index.str.extract(DIRECTION_PATTERN)
multiindex = pd.MultiIndex.from_arrays(
    extracted.values.T, 
    names=['method', 'dataset', 'position', 'layer'],
)
df.index = multiindex
cols_extracted = df.columns.str.extract(DIRECTION_PATTERN)
multiindex = pd.MultiIndex.from_arrays(
    cols_extracted.values.T, 
    names=['method', 'dataset', 'position', 'layer'],
)
df.columns = multiindex
df.columns = df.columns.droplevel('position')
df.columns = df.columns.remove_unused_levels()
df.index = df.index.droplevel('position')
df.index = df.index.remove_unused_levels()
df.columns = df.columns.droplevel('layer')
df.columns = df.columns.remove_unused_levels()
df.index = df.index.droplevel('layer')
df.index = df.index.remove_unused_levels()

df = df.reset_index()
df = df.melt(
    var_name=["test_method", "test_set"],
    value_name="similarity",
    id_vars=["method", "dataset"],
).rename(columns={'dataset': 'train_set'})
df = df.loc[df.method.isin(['das', 'mean_diff', 'logistic_regression'])]
df = df.loc[df.method == df.test_method].drop('test_method', axis=1)
df.train_set = df.train_set.str.replace("simple_train", "simple_movie").str.replace("simple_", "")
df.test_set = df.test_set.str.replace("simple_train", "simple_movie").str.replace("simple_", "")
# df.test_set = clean_labels(df.test_set)
# df.test_set = df.test_set.str.replace("_layer00", "")
# df.index = clean_labels(df.index)
# df.index = df.index.str.replace("_layer00", "")
df
#%%
methods = df['method'].unique()
train_sets = df['train_set'].unique()
test_sets = df['test_set'].unique()
result_array = np.zeros((len(methods), len(train_sets), len(test_sets)))
for i, method in enumerate(methods):
    for j, train_set in enumerate(train_sets):
        for k, test_set in enumerate(test_sets):
            mask = (
                (df['method'] == method) &
                (df['train_set'] == train_set) &
                (df['test_set'] == test_set)
            )
            value = df[mask]['similarity'].values
            if value.size > 0:
                result_array[i, j, k] = value[0]

fig = px.imshow(
        result_array,
        facet_col=0,
        x=train_sets,
        y=test_sets,
        labels={
            "value": "Similarity",
            "x": "Train set",
            "y": "Test set",
            "facet_col": "Method",
        },
        zmin=0,
        zmax=1,
        text_auto='.0%',
    )
for i, label in enumerate(methods):
    fig.layout.annotations[i]['text'] = label
fig.update_layout(
    title=dict(
        text=f"Cosine similarity across datasets in {MODEL_NAME}",
        x=0.5,
    ),
)
fig.show()
save_html(
    fig, 
    "cosine_similarity_cross_dataset", 
    MODEL_NAME,
    font_size=40,
)
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
