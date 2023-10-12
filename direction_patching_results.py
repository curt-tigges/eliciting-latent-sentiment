#%%
import glob
import itertools
import os
import einops
import re
from fancy_einsum import einsum
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils
from typing import Dict, Iterable, Literal, Tuple, Union, List, Optional, Callable
from functools import partial
from IPython.display import display, HTML
from tqdm.notebook import tqdm
from path_patching import act_patch, Node, IterNode
from utils.prompts import CleanCorruptedCacheResults, get_dataset, PromptType, ReviewScaffold
from utils.circuit_analysis import create_cache_for_dir_patching, logit_diff_denoising, prob_diff_denoising, logit_flip_denoising, PatchingMetric
from utils.store import save_array, load_array, save_html, save_pdf, to_csv, get_model_name, extract_layer_from_string, zero_pad_layer_string, DIRECTION_PATTERN, is_file, get_csv, get_csv_path, flatten_multiindex
from utils.residual_stream import get_resid_name
#%%
torch.set_grad_enabled(False)
pio.renderers.default = "notebook"
#%%
def get_cached_csv(
    metric_label: str,
    use_heads_label: str,
    scaffold: ReviewScaffold,
    model: Union[str, HookedTransformer],
    proj: Optional[Literal["ortho", "para"]] = None,
    all_but_one: bool = False,
):
    path = f"direction_patching_{metric_label}_{use_heads_label}_{scaffold.value}"
    if proj is not None:
        path += f"_{proj}"
    if all_but_one:
        path += "_all_but_one"
    return get_csv(
        path, 
        model, 
        index_col=0, 
        header=[0, 1],
    )
#%%
def concat_metric_data(
    models: List[str], metric_labels: List[str], use_heads_label: str,
    scaffold: ReviewScaffold = ReviewScaffold.CLASSIFICATION,
):
    metric_data = []
    for model in models:
        for metric_label in metric_labels:
            results = get_cached_csv(metric_label, use_heads_label, scaffold, model)
            if results.empty:
                continue
            s_df = results[~results.index.str.contains("treebank")].copy()
            if s_df.empty:
                continue
            matches = s_df.index.str.extract(DIRECTION_PATTERN)
            multiindex = pd.MultiIndex.from_arrays(
                matches.values.T, 
                names=['method', 'dataset', 'position', 'layer'],
            )
            s_df.index = multiindex
            s_df = s_df.reset_index()
            s_df.dataset = s_df.dataset.fillna("")
            s_df.position = s_df.position.fillna("")
            s_df = (
                s_df
                .groupby(['method', 'dataset', 'position'])
                .max()
                .drop('layer', axis=1, level=0)
            )
            s_df = flatten_multiindex(s_df)
            s_df = s_df[["simple_test_ALL", "treebank_test_ALL"]]
            s_df.columns = s_df.columns.str.replace("test_", "").str.replace("_ALL", "")
            s_df.columns = s_df.columns + f"_{metric_label}"
            s_df.index = s_df.index.str.replace("_simple_train_ADJ", "")
            s_df.index = s_df.index.str.replace("_direction__", "")
            metric_data.append(s_df)
    metric_df = pd.concat(metric_data, axis=1)
    s_style = (
        metric_df
        .style
        .background_gradient(cmap="Reds")
        .format("{:.1f}%")
        .set_caption(f"Direction patching ({metric_labels[0]}, {use_heads_label}) in {models[0]}")
    )
    save_html(
        s_style, 
        f"direction_patching_{use_heads_label}_simple", 
        models[0],
        font_size=40,
        )
    display(s_style)
#%%
def concat_cross_data(
    models: List[str], metric_labels: List[str], use_heads_label: str,
    scaffold: ReviewScaffold = ReviewScaffold.CLASSIFICATION,
    proj: Optional[Literal["ortho", "para"]] = None,
    all_but_one: bool = False,
):
    metric_data = []
    for model in models:
        for metric_label in metric_labels:
            results = get_cached_csv(
                metric_label, use_heads_label, scaffold, model, proj=proj,
                all_but_one=all_but_one,
            )
            if results.empty:
                continue
            s_df = results[~results.index.str.contains("treebank")].copy()
            if s_df.empty:
                continue
            matches = (
                s_df.index
                .str.replace(f"_{proj}.npy", ".npy")
                .str.extract(DIRECTION_PATTERN)
            )
            multiindex = pd.MultiIndex.from_arrays(
                matches.values.T, 
                names=['method', 'dataset', 'position', 'layer'],
            )
            s_df.index = multiindex
            s_df = s_df.reset_index()
            s_df.dataset = s_df.dataset.fillna("")
            s_df.position = s_df.position.fillna("")
            s_df = (
                s_df
                .groupby(['method', 'dataset', 'position'])
                .max()
                .drop('layer', axis=1, level=0)
            )
            s_df.columns = s_df.columns.get_level_values(0)
            # s_df = flatten_multiindex(s_df)
            # s_df.columns = s_df.columns + f"_{metric_label}"
            # s_df.index = s_df.index.str.replace("_direction__", "")
            metric_data.append(s_df)
    metric_df = pd.concat(metric_data, axis=1).reset_index()
    metric_df = metric_df.melt(
        id_vars=["method", "dataset", "position"],
        var_name="test_set",
        value_name="metric",
    ).rename(columns={'dataset': 'train_set'})
    for col in ['train_set', 'test_set']:
        metric_df[col] = metric_df[col].str.replace('simple_train', 'simple_movie')
        metric_df[col] = metric_df[col].str.replace('simple_', '')
    methods = metric_df['method'].unique()
    train_sets = metric_df['train_set'].unique()
    test_sets = metric_df['test_set'].unique()
    result_array = np.zeros((len(methods), len(train_sets), len(test_sets)))
    for i, method in enumerate(methods):
        for j, train_set in enumerate(train_sets):
            for k, test_set in enumerate(test_sets):
                mask = (
                    (metric_df['method'] == method) &
                    (metric_df['train_set'] == train_set) &
                    (metric_df['test_set'] == test_set)
                )
                value = metric_df[mask]['metric'].values
                if value.size > 0:
                    result_array[i, j, k] = value[0]

    # print(result_array.shape)
    # print(len(methods), len(train_sets), len(test_sets))
    fig = px.imshow(
        result_array,
        facet_col=0,
        y=train_sets,
        x=test_sets,
        labels={
            "value": f"{metric_labels[0]} (%)",
            "x": "Test set",
            "y": "Train set",
            "facet_col": "Method",
        },
        zmin=0,
        zmax=100,
        text_auto='.0f',
    )
    for i, label in enumerate(methods):
        fig.layout.annotations[i]['text'] = label
    title = f"Direction patching ({metric_labels[0]}, {use_heads_label}) in {models[0]}"
    if proj is not None:
        title += f" ({proj} projection)"
    if all_but_one:
        title += " (all but one)"
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
        ),
    )
    fig.show()
    save_html(
        fig, 
        f"direction_patching_cross_dataset_{proj}", 
        models[0],
        font_size=40,
    )
#%%
concat_cross_data(
    ["gpt2-small"],
    ["logit_diff"],
    "resid",
    ReviewScaffold.CONTINUATION,
    proj="ortho",
    all_but_one=True,
)
#%%
concat_cross_data(
    ["gpt2-small"],
    ["logit_diff"],
    "resid",
    ReviewScaffold.CONTINUATION,
    proj="para"
)
#%%
concat_cross_data(
    ["gpt2-small"],
    ["logit_diff"],
    "resid",
    ReviewScaffold.CONTINUATION,
)
#%%
concat_metric_data(
    ["pythia-1.4b"],
    ["logit_diff", "logit_flip"],
    "resid",
)
#%%
def concat_layer_data(
    models: Iterable[str], metric_label: str, use_heads_label: str,
    scaffold: ReviewScaffold = ReviewScaffold.CONTINUATION
):
    layer_data = []
    for model in models:
        results = get_cached_csv(metric_label, use_heads_label, scaffold, model)
        if results.empty:
            print(f"No results for {model}")
            continue
        p_df = results[~results.index.str.contains("treebank")].copy()
        matches = p_df.index.str.extract(DIRECTION_PATTERN)
        print(matches)
        multiindex = pd.MultiIndex.from_arrays(
            matches.values.T, names=['method', 'dataset', 'position', 'layer']
        )
        p_df.index = multiindex
        p_df = p_df[("treebank_test", "ALL")]
        p_df = p_df.reset_index()
        p_df.columns = p_df.columns.get_level_values(0)
        p_df.layer = p_df.layer.astype(int)
        p_df['model'] = model
        layer_data.append(p_df)
    layer_df = pd.concat(layer_data)
    layer_df = layer_df.loc[layer_df.method.isin([
        "das", "kmeans", "logistic_regression"
    ])]
    fig = px.line(
        x="layer", 
        y="treebank_test", 
        color="method", 
        facet_col="model", 
        data_frame=layer_df,
        labels={
            "treebank_test": f"{metric_label} (%)",
        }
    )
    fig.update_layout(
        title=dict(
            text=f"Out-of-distribution direction patching performance by method and layer",
            x=0.5,
        ),
        width=1600,
        height=400,
        font=dict(  # global font settings
            size=24  # global font size
        ),
    )
    for axis in fig.layout:
        if "xaxis" in axis:
            fig.layout[axis].matches = None
    models_label = models[0].split("-")[0]
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_{models_label}_facet_plot", model)
    save_html(fig, f"direction_patching_{metric_label}_{use_heads_label}_{models_label}_facet_plot", model)
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_{models_label}_facet_plot", model)
    fig.show()
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_{models_label}_facet_plot", model)
#%%
concat_layer_data(
    [
        "gpt2-small", 
        # "gpt2-medium",
        "pythia-160m", "pythia-410m", 
        "pythia-1.4b",
        #  "pythia-2.8b"
    ], 
    "logit_diff", 
    "resid"
)
#%%
