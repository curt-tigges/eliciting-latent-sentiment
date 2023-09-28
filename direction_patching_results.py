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
from typing import Dict, Iterable, Tuple, Union, List, Optional, Callable
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
    model: HookedTransformer,
):
    return get_csv(
        f"direction_patching_{metric_label}_{use_heads_label}_{scaffold.value}", model, 
        index_col=0, header=[0, 1],
    )
#%%
# def export_results(
#     results: pd.DataFrame, metric_label: str, use_heads_label: str
# ) -> None:
#     all_layers = pd.Series([extract_layer_from_string(label) for label in results.index])
#     das_treebank_layers = all_layers[results.index.str.contains("das_treebank")]
#     if len(das_treebank_layers) > 0:
#         mask = ~results.index.str.contains("das") | all_layers.isin(das_treebank_layers)
#         mask.index = results.index
#         results = results.loc[mask]

#     layers_style = (
#         flatten_multiindex(results)
#         .style
#         .background_gradient(cmap="Reds", axis=None, low=0, high=1)
#         .format("{:.1f}%")
#         .set_caption(f"Direction patching ({metric_label}, {use_heads_label}) in {model}")
#     )
#     save_html(layers_style, f"direction_patching_{metric_label}_{use_heads_label}", model)
#     display(layers_style)

#     missing_data = (
#         not results.columns.get_level_values(0).str.contains("treebank").any() or 
#         not results.columns.get_level_values(0).str.contains("simple").any()
#     )
#     if missing_data:
#         return

#     s_df = results[~results.index.str.contains("treebank")].copy()
#     matches = s_df.index.str.extract(DIRECTION_PATTERN)
#     multiindex = pd.MultiIndex.from_arrays(matches.values.T, names=['method', 'dataset', 'position', 'layer'])
#     s_df.index = multiindex
#     s_df = s_df.reset_index().groupby(['method', 'dataset', 'position']).max().drop('layer', axis=1, level=0)
#     s_df = flatten_multiindex(s_df)
#     s_df = s_df[["simple_test_ALL", "treebank_test_ALL"]]
#     s_df.columns = s_df.columns.str.replace("test_", "").str.replace("_ALL", "")
#     s_df.index = s_df.index.str.replace("_simple_train_ADJ", "")
#     s_style = (
#         s_df
#         .style
#         .background_gradient(cmap="Reds")
#         .format("{:.1f}%")
#         .set_caption(f"Direction patching ({metric_label}, {use_heads_label}) in {model.name}")
#     )
#     to_csv(s_df, f"direction_patching_{metric_label}_simple", model, index=True)
#     save_html(
#         s_style, f"direction_patching_{metric_label}_{use_heads_label}_simple", model,
#         font_size=40,
#         )
#     display(s_style)
    
#     t_df = results[results.index.str.contains("das_treebank") & ~results.index.str.contains("None")].copy()
#     t_df = t_df.loc[:, t_df.columns.get_level_values(0).str.contains("treebank")]
#     matches = t_df.index.str.extract(DIRECTION_PATTERN)
#     multiindex = pd.MultiIndex.from_arrays(matches.values.T, names=['method', 'dataset', 'position', 'layer'])
#     t_df.index = multiindex
#     t_df = t_df.loc[t_df.index.get_level_values(-1).astype(int) < t_df.index.get_level_values(-1).astype(int).max() - 1]
#     t_df.sort_index(level=3)
#     t_df = flatten_multiindex(t_df)
#     t_df.index = t_df.index.str.replace("das_treebank_train_ALL_0", "")
#     t_df.columns = ["logit_diff"]
#     t_df = t_df.T
#     t_style = t_df.style.background_gradient(cmap="Reds").format("{:.1f}%")
#     to_csv(t_df, f"direction_patching_{metric_label}_treebank", model, index=True)
#     save_html(t_style, f"direction_patching_{metric_label}_{use_heads_label}_treebank", model)
#     display(t_style)

#     p_df = results[~results.index.str.contains("treebank")].copy()
#     matches = p_df.index.str.extract(DIRECTION_PATTERN)
#     multiindex = pd.MultiIndex.from_arrays(
#         matches.values.T, names=['method', 'dataset', 'position', 'layer']
#     )
#     p_df.index = multiindex
#     p_df = p_df[("treebank_test", "ALL")]
#     p_df = p_df.reset_index()
#     p_df.columns = p_df.columns.get_level_values(0)
#     p_df.layer = p_df.layer.astype(int)
#     fig = px.line(x="layer", y="treebank_test", color="method", data_frame=p_df)
#     fig.update_layout(
#         title="Out-of-distribution directional patching performance by method and layer"
#     )
#     fig.show()
#     p_df = flatten_multiindex(p_df)
#     if use_heads_label == "resid":
#         to_csv(p_df, f"direction_patching_{metric_label}_layers", model, index=True) # FIXME: add {heads_label}
#     save_html(fig, f"direction_patching_{metric_label}_{use_heads_label}_plot", model)
#     save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_plot", model)
#%%
def concat_metric_data(
    models: Iterable[str], metric_labels: List[str], use_heads_label: str,
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
        .set_caption(f"Direction patching ({metric_label}, {use_heads_label}) in {model}")
    )
    save_html(
        s_style, f"direction_patching_{metric_label}_{use_heads_label}_simple", model,
        font_size=40,
        )
    display(s_style)
#%%
concat_metric_data(
    ["pythia-1.4b"],
    ["logit_diff", "logit_flip"],
    "resid",
)
#%%
def concat_layer_data(
    models: Iterable[str], metric_label: str, use_heads_label: str,
    scaffold: ReviewScaffold
):
    layer_data = []
    for model in models:
        results = get_cached_csv(metric_label, use_heads_label, scaffold, model)
        p_df = results[~results.index.str.contains("treebank")].copy()
        matches = p_df.index.str.extract(DIRECTION_PATTERN)
        multiindex = pd.MultiIndex.from_arrays(
            matches.values.T, names=['method', 'dataset', 'position', 'layer']
        )
        p_df.index = multiindex
        p_df = p_df[("treebank_test", "ALL")]
        p_df = p_df.reset_index()
        p_df.columns = p_df.columns.get_level_values(0)
        p_df.layer = p_df.layer.astype(int)
        p_df['model'] = model
        p_df['max_layer'] = p_df.layer.max()
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
        title="Out-of-distribution directional patching performance by method and layer",
        width=1600,
        height=500,
        title_x=0.5,
        font=dict(  # global font settings
            size=16  # global font size
        ),
    )
    for axis in fig.layout:
        if "xaxis" in axis:
            fig.layout[axis].matches = None
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_facet_plot", model)
    save_html(fig, f"direction_patching_{metric_label}_{use_heads_label}_facet_plot", model)
    save_pdf(fig, f"direction_patching_{metric_label}_{use_heads_label}_facet_plot", model)
    fig.show()
#%%
concat_layer_data(
    ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"], 
    "logit_diff", 
    "resid_gpt2"
)
#%%
concat_layer_data(
    ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"], 
    "logit_diff", 
    "resid_pythia"
)
#%%
