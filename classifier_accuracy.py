#%%
from functools import partial
import itertools
import gc
import numpy as np
import torch
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
import einops
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union, Literal
from transformer_lens import HookedTransformer
from transformer_lens.evals import make_owt_data_loader
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from circuitsvis.activations import text_neuron_activations
from circuitsvis.utils.render import RenderedHTML
from tqdm.notebook import tqdm
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import scipy.stats as stats
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text, to_csv, get_csv, save_pdf, save_pickle, update_csv
from utils.neuroscope import (
    plot_neuroscope, get_dataloader, get_projections_for_text, plot_top_p, plot_topk, 
    harry_potter_start, harry_potter_fr_start, get_batch_pos_mask, extract_text_window,
    extract_activations_window
)
#%%
pd.set_option('display.max_colwidth', 200)
torch.set_grad_enabled(False)
#%%
DIRECTIONS = [
    "kmeans_simple_train_ADJ_layer1",
    "pca_simple_train_ADJ_layer1",
    "mean_diff_simple_train_ADJ_layer1",
    "logistic_regression_simple_train_ADJ_layer1",
    "das_simple_train_ADJ_layer1",
]
#%%
device = "cuda"
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
)
#%%
dataloader = get_dataloader(model, "stas/openwebtext-10k", batch_size=16)
#%%
def get_activations_from_dataloader(
    data: torch.utils.data.dataloader.DataLoader,
    direction: Float[Tensor, "d_model"],
    max_batches: int = None,
) -> Float[Tensor, "row pos"]:
    all_acts = []
    for batch_idx, batch_value in tqdm(enumerate(data), total=len(data)):
        batch_tokens = batch_value['tokens'].to(device)
        batch_acts: Float[Tensor, "batch pos layer"] = get_projections_for_text(
            batch_tokens, direction, model
        )
        all_acts.append(batch_acts)
        if max_batches is not None and batch_idx >= max_batches:
            break
    # Concatenate the activations into a single tensor
    all_acts: Float[Tensor, "row pos layer"] = torch.cat(all_acts, dim=0)
    return all_acts
#%%
class ClearCache:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        model.cuda()

    def __exit__(self, exc_type, exc_val, exc_tb):
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
#%%
def get_activations_cached(
    data: torch.utils.data.dataloader.DataLoader,
    direction_label: str,
):
    path = direction_label + "_activations.npy"
    if is_file(path, model):
        print("Loading activations from file")
        sentiment_activations = load_array(path, model)
        sentiment_activations: Float[Tensor, "row pos layer"]  = torch.tensor(
            sentiment_activations, device=device, dtype=torch.float32
        )
    else:
        print("Computing activations")
        with ClearCache():
            direction = load_array(direction_label + ".npy", model)
            direction = torch.tensor(
                direction, device=device, dtype=torch.float32
            )
            sentiment_activations: Float[Tensor, "row pos layer"]  = get_activations_from_dataloader(
                data, direction
            )
        save_array(sentiment_activations, path, model)
    return sentiment_activations
#%%
def sample_by_bin(
    data: Float[Tensor, "batch pos"],
    bins: int = 20,
    samples_per_bin: int = 20,
    seed: int = 0,
    window_size: int = 10,
    verbose: bool = False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    flat = data.flatten()
    hist, bin_edges = np.histogram(flat.cpu().numpy(), bins=bins)
    bin_indices: Int[np.ndarray, "batch pos"] = np.digitize(data.cpu().numpy(), bin_edges)
    if verbose:
        print(bin_edges)
    indices = []
    for bin_idx in range(1, bins + 1):
        lb = bin_edges[bin_idx - 1]
        ub = bin_edges[bin_idx]
        bin_batches, bin_positions = np.where(bin_indices == bin_idx)
        bin_samples = np.random.randint(0, len(bin_batches), samples_per_bin)
        indices += [
            (
                bin_idx, 
                lb, 
                ub, 
                bin_batches[bin_sample], 
                bin_positions[bin_sample],
                activations[bin_batches[bin_sample], bin_positions[bin_sample], 1].item(),
            )
            for bin_sample in bin_samples
        ]
    df =  pd.DataFrame(
        indices, 
        columns=["bin", "lb", "ub", "batch", "position", "activation"]
    )
    tokens = []
    texts = []
    for _, row in df.iterrows():
        text = extract_text_window(
            int(row.batch), int(row.position), dataloader, model, window_size=window_size
        )
        tokens.append(text[window_size])
        texts.append("".join(text))
    df.reset_index(drop=True, inplace=True)
    df['token'] = tokens
    df['text'] = texts
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
#%% # Save samples
for direction_label in tqdm(DIRECTIONS):
    samples_path = direction_label + "_bin_samples.csv"
    if is_file(samples_path, model):
        continue
    activations = get_activations_cached(dataloader, direction_label)
    bin_samples = sample_by_bin(
        activations[:, :, 1], verbose=False
    )
    to_csv(bin_samples, samples_path, model)
#%%
#%%
def plot_bin_proportions(
    df: pd.DataFrame, 
    label: str,
    nbins=50, 
):
    if df.activation.dtype == pd.StringDtype:
        df.activation = df.activation.map(lambda x: eval(x).item()).astype(float)
    if "das" in label:
        labelled_bin_samples.activation *= -1
    sentiments = sorted(df['sentiment'].unique())
    df = df.sort_values(by='activation').reset_index(drop=True)
    df['activation_cut'] = pd.cut(df.activation, bins=nbins)
    df.activation_cut = df.activation_cut.apply(lambda x: 0.5 * (x.left + x.right))
    
    fig = go.Figure()
    data = []
    
    for x, bin_df in df.groupby('activation_cut'):
        if bin_df.empty:
            continue
        label_props = bin_df.value_counts('sentiment', normalize=True, sort=False)
        data.append([label_props.get(sentiment, 0) for sentiment in sentiments])
    
    data = pd.DataFrame(data, columns=sentiments)
    cumulative_data = data.cumsum(axis=1)  # Cumulative sum along columns
    
    x_values = df['activation_cut'].unique()
    
    # Adding traces for the rest of the sentiments
    for idx, sentiment in enumerate(sentiments):
        fig.add_trace(go.Scatter(
            x=x_values, y=cumulative_data[sentiment], name=sentiment,
            hovertemplate='<br>'.join([
                'Sentiment: ' + sentiment,
                'Activation: %{x}',
                'Cum. Label proportion: %{y:.4f}',
            ]),
            fill='tonexty',
            mode='lines',
        ))
    
    fig.update_layout(
        title=f"Proportion of Sentiment by Activation ({label})",
        title_x=0.5,
        showlegend=True,
        xaxis_title="Activation",
        yaxis_title="Cum. Label proportion",
    )

    return fig
#%%
for direction_label in DIRECTIONS:
    labelled_bin_samples = get_csv(
        "labelled_" + direction_label + "_bin_samples", model
    )
    out_name = direction_label + "_bin_proportions"
    fig = plot_bin_proportions(
        labelled_bin_samples, 
        f"{direction_label.split('_')[0]}, {model.cfg.model_name}"
    )
    save_pdf(fig, out_name, model)
    save_html(fig, out_name, model)
    save_pdf(fig, out_name, model)
    fig.show()

    activations = get_activations_cached(dataloader, direction_label)
    positive_threshold = activations[:, :, 1].flatten().quantile(.999).item()
    negative_threshold = activations[:, :, 1].flatten().quantile(.001).item()

    bottom_counts = labelled_bin_samples[labelled_bin_samples.activation.lt(negative_threshold)].sentiment.value_counts(normalize=True)
    bottom_counts = bottom_counts[bottom_counts.index != "Neutral"]
    
    top_counts = labelled_bin_samples[labelled_bin_samples.activation.gt(positive_threshold)].sentiment.value_counts(normalize=True)
    top_counts = top_counts[top_counts.index != "Neutral"]

    metric = (bottom_counts.max() + top_counts.max()) / 2
    df = pd.DataFrame({
        "direction": [direction_label],
        "metric": [metric],
    })
    update_csv(df, "classifier_accuracy", model, key_cols=["direction"])
#%%
