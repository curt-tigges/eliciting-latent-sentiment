# %%
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
from typing import Dict, Iterable, List, Optional, Tuple, Union, Literal
from transformer_lens import HookedTransformer
from transformer_lens.evals import make_owt_data_loader
from transformer_lens.utils import (
    get_dataset,
    tokenize_and_concatenate,
    get_act_name,
    test_prompt,
)
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
from utils.store import (
    load_array,
    save_html,
    save_array,
    is_file,
    get_model_name,
    clean_label,
    save_text,
    to_csv,
    get_csv,
    save_pdf,
    save_pickle,
    update_csv,
)
from utils.neuroscope import (
    plot_neuroscope,
    get_dataloader,
    get_projections_for_text,
    plot_top_p,
    plot_topk,
    harry_potter_start,
    harry_potter_fr_start,
    get_batch_pos_mask,
    extract_text_window,
    extract_activations_window,
    get_activations_from_dataloader,
    get_activations_cached,
)

# %%
pd.set_option("display.max_colwidth", 200)
torch.set_grad_enabled(False)
# %%
DIRECTIONS = [
    "kmeans_simple_train_ADJ_layer1",
    "pca_simple_train_ADJ_layer1",
    "mean_diff_simple_train_ADJ_layer1",
    "logistic_regression_simple_train_ADJ_layer1",
    "das_simple_train_ADJ_layer1",
]
# %%
device = "cuda"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
BATCH_SIZE = 16
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
    torch_dtype=torch.float16,
    dtype="float16",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
# %%
dataloader = get_dataloader(model, "stas/openwebtext-10k", batch_size=BATCH_SIZE)


# %%
def get_window(
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    window_size: int = 10,
) -> Tuple[int, int]:
    """Helper function to get the window around a position in a batch (used in topk plotting))"""
    lb = max(0, pos - window_size)
    ub = min(len(dataloader.dataset[batch]["tokens"]), pos + window_size + 1)
    return lb, ub


def extract_text_window(
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    window_size: int = 10,
) -> List[str]:
    """Helper function to get the text window around a position in a batch (used in topk plotting)"""
    assert model.tokenizer is not None
    expected_size = 2 * window_size + 1
    lb, ub = get_window(batch, pos, dataloader=dataloader, window_size=window_size)
    tokens = dataloader.dataset[batch]["tokens"][lb:ub]
    str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
    padding_to_add = expected_size - len(str_tokens)
    if padding_to_add > 0 and model.tokenizer.padding_side == "right":
        str_tokens += [model.tokenizer.bos_token] * padding_to_add
    elif padding_to_add > 0 and model.tokenizer.padding_side == "left":
        str_tokens = [model.tokenizer.bos_token] * padding_to_add + str_tokens
    assert len(str_tokens) == expected_size, (
        f"Expected text window of size {expected_size}, "
        f"found {len(str_tokens)}: {str_tokens}"
    )
    return str_tokens  # type: ignore


# %%
def sample_by_bin(
    data: Float[Tensor, "batch pos"],
    bins: int = 20,
    samples_per_bin: int = 20,
    seed: int = 0,
    window_size: int = 10,
    verbose: bool = True,
):
    if verbose:
        print("Calling sample_by_bin...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    flat = data.flatten().cpu().numpy()
    flat = flat[flat != 0]
    hist, bin_edges = np.histogram(flat, bins=bins)

    # plot histogram
    if verbose:
        print("Plotting histogram...")
    sample_to_plot = flat[np.random.randint(0, len(flat), bins * samples_per_bin)]
    fig = px.histogram(
        sample_to_plot,
        nbins=bins,
        labels={"value": "Activation"},
        title="Histogram of activations",
    )
    fig.show()

    bin_indices: Int[np.ndarray, "batch pos"] = np.digitize(
        data.cpu().numpy(), bin_edges
    )
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
                activations[
                    bin_batches[bin_sample], bin_positions[bin_sample], 1
                ].item(),
            )
            for bin_sample in bin_samples
        ]
    if verbose:
        print("Constructing dataframe...")
    df = pd.DataFrame(
        indices, columns=["bin", "lb", "ub", "batch", "position", "activation"]
    )
    tokens = []
    texts = []
    for _, row in df.iterrows():
        text = extract_text_window(
            int(row.batch),
            int(row.position),
            dataloader,
            model,
            window_size=window_size,
        )
        tokens.append(text[window_size])
        texts.append("".join(text))
    df.reset_index(drop=True, inplace=True)
    df["token"] = tokens
    df["text"] = texts
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# %% # Save samples
bar = tqdm(DIRECTIONS)
for direction_label in bar:
    bar.set_description(direction_label)
    samples_path = direction_label + "_bin_samples.csv"
    if is_file(samples_path, model):
        continue
    activations = get_activations_cached(dataloader, direction_label, model)
    bin_samples = sample_by_bin(activations[:, :, 1], verbose=False)
    to_csv(bin_samples, samples_path, model)


# %%
def plot_bin_proportions(
    df: pd.DataFrame,
    label: str,
    sentiments=(
        "Positive",
        "Negative",
        "Neutral",
    ),
    nbins: Optional[int] = 50,
):
    if "activation" in df.columns:
        assert nbins is not None
        if df.activation.dtype == pd.StringDtype:
            df.activation = df.activation.map(lambda x: eval(x).item()).astype(float)
        sentiments = sorted(df["sentiment"].unique())
        df = df.sort_values(by="activation").reset_index(drop=True)
        df["activation_cut"] = pd.cut(df.activation, bins=nbins)
        df.activation_cut = df.activation_cut.apply(lambda x: 0.5 * (x.left + x.right))
    else:
        df["activation"] = df["activation_cut"] = (df.lb + df.ub) * 0.5
        df = df.sort_values(by="activation_cut").reset_index(drop=True)

    fig = go.Figure()
    data = []

    for x, bin_df in df.groupby("activation_cut"):
        if bin_df.empty:
            continue
        label_props = bin_df.value_counts("sentiment", normalize=True, sort=False)
        data.append([label_props.get(sentiment, 0) for sentiment in sentiments])

    data = pd.DataFrame(data, columns=sentiments)
    cumulative_data = data.cumsum(axis=1)  # Cumulative sum along columns

    x_values = df["activation_cut"].unique()

    # Adding traces for the rest of the sentiments
    for idx, sentiment in enumerate(sentiments):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=cumulative_data[sentiment],
                name=sentiment,
                hovertemplate="<br>".join(
                    [
                        "Sentiment: " + sentiment,
                        "Sentiment Activation: %{x}",
                        "Cum. Label proportion: %{y:.4f}",
                    ]
                ),
                fill="tonexty",
                mode="lines",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Proportion of Sentiment by Activation ({label})",
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=True,
        xaxis_title="Sentiment Activation",
        yaxis_title="Cum. Label proportion",
        font=dict(size=24),  # global font settings  # global font size
    )

    return fig


# %%
for direction_label in DIRECTIONS:
    labelled_bin_samples = get_csv(
        "labelled_" + direction_label + "_bin_samples", model
    )
    out_name = direction_label + "_bin_proportions"
    if "das" in direction_label or "pca" in direction_label:
        labelled_bin_samples.activation *= -1
    fig = plot_bin_proportions(
        labelled_bin_samples, f"{direction_label.split('_')[0]}, {model.cfg.model_name}"
    )
    save_pdf(fig, out_name, model)
    save_html(fig, out_name, model)
    save_pdf(fig, out_name, model)
    fig.show()

    activations = get_activations_cached(dataloader, direction_label, model)
    flat = activations[:, :, 1].flatten().cpu()
    flat = flat[flat != 0]
    if "das" in direction_label or "pca" in direction_label:
        flat *= -1
    positive_threshold = flat.quantile(0.999).item()
    negative_threshold = flat.quantile(0.001).item()

    bottom_counts = labelled_bin_samples[
        labelled_bin_samples.activation.lt(negative_threshold)
    ].sentiment.value_counts(normalize=True)
    bottom_counts = bottom_counts[bottom_counts.index != "Neutral"]

    top_counts = labelled_bin_samples[
        labelled_bin_samples.activation.gt(positive_threshold)
    ].sentiment.value_counts(normalize=True)
    top_counts = top_counts[top_counts.index != "Neutral"]

    metric = (bottom_counts.max() + top_counts.max()) / 2
    df = pd.DataFrame(
        {
            "direction": [direction_label],
            "metric": [metric],
        }
    )
    assert not np.isnan(metric), (
        f"Metric is NaN for {direction_label}. "
        f"Bottom counts: {bottom_counts}. "
        f"Top counts: {top_counts}. "
        f"Thresholds: {negative_threshold}, {positive_threshold}"
    )
    print(direction_label, metric)
    update_csv(df, "classifier_accuracy", model, key_cols=["direction"])
# %%
