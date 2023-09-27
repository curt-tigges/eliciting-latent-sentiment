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
    extract_activations_window, get_activations_from_dataloader,
    get_activations_cached,
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
NEGATIONS = [
    "Don't doubt it.",
    "Don't hesitate.",
    'He was less than honorable in his actions.',
    "He's no stranger to success.",
    'I am not uncertain.',
    'I am not unclear.',
    "I don't find it amusing at all.",
    "I don't like you.",
    "I don't respect that.",
    "I don't trust you.",
    "It's hardly a success from my perspective.",
    "It's hardly a triumph in my eyes.",
    "It's hardly a victory in my eyes.",
    'She failed to show any kindness.',
    'She was less than truthful in her account.',
    "She's no amateur.",
    "She's no novice.",
    'That was not a wise choice.',
    'You never disappoint.',
    'You never fail.',
    "You're nothing short of a genius.",
    "You're nothing short of amazing.",
    "You're nothing short of brilliant.",
    "You're nothing short of exceptional."
    "I don't enjoy this.",
    "You're nothing short of incredible.",
    "I don't want to be with you",
    "I don't want this.",
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
direction_scores = {}
for direction_label in tqdm(DIRECTIONS):
    direction = load_array(direction_label, model)
    activations = get_activations_cached(dataloader, direction_label, model)
    std_dev = activations[:, :, :11].flatten().std().item()
    z_scores = []
    for text, word in NEGATIONS.items():
        str_tokens = [tok.strip() for tok in model.to_str_tokens(text)]
        word_idx = str_tokens.index(word)
        text_activations = get_projections_for_text(
            text, direction, model
        )
        assert text_activations.shape[0] == 1
        assert len(str_tokens) == text_activations.shape[1]
        act_change = (
            text_activations[0, word_idx, 10] - 
            text_activations[0, word_idx, 1]
        ).item()
        z_score = abs(act_change) / std_dev
        z_scores.append(z_score)
    overall_score = np.mean(z_scores)
    direction_scores[direction_label] = overall_score
#%%
df = pd.DataFrame.from_dict(direction_scores, orient="index", columns=["z_score"])
df = df.sort_values("z_score", ascending=False)
to_csv(df, "negation_experiment")
df
#%%
