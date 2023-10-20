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
NEGATIONS = {
    "Don't doubt it.": "doubt",
    "Don't hesitate.": "hesitate",
    'He was less than honorable in his actions.': 'honorable',
    "He's no stranger to success.": 'stranger',
    'I am not uncertain.': 'uncertain',
    'I am not unclear.': 'unclear',
    "I don't find it amusing at all.": 'amusing',
    "I don't like you.": 'like',
    "I don't respect that.": 'respect',
    # "I don't trust you.": 'trust',
    "It's hardly a success from my perspective.": 'success',
    "It's hardly a triumph in my eyes.": 'triumph',
    "It's hardly a victory in my eyes.": 'victory',
    'She failed to show any kindness.': 'kindness',
    'She was less than truthful in her account.': 'truthful',
    "She's no amateur.": 'amateur',
    "She's no novice.": 'novice',
    'That was not a wise choice.': 'wise',
    'You never disappoint.': 'disappoint',
    'You never fail.': 'fail',
    "You're nothing short of a genius.": 'short',
    "You're nothing short of amazing.": 'short',
    "You're nothing short of brilliant.": 'short',
    "You're nothing short of exceptional.": 'short',
    "You're nothing short of incredible.": 'short',
    "I don't enjoy this.": 'enjoy',
    "I don't want to be with you": 'want',
    "I don't want this.": 'want',
}
print(len(NEGATIONS))
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
def run_experiment(
    initial_layer: int,
    final_layer: int,
) -> pd.DataFrame:
    direction_scores = []
    bar = tqdm(DIRECTIONS)
    for direction_label in bar:
        bar.set_description(direction_label)
        direction = load_array(direction_label, model)
        direction = torch.tensor(direction, device=device, dtype=torch.float32)
        activations = get_activations_cached(dataloader, direction_label, model)
        mean = activations[:, :, :11].flatten().mean().item()
        std_dev = activations[:, :, :11].flatten().std().item()
        print(direction_label, mean, std_dev)
        raw_scores = []
        flip_binary = []
        flip_sizes = []
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
                text_activations[0, word_idx, final_layer] - 
                text_activations[0, word_idx, initial_layer]
            ).item()
            z_score = abs(act_change) / std_dev
            flip_size = 0.5 * act_change / (
                mean - text_activations[0, word_idx, initial_layer]
            )
            z_scores.append(z_score)
            flip_binary.append(flip_size > 0)
            flip_sizes.append(flip_size)
            raw_scores.append(act_change)
            if flip_size < -30:
                print(
                    text, flip_size,
                    text_activations[0, word_idx, initial_layer],
                    text_activations[0, word_idx, final_layer],
                )
        direction_scores.append([
            np.mean(raw_scores),
            np.mean(z_scores),
            np.mean(flip_binary),
            np.median(flip_sizes),
        ])
    df = pd.DataFrame(
        direction_scores,
        columns=["raw_score", "z_score", "flip_percent", "flip_size"],
        index=DIRECTIONS,
    )
    df = df.sort_values("raw_score", ascending=False)
    to_csv(df, "negation_experiment", model)
    return df
#%%
# for initial_layer in [0, 1]:
#     for final_layer in range(6, 13):
#         print(initial_layer, final_layer, run_experiment(
#             initial_layer=initial_layer,
#             final_layer=final_layer,
#         ))
#%%
run_experiment(1, 10)
#%%
