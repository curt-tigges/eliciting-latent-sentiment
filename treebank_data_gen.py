# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# !ls

# %%
# %cd eliciting-latent-sentiment

# %%
# !pip install circuitsvis

# %%
from enum import Enum
from typing import List
import pandas as pd
from datasets import Dataset, DatasetDict
import os
from transformer_lens import HookedTransformer
import torch
import plotly.express as px
from tqdm.auto import tqdm
from utils.treebank import get_merged_dataframe, convert_to_dataset_dict, create_datasets_for_model
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentence_phrase_df = get_merged_dataframe()
sentence_phrase_df.head()
# %%
sentence_phrase_df.split.value_counts()

# %%
convert_to_dataset_dict(sentence_phrase_df)
# %%
MODELS = [
    'gpt2-small',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
]
for model in tqdm(MODELS):
    model = HookedTransformer.from_pretrained(model, device=device)
    create_datasets_for_model(
        model, sentence_phrase_df, padding_side="left",
        batch_size=16, 
    )
 #%%
