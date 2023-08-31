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
import torch

import random

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformer_lens import HookedTransformer
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm.notebook import tqdm
from utils.store import load_pickle, load_array
from utils.ablation import ablate_resid_with_precalc_mean


# %%
BATCH_SIZE = 5
MODEL_NAME ="gpt2"
DATASET_FOLDER = "sst2"


# %%

dataset = load_from_disk(DATASET_FOLDER)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(7000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))

# %%
model = AutoModelForCausalLM.from_pretrained("./gpt2_imdb_classifier")
class_layer_weights = load_pickle("gpt2_imdb_classifier_classification_head_weights", 'gpt2')

model = HookedTransformer.from_pretrained(
    "gpt2",
    hf_model=model,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# %%
def get_classification_prediction(eval_dataset, dataset_idx, verbose=False):

    logits, cache = model.run_with_cache(small_eval_dataset[dataset_idx]['text'])
    last_token_act = cache['ln_final.hook_normalized'][0, -1, :]
    res = torch.softmax(torch.tensor(class_layer_weights['score.weight']) @ last_token_act.cpu(), dim=-1)
    if verbose:
        print(f"Sentence: {small_eval_dataset[dataset_idx]['text']}")
        print(f"Prediction: {res.argmax()} Label: {small_eval_dataset[dataset_idx]['label']}")

    return res.argmax(), small_eval_dataset[dataset_idx]['label'], res

# %%
def get_accuracy(eval_dataset, n=300):
    correct = 0
    for idx in range(min(len(eval_dataset), n)):
        pred, label, _ = get_classification_prediction(eval_dataset, idx)
        if pred == label:
            correct += 1
    return correct / n

get_accuracy(small_eval_dataset)

# %%
