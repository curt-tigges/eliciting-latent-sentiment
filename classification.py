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
from functools import partial
from typing import List, Optional, Union
from typeguard import typechecked
import jaxtyping
import torch
from torch import Tensor
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from tqdm.notebook import tqdm
from utils.store import load_pickle, load_array
from utils.classifier import HookedClassifier


# %%
BATCH_SIZE = 5
MODEL_NAME ="gpt2"
DATASET_FOLDER = "data/sst2"


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
model = HookedClassifier.from_pretrained(
    "data/gpt2-small/gpt2_imdb_classifier",
    "gpt2_imdb_classifier_classification_head_weights",
    "gpt2",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
#%%
model([small_eval_dataset[i]['text'] for i in range(5)]).shape

# %%
def get_classification_prediction(eval_dataset, dataset_idx, verbose=False):

    _, cache = model.run_with_cache(eval_dataset[dataset_idx]['text'], return_type=None)
    last_token_act = cache['ln_final.hook_normalized'][0, -1, :]
    res = torch.softmax(torch.tensor(class_layer_weights['score.weight']) @ last_token_act.cpu(), dim=-1)
    if verbose:
        print(f"Sentence: {eval_dataset[dataset_idx]['text']}")
        print(f"Prediction: {res.argmax()} Label: {eval_dataset[dataset_idx]['label']}")

    return res.argmax(), eval_dataset[dataset_idx]['label'], res
#%%
get_classification_prediction(small_eval_dataset, 0, verbose=True)
#%%
def forward_override(
    model: HookedTransformer,
    input: Union[str, List[str], jaxtyping.Int[Tensor, 'batch pos']],
    return_type: Optional[str] = 'logits',
):
    _, cache = model.run_with_cache(input, return_type=None)
    last_token_act = cache['ln_final.hook_normalized'][0, -1, :]
    logits = torch.softmax(
        torch.tensor(class_layer_weights['score.weight']) @ last_token_act.cpu(), 
        dim=-1
    )
    if return_type == 'logits':
        return logits
    elif return_type == 'prediction':
        return logits.argmax()
#%%
forward_override(model, small_eval_dataset[0]['text'], return_type='prediction')
#%%
model.forward = forward_override
#%%
model(small_eval_dataset[0]['text'])
# %%
def get_accuracy(eval_dataset, n=300):
    correct = 0
    for idx in range(min(len(eval_dataset), n)):
        pred, label, _ = get_classification_prediction(eval_dataset, idx)
        if pred == label:
            correct += 1
    return correct / n
#%%
get_accuracy(small_eval_dataset)
# %%
