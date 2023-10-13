#%%
import einops
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
import yaml
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text
#%%
torch.set_grad_enabled(False)
device = "cuda"
model = HookedTransformer.from_pretrained(
    "gpt2-small",
)
#%%
ACT_NAME = get_act_name("resid_post", 0)
# #%%
# BATCH_SIZE = 64
# owt_data = load_dataset("stas/openwebtext-10k", split="train")
# dataset = tokenize_and_concatenate(owt_data, model.tokenizer)
# data_loader = DataLoader(
#     dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
# )
# #%% # Neutral
# count = 0
# total = torch.zeros(model.cfg.d_model)
# for batch in tqdm(data_loader):
#     _, cache = model.run_with_cache(
#         batch['tokens'], 
#         return_type=None, 
#         names_filter = lambda name: name == ACT_NAME
#     )
#     count += 1
#     total += cache[ACT_NAME][:, 1, :].mean(dim=0).cpu()
# neutral_activation = total / count
# print(neutral_activation.shape, neutral_activation.norm())
#%% Handmade prompts
with open("prompts.yaml", "r") as f:
    prompt_dict = yaml.safe_load(f)
#%% Handmade neutral
neutral_str_tokens = prompt_dict['neutral_adjectives']
neutral_single_tokens = []
for token in neutral_str_tokens:
    token = " " + token
    if len(model.to_str_tokens(token, prepend_bos=False)) == 1:
        neutral_single_tokens.append(token)
neutral_tokens = model.to_tokens(
    neutral_single_tokens,
    prepend_bos=True,
)
assert neutral_tokens.shape[1] == 2
_, neutral_cache = model.run_with_cache(
    neutral_tokens,
    return_type=None,
    names_filter = lambda name: name == ACT_NAME
)
neutral_activation = neutral_cache[ACT_NAME][:, -1].mean(dim=0).cpu()
print(neutral_activation.shape, neutral_activation.norm())
#%% # Positive
#%%
positive_str_tokens = (
    prompt_dict['positive_adjectives_train'] + 
    prompt_dict['positive_comment_adjectives'] +
    prompt_dict['positive_nouns'] + 
    prompt_dict['positive_verbs'] + 
    prompt_dict['positive_infinitives']
)
positive_single_tokens = []
for token in positive_str_tokens:
    token = " " + token
    if len(model.to_str_tokens(token, prepend_bos=False)) == 1:
        positive_single_tokens.append(token)
positive_tokens = model.to_tokens(
    positive_single_tokens,
    prepend_bos=True,
)
assert positive_tokens.shape[1] == 2
_, positive_cache = model.run_with_cache(
    positive_tokens,
    return_type=None,
    names_filter = lambda name: name == ACT_NAME
)
positive_activation = positive_cache[ACT_NAME][:, -1].mean(dim=0).cpu()
print(positive_activation.shape, positive_activation.norm())
#%% # Negative
negative_str_tokens = (
    prompt_dict['negative_adjectives_train'] + 
    prompt_dict['negative_comment_adjectives'] +
    prompt_dict['negative_nouns'] + 
    prompt_dict['negative_verbs'] + 
    prompt_dict['negative_infinitives']
)
negative_single_tokens = []
for token in negative_str_tokens:
    token = " " + token
    if len(model.to_str_tokens(token, prepend_bos=False)) == 1:
        negative_single_tokens.append(token)
negative_tokens = model.to_tokens(
    negative_single_tokens,
    prepend_bos=True,
)
assert negative_tokens.shape[1] == 2
_, negative_cache = model.run_with_cache(
    negative_tokens,
    return_type=None,
    names_filter = lambda name: name == ACT_NAME
)
negative_activation = negative_cache[ACT_NAME][:, -1].mean(dim=0).cpu()
print(negative_activation.shape, negative_activation.norm())
# %%
positive_direction = positive_activation - neutral_activation
negative_direction = negative_activation - neutral_activation
positive_direction = positive_direction / positive_direction.norm()
negative_direction = negative_direction / negative_direction.norm()
#%%
save_array(
    positive_direction.cpu().numpy(), "mean_diff_positive_layer01", model
)
save_array(
    negative_direction.cpu().numpy(), "mean_diff_negative_layer01", model
)
#%% # compute cosine similarity
torch.cosine_similarity(positive_direction, negative_direction, dim=0)

# %%
