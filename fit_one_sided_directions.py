#%%
import einops
from functools import partial
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Optional, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
import yaml
import plotly.express as px
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text
#%%
torch.set_grad_enabled(False)
device = "cuda"
model = HookedTransformer.from_pretrained(
    "gpt2-small",
)
#%%
ACT_NAME = get_act_name("resid_post", 0)
#%%
BATCH_SIZE = 64
owt_data = load_dataset("stas/openwebtext-10k", split="train")
dataset = tokenize_and_concatenate(owt_data, model.tokenizer)
data_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)
#%% # Neutral
count = 0
total = torch.zeros(model.cfg.d_model)
for batch in tqdm(data_loader):
    _, cache = model.run_with_cache(
        batch['tokens'], 
        return_type=None, 
        names_filter = lambda name: name == ACT_NAME
    )
    count += 1
    total += cache[ACT_NAME][:, 1, :].mean(dim=0).cpu()
neutral_activation = total / count
print(neutral_activation.shape, neutral_activation.norm())
#%% Handmade prompts
with open("prompts.yaml", "r") as f:
    prompt_dict = yaml.safe_load(f)
#%% Handmade neutral
# neutral_str_tokens = prompt_dict['neutral_adjectives']
# neutral_single_tokens = []
# for token in neutral_str_tokens:
#     token = " " + token
#     if len(model.to_str_tokens(token, prepend_bos=False)) == 1:
#         neutral_single_tokens.append(token)
# neutral_tokens = model.to_tokens(
#     neutral_single_tokens,
#     prepend_bos=True,
# )
# assert neutral_tokens.shape[1] == 2
# _, neutral_cache = model.run_with_cache(
#     neutral_tokens,
#     return_type=None,
#     names_filter = lambda name: name == ACT_NAME
# )
# neutral_activation = neutral_cache[ACT_NAME][:, -1].mean(dim=0).cpu()
# print(neutral_activation.shape, neutral_activation.norm())
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
torch.cosine_similarity(positive_direction, negative_direction, dim=0)
#%%
is_valenced_direction = positive_direction + negative_direction
is_valenced_direction = is_valenced_direction / is_valenced_direction.norm()
is_valenced_direction = is_valenced_direction.to(device)
sentiment_direction = positive_direction - negative_direction
sentiment_direction = sentiment_direction / sentiment_direction.norm()
sentiment_direction = sentiment_direction.to(device)
torch.cosine_similarity(is_valenced_direction, sentiment_direction, dim=0)
#%%
def get_token_sentiment_valence(
    max_tokens: int = 10_000,
    max_sentiment: Optional[float] = None,
    min_valence: Optional[float] = None,
):
    all_tokens = torch.tensor([], dtype=torch.int32, device=device,)
    val_scores = torch.tensor([], dtype=torch.float32, device=device,)
    sent_scores = torch.tensor([], dtype=torch.float32, device=device,)
    all_acts = torch.tensor([], dtype=torch.float32, device=device,)
    for batch in tqdm(data_loader):
        batch_tokens = batch['tokens'].to(device)
        _, cache = model.run_with_cache(
            batch_tokens, 
            return_type=None, 
            names_filter = lambda name: name == ACT_NAME
        )
        val_score = einops.einsum(
            cache[ACT_NAME],
            is_valenced_direction,
            "batch pos d_model, d_model -> batch pos",
        )
        sent_score = einops.einsum(
            cache[ACT_NAME],
            sentiment_direction,
            "batch pos d_model, d_model -> batch pos",
        )
        val_score = einops.rearrange(
            val_score, "batch pos -> (batch pos)"
        )
        sent_score = einops.rearrange(
            sent_score, "batch pos -> (batch pos)"
        )
        flat_tokens = einops.rearrange(
            batch_tokens, "batch pos -> (batch pos)"
        )
        flat_act = einops.rearrange(
            cache[ACT_NAME], "batch pos d_model -> (batch pos) d_model"
        )
        mask = torch.ones_like(flat_tokens, dtype=torch.bool)
        if max_sentiment is not None:
            mask &= sent_score.abs() < max_sentiment
        if min_valence is not None:
            mask &= val_score > min_valence
        flat_tokens = flat_tokens[mask]
        val_score = val_score[mask]
        sent_score = sent_score[mask]
        flat_act = flat_act[mask]
        all_tokens = torch.cat([all_tokens, flat_tokens])
        val_scores = torch.cat([val_scores, val_score])
        sent_scores = torch.cat([sent_scores, sent_score])
        all_acts = torch.cat([all_acts, flat_act])
        if len(all_tokens) > max_tokens:
            break
    val_scores = val_scores.cpu().numpy()
    sent_scores = sent_scores.cpu().numpy()
    all_tokens = all_tokens.cpu().numpy()
    all_acts = all_acts.cpu().numpy()
    print(val_scores.shape, sent_scores.shape, all_tokens.shape, all_acts.shape)
    return val_scores, sent_scores, all_tokens, all_acts
#%%
val_scores, sent_scores, all_tokens, all_acts = get_token_sentiment_valence(
    max_tokens=100,
    max_sentiment=0.5,
    min_valence=20,
)
if len(all_tokens) <= 1_000:
    all_tokens = model.to_str_tokens(all_tokens)
    save_text("\n".join(all_tokens), "valenced_tokens", model)
#%%
fig = px.scatter(
    x=val_scores,
    y=sent_scores,
    text=all_tokens,
    labels=dict(x="Valenced", y="Sentiment"),
)
fig.update_layout(
    title=dict(
        text="Valenced vs Sentiment activations",
        x=0.5,
    ),
    font=dict(
        size=8,
    ),
)
fig.show()
#%%
neutral_valenced_activation = all_acts.mean(axis=0)
# %%
positive_direction = positive_activation - neutral_valenced_activation
negative_direction = negative_activation - neutral_valenced_activation
positive_direction = positive_direction / positive_direction.norm()
negative_direction = negative_direction / negative_direction.norm()
torch.cosine_similarity(positive_direction, negative_direction, dim=0)
#%%
# save_array(
#     positive_direction.cpu().numpy(), "mean_diff_positive_layer01", model
# )
# save_array(
#     negative_direction.cpu().numpy(), "mean_diff_negative_layer01", model
# )

# %%
