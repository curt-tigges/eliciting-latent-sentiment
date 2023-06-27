#%%
import einops
import numpy as np
from jaxtyping import Float, Int, Bool
import plotly.express as px
from utils.prompts import get_dataset, get_onesided_datasets
from utils.circuit_analysis import get_log_probs
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from typing import Tuple, Union, List, Optional, Callable
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from utils.store import store_array
#%% # Model loading
device = torch.device('cpu')
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.name = MODEL_NAME
model.requires_grad_ = False
#%%
pos_adj = [
    ' perfect', ' fantastic',' delightful',' cheerful',' marvelous',' good',' remarkable',' wonderful',
    ' fabulous',' outstanding',' awesome',' exceptional',' incredible',' extraordinary',
    ' amazing',' lovely',' brilliant',' charming',' terrific',' superb',' spectacular',' great',' splendid',
    ' beautiful',' joyful',' positive',' excellent',
    ' breathtaking', ' stunning', ' impressive', ' admirable', ' phenomenal', 
    ' radiant', ' sublime', ' glorious', ' magical', ' sensational', ' pleasing', ' movie',
     
]
neg_adj = [
    ' dreadful',' bad',' dull',' depressing',' miserable',' tragic',' nasty',' inferior',' horrific',' terrible',
    ' ugly',' disgusting',' disastrous',' horrendous',' annoying',' boring',' offensive',' frustrating',' wretched',' dire',
    ' awful',' unpleasant',' horrible',' mediocre',' disappointing',' inadequate',
    ' foul', ' vile', ' appalling', ' rotten', ' grim', ' dismal',
    ' deficient',
    ' disastrous',
    ' dismal',
    ' dreadful',
    ' hateful',
    ' hideous',
    ' inadequate',
    ' inferior',
    ' insufficient',
    ' lousy',
    ' miserable',
    ' poor',
    ' shameful',
    ' sorry',
    ' tragic',
    ' troublesome',
    ' unbearable',
    ' uncomfortable',
    ' unfavorable',
    ' unfortunate',
    ' unlucky',
    ' unpleasant',
    ' unworthy',
    ' wretched'
    
]
neutral_adj = [
    ' average',' normal',' standard',' typical',' common',' ordinary',
    ' regular',' usual',' familiar',' generic',' conventional',
    ' fine', ' okay', ' ok', ' decent', ' fair', ' satisfactory', 
    ' adequate', ' alright',

]

for adj in pos_adj + neg_adj + neutral_adj:
    model.to_single_token(adj)
#%%
prompt_return_dict, answer_tokens = get_onesided_datasets(
    model, 
    device, 
    answer_sentiment='negative',
    dataset_sentiments=['positive', 'negative', 'neutral'],
    n_answers=5,
    positive_adjectives=pos_adj,
    negative_adjectives=neg_adj,
    neutral_adjectives=neutral_adj,
)
orig_tokens = prompt_return_dict['positive']
neutral_tokens = prompt_return_dict['neutral']
new_tokens = prompt_return_dict['negative']
#%%
all_tokens = torch.cat([orig_tokens, new_tokens, neutral_tokens])
all_tokens.shape
#%%
example_prompt = model.to_str_tokens(orig_tokens[0])
adj_token = example_prompt.index(' perfect')
verb_token = example_prompt.index(' loved')
s2_token = example_prompt.index(' movie', example_prompt.index(' movie') + 1)
end_token = len(example_prompt) - 1
#%%
ACT_NAME = 'blocks.0.hook_resid_post'
_, cache = model.run_with_cache(
    all_tokens, names_filter=lambda name: name == ACT_NAME
)
#%%
cache.to(device)
cache[ACT_NAME].requires_grad = True
#%%
#%%
def grad_hook(_: Float[Tensor, "batch pos d_model"], hook: HookPoint):
    assert hook.name == ACT_NAME
    return cache[ACT_NAME]
#%%
logits = model.run_with_hooks(
    all_tokens, 
    fwd_hooks=[(ACT_NAME, grad_hook)],
)
# %%
log_prob = get_log_probs(
    logits, answer_tokens, per_prompt=False
)
#%%
log_prob.backward()
#%%
gradient = einops.reduce(
    cache[ACT_NAME].grad[:, (adj_token, verb_token), :],
    'batch pos d_model -> d_model',
    'mean',
)
# %%
gradient.shape
# %%
km_line = torch.tensor(
    np.load('data/km_line_embed_and_mlp0.npy'), 
    device=device,
    dtype=torch.float32,
)
#%%
cosine_sim = einops.einsum(
    gradient / gradient.norm(),
    km_line / km_line.norm(),
    'd_model, d_model -> ',
)
cosine_sim
# %%
store_array(gradient, 'derivative_log_prob', model)
# %%
