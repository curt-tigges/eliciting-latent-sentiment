#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.prompts import get_dataset
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils
from cache_utils import (
    residual_sentiment_sim_by_head, residual_sentiment_sim_by_pos
)
#%%
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.cfg.use_attn_results = True
# %%
sentiment_dir: Float[np.ndarray, "d_model"] = np.load(
    'data/km_line_embed_and_mlp0.npy'
)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(
    device, dtype=torch.float32
)
positive_dir: Float[np.ndarray, "d_model"] = np.load(
    'data/km_positive_embed_and_mlp0.npy'
)
positive_dir: Float[Tensor, "d_model"] = torch.tensor(positive_dir).to(
    device, dtype=torch.float32
)
negative_dir: Float[np.ndarray, "d_model"] = np.load(
    'data/km_negative_embed_and_mlp0.npy'
)
negative_dir: Float[Tensor, "d_model"] = torch.tensor(negative_dir).to(
    device, dtype=torch.float32
)
# %%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(
    model, device
)
#%%
def name_filter(name: str) -> bool:
    return name.endswith('result') or name.endswith('z') or name.endswith('_scale')
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, 
    names_filter = name_filter,
)
clean_cache.to(device)
# %%
heads = model.cfg.n_heads
layers = model.cfg.n_layers
batch_size, seq_len = clean_tokens.shape
#%%
example_prompt = model.to_str_tokens(clean_tokens[0])
example_prompt[4] = 'SBJ1'
example_prompt[17] = 'SBJ2'
example_prompt
#%%
sentiment_repeated = einops.repeat(
    sentiment_dir, "d_model -> batch d_model", batch=batch_size
)
even_batch_repeated = einops.repeat(
    torch.arange(batch_size, device=device) % 2 == 0, 
    "batch -> batch d_model", 
    d_model=len(sentiment_dir)
)
sentiment_directions: Float[Tensor, "batch d_model"] = torch.where(
    even_batch_repeated,
    sentiment_repeated,
    -sentiment_repeated,
).to(device)
#%%
del model
#%%
NORMALISE_RESIDUALS = False
CENTRE_RESIDUALS = True
HTML_SUFFIX = (
    ('_normalised' if NORMALISE_RESIDUALS else '') + 
    ('_centred' if CENTRE_RESIDUALS else '')
)
#%%
# ============================================================================ #
# By head

#%%

per_head_sentiment: Float[Tensor, "layer head"] = residual_sentiment_sim_by_head(
    clean_cache,
    sentiment_directions,
    centre_residuals=CENTRE_RESIDUALS,
    normalise_residuals=NORMALISE_RESIDUALS,
    layers=layers,
    heads=heads,
)
# %%
head_title = (
    'Which components align with the sentiment direction at END?'
    + (' (normalised)' if NORMALISE_RESIDUALS else '')
    + (' (centred)' if CENTRE_RESIDUALS else '')
)
fig = px.imshow(
    per_head_sentiment.cpu().detach().numpy(),
    labels={'x': 'Head', 'y': 'Layer'},
    title=head_title,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)
fig.write_html(f'data/sentiment_by_head{HTML_SUFFIX}.html')
fig.show()
#%%
# ============================================================================ #
# Split by position
#%%

#%%
per_pos_sentiment: Float[
    Tensor, "components pos"
] = residual_sentiment_sim_by_pos(
    clean_cache, 
    sentiment_directions,
    seq_len=seq_len,
    centre_residuals=CENTRE_RESIDUALS,
    normalise_residuals=NORMALISE_RESIDUALS,
)
# %%
pos_title = (
    'Which components align with the sentiment direction at each position?<br>'
    f'Settings: normalised={NORMALISE_RESIDUALS}, centred={CENTRE_RESIDUALS}'
)
fig = px.imshow(
    per_pos_sentiment.squeeze().cpu().detach().numpy(),
    labels={'x': 'Position', 'y': 'Component'},
    title=pos_title,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    x=example_prompt,
    y=[f'L{l}H{h}' for l in range(layers) for h in range(heads)],
    height = heads * layers * 20,
)
fig.write_html(f'data/sentiment_by_position{HTML_SUFFIX}.html')
fig.show()
# %%
fig = px.line(
    per_pos_sentiment.squeeze().cpu().detach().numpy(),
    labels={'index': 'Component', 'value': 'dot product', 'variable': 'Position'},
    title='Which components align with the sentiment direction at each position?',
    hover_name=[f'L{l}H{h}' for l in range(layers) for h in range(heads)],
)
example_prompt_dict = {f'{i}': f'{i}: {t}' for i, t in enumerate(example_prompt)}
fig.for_each_trace(lambda t: t.update(
    name = example_prompt_dict[t.name],
    legendgroup = example_prompt_dict[t.name],
    hovertemplate = t.hovertemplate.replace(t.name, example_prompt_dict[t.name])
))
fig.write_html(f'data/sentiment_by_position_line{HTML_SUFFIX}.html')
fig.show()
# %%
