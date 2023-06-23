#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prompt_utils import get_dataset
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils
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
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
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
NORMALISE_RESIDUALS = True
CENTRE_RESIDUALS = False
HTML_SUFFIX = (
    ('_normalised' if NORMALISE_RESIDUALS else '') + 
    ('_centred' if CENTRE_RESIDUALS else '')
)
#%%
def residual_sentiment_sim_by_head(
    cache: ActivationCache,
) -> Float[Tensor, "layer head"]:
    residual_stack: Float[
        Tensor, "components batch d_model"
    ] = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=False)
    residual_stack: Float[
        Tensor, "components batch d_model"
    ] = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    if CENTRE_RESIDUALS:
        residual_stack -= einops.reduce(
            residual_stack, 
            "components batch d_model -> components 1 d_model", 
            "mean"
        )
    if NORMALISE_RESIDUALS:
        residual_stack = (
            residual_stack.T /
            residual_stack.norm(dim=-1).T
        ).T
        sent_dirs = (
            sentiment_directions.T /
            sentiment_directions.norm(dim=-1).T
        ).T
    else:
        sent_dirs = sentiment_directions
    component_means: Float[Tensor, "components"] = einops.einsum(
        residual_stack, sent_dirs, 
        "components batch d_model, batch d_model -> components"
    ) / batch_size
    return einops.rearrange(
        component_means, 
        "(layer head) -> layer head", 
        layer=layers, 
        head=heads,
    )

# ============================================================================ #
# By head

#%%

per_head_sentiment: Float[Tensor, "layer head"] = residual_sentiment_sim_by_head(
    clean_cache
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
def residual_sentiment_sim_by_pos(
    cache: ActivationCache,
) -> Float[Tensor, "components"]:
    residual_stack: Float[
        Tensor, "components batch pos d_model"
    ] = cache.stack_head_results(layer=-1, return_labels=False)
    for pos in range(seq_len):
        residual_stack[:, :, pos, :] = cache.apply_ln_to_stack(
            residual_stack[:, :, pos, :], layer=-1, pos_slice=pos
        )
    if CENTRE_RESIDUALS:
        residual_stack -= einops.reduce(
            residual_stack, 
            "components batch pos d_model -> components 1 1 d_model", 
            "mean"
        )
    if NORMALISE_RESIDUALS:
        residual_stack = (
            residual_stack.T /
            residual_stack.norm(dim=-1).T
        ).T
        sent_dirs = (
            sentiment_directions.T /
            sentiment_directions.norm(dim=-1).T
        ).T
    else:
        sent_dirs = sentiment_directions
    component_means: Float[Tensor, "components"] = einops.einsum(
        residual_stack, sent_dirs, 
        "components batch pos d_model, batch d_model -> components pos"
    ) / batch_size
    return component_means
#%%
per_pos_sentiment: Float[
    Tensor, "components pos"
] = residual_sentiment_sim_by_pos(clean_cache)
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
