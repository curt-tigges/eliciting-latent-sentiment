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
# %%
sentiment_dir: Float[np.ndarray, "d_model"] = np.load(
    'data/km_line_embed_and_mlp0.npy'
)
sentiment_dir /= np.linalg.norm(sentiment_dir)
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
clean_tokens, corrupted_tokens, answer_tokens = get_dataset(model, device)
#%%
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
# %%
heads = model.cfg.n_heads
layers = model.cfg.n_layers
batch_size, seq_len = clean_tokens.shape
#%%
example_prompt = model.to_str_tokens(clean_tokens[0])
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
def residual_stack_to_sentiment_metrics(
    residual_stack: Float[Tensor, "components batch d_model"], 
    cache: ActivationCache,
    normalise_residuals: bool = True,
) -> Float[Tensor, "components"]:
    scaled_residual_stack: Float[
        Tensor, "components batch d_model"
    ] = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    if normalise_residuals: # for cosine similarity
        scaled_residual_stack = (
            scaled_residual_stack.T /
            scaled_residual_stack.norm(dim=-1).T
        ).T
    component_means: Float[Tensor, "components"] = einops.einsum(
        scaled_residual_stack, sentiment_directions, 
        "components batch d_model, batch d_model -> components"
    ) / batch_size
    return component_means
# by head
#%%
per_head_residual: Float[
    Tensor, "components batch d_model"
] = clean_cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=False)
per_head_sentiment_flat: Float[
    Tensor, "components"
] = residual_stack_to_sentiment_metrics(per_head_residual, clean_cache)
per_head_sentiment: Float[Tensor, "layer head"] = einops.rearrange(
    per_head_sentiment_flat, 
    "(layer head) -> layer head", 
    layer=model.cfg.n_layers, 
    head=model.cfg.n_heads
)
# %%
fig = px.imshow(
    per_head_sentiment.cpu().detach().numpy(),
    labels={'x': 'Head', 'y': 'Layer'},
    title='Which components align with the sentiment direction?',
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)
fig.show()
del per_head_residual, per_head_sentiment_flat, per_head_sentiment, model
#%%
# ============================================================================ #
# Split by position
#%%
def residual_cosine_sim_by_pos(
    cache: ActivationCache,
) -> Float[Tensor, "components"]:
    residual_stack: Float[
        Tensor, "components batch pos d_model"
    ] = cache.stack_head_results(layer=-1, return_labels=False)
    for pos in range(seq_len):
        residual_stack[:, :, pos, :] = cache.apply_ln_to_stack(
            residual_stack[:, :, pos, :], layer=-1, pos_slice=pos
        )
        residual_stack[:, :, pos, :] = (
            residual_stack[:, :, pos, :].T /
            residual_stack[:, :, pos, :].norm(dim=-1).T
        ).T
    component_means: Float[Tensor, "components"] = einops.einsum(
        residual_stack, sentiment_directions, 
        "components batch pos d_model, batch d_model -> components pos"
    ) / batch_size
    return einops.rearrange(
        component_means, 
        "(layer head) pos -> layer head pos", 
        layer=layers, 
        head=heads,
    )
#%%
per_pos_sentiment: Float[
    Tensor, "layer head pos"
] = residual_cosine_sim_by_pos(clean_cache)
# %%
fig = make_subplots(
    cols=1, 
    rows=seq_len, 
    subplot_titles=[
        f'Position {pos}: {example_prompt[pos]}' for pos in range(seq_len)
    ],
)
# plot a heatmap for each token position
for pos in range(seq_len):
    fig.add_trace(
        go.Heatmap(
            z=per_pos_sentiment[:, :, pos].cpu().detach().numpy(),
            x=np.arange(layers),
            y=np.arange(heads),
            name=f'pos_{pos}',
            colorscale="RdBu",
            zmin=-0.1,
            zmax=0.1,
            hovertemplate=(
                "Head: %{x}<br>" +
                "Layer: %{y}<br>" +
                "Cosine similarity: %{z}<br>"
            ),
        ),
        col=1,
        row=pos+1,
    )
    fig.update_yaxes(
        title_text="Layer",
    )
    fig.update_xaxes(
        title_text="Head",
    )
fig.update_layout(
    title='Which positions align with the sentiment direction?',
    height=seq_len*200,
)
fig.write_html('data/sentiment_by_position.html')
fig.show()
#%%
head_averages = einops.reduce(
    per_pos_sentiment,
    "layer head pos -> layer pos",
    reduction='mean',
)
fig = px.imshow(
    head_averages.cpu().detach().numpy(),
    labels={'x': 'Position', 'y': 'Layer'},
    title='Which positions align with the sentiment direction?',
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    x=example_prompt,
)
fig.show()
# %%
fig = px.imshow(
    per_pos_sentiment[:, 4, :].squeeze().cpu().detach().numpy(),
    labels={'x': 'Position', 'y': 'Layer'},
    title='Head 4: Which positions align with the sentiment direction?',
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    x=example_prompt,
)
fig.show()
# %%
