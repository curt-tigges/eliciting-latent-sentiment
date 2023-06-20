#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
from prompt_utils import get_dataset
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, utils
#%%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
)
#%%
def residual_stack_to_sentiment_directions(
    residual_stack: Float[Tensor, "components batch d_model"], 
    cache: ActivationCache,
) -> Float[Tensor, "layer head_index"]:
    scaled_residual_stack: Float[
        Tensor, "components batch d_model"
    ] = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    component_means: Float[Tensor, "components"] = einops.einsum(
        scaled_residual_stack, sentiment_directions, 
        "components batch d_model, batch d_model -> components"
    ) / batch_size
    return einops.rearrange(
        component_means, 
        "(layer head_index) -> layer head_index", 
        layer=model.cfg.n_layers, 
        head_index=model.cfg.n_heads
    )

#%%
per_head_residual: Float[Tensor, "components batch d_model"]
per_head_residual, labels = clean_cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_sentiment_metrics = residual_stack_to_sentiment_directions(
    per_head_residual, clean_cache
)

# %%
px.imshow(
    per_head_sentiment_metrics.cpu().detach().numpy(),
    labels={'x': 'Position', 'y': 'Layer'},
    title='Which components align with the sentiment direction?',
)
# %%
