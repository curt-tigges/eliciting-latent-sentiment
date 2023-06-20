#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
from prompt_utils import get_dataset
import torch
from torch import Tensor
from transformer_lens import HookedTransformer, utils
# %%
special_dir = np.load('data/km_line_embed_and_mlp0.npy')
# special_dir /= np.linalg.norm(special_dir)
#%%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
clean_tokens, corrupted_tokens, answer_tokens = get_dataset(model, device)
#%%
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
# %%
heads = model.cfg.n_heads
layers = model.cfg.n_layers
batch_size, seq_len = clean_tokens.shape

# %%
cosine_similarities = np.empty((layers, seq_len))
for layer in range(layers):
    layer_name = utils.get_act_name('attn_out', layer)
    layer_cache: Float[Tensor, "batch pos d_model"] = clean_cache[layer_name]
    for pos in range(seq_len):
        avg_residual: Float[np.ndarray, "d_model"] = (
            layer_cache[:, pos, :].mean(dim=0).cpu().detach().numpy()
        )
        # avg_residual = avg_residual / avg_residual.norm()
        dot = np.dot(avg_residual, special_dir)
        cosine_similarities[layer, pos] = dot
# %%
px.imshow(cosine_similarities)