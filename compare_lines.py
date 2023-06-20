#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
# %%
embed_only = np.load('data/km_line_embed_only.npy')
embed_and_mlp = np.load('data/km_line_embed_and_mlp0.npy')
unembed_transpose = np.load('data/km_line_unembed_transpose.npy')
# %%
# %%
stacked: Float[np.ndarray, "3 d"] = np.stack([
    embed_only, embed_and_mlp, unembed_transpose
])
stacked = (stacked.T / np.linalg.norm(stacked, axis=1)).T
similarities: Float[np.ndarray, "3 d"] = einops.einsum(
    stacked, stacked, 'm d, n d -> m n'
)
# %%
px.imshow(
    similarities, 
    labels={'x': 'embed_type', 'y': 'embed_type'},
    x=['embed_only', 'embed_and_mlp', 'unembed_transpose'],
    y=['embed_only', 'embed_and_mlp', 'unembed_transpose'],
    # add labels to grid

)
# %%
1 / np.sqrt(768)
# %%
similarities
# %%
