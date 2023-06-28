#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
from utils.store import load_array
# %%
MODEL_NAME = "gpt2-small"
embed_only = load_array('km_line_embed_only', MODEL_NAME)
embed_and_mlp = load_array('km_line_embed_and_mlp0', MODEL_NAME)
unembed_transpose = load_array('km_line_unembed_transpose', MODEL_NAME)
# %%
px.line(embed_and_mlp, title='Components of sentiment direction')
#%%
px.histogram(
    embed_and_mlp, title='Distribution of components of sentiment direction'
)
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
