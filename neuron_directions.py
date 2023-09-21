#%%
from functools import partial
import itertools
import gc
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
import einops
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union, Literal
from transformer_lens import HookedTransformer
from transformer_lens.evals import make_owt_data_loader
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from circuitsvis.activations import text_neuron_activations
from circuitsvis.utils.render import RenderedHTML
from tqdm.notebook import tqdm
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import scipy.stats as stats
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text, to_csv, get_csv, save_pdf, save_pickle
#%%
pd.set_option('display.max_colwidth', 200)
torch.set_grad_enabled(False)
#%%
device = "cuda"
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
)
model.name = MODEL_NAME
#%%
out_directions = einops.rearrange(
    model.W_out,
    "layer d_mlp d_model -> (layer d_mlp) d_model",
)
out_directions /= out_directions.norm(dim=-1, keepdim=True)
neuron_labels = [
    f"L{layer}N{neuron}"
    for layer, neuron in itertools.product(
        range(model.cfg.n_layers),
        range(model.cfg.d_mlp)
    )
]
#%%
sentiment_dir = load_array("kmeans_simple_train_ADJ_layer1", model)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(device=device, dtype=torch.float32)
sentiment_dir /= sentiment_dir.norm()
#%%
neuron_similarities = einops.einsum(
    out_directions,
    sentiment_dir,
    "neuron d_mlp, d_mlp -> neuron",
)
#%%
sim_df = pd.DataFrame({
    "neuron": neuron_labels,
    "similarity": neuron_similarities.cpu().numpy(),
    "layer": [lab.split("N")[0] for lab in neuron_labels],
})
#%%
#%%
fig = px.histogram(
    data_frame=sim_df, 
    x="similarity", 
    color="layer", 
    hover_name="neuron",
    marginal="box", 
    nbins=200,
)
for index, row in sim_df.iterrows():
    if row["neuron"] in ("L6N828", "L3N1605", "L5N671", "L6N1237"):
        fig.add_annotation(
            x=row["similarity"],
            y=row["layer"],  
            text=row["neuron"],
            xref="x2",
            yref="y2",
            showarrow=True,
        )
fig.update_layout(
    title_text="Similarity of Neuron Out Directions to Sentiment Direction",
    title_x=0.5,
)
save_pdf(fig, "neuron_out_similarities", model)
save_html(fig, "neuron_out_similarities", model)
save_pdf(fig, "neuron_out_similarities", model)
fig.show()
# %%
