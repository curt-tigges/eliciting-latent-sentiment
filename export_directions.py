#%%
import einops
import numpy as np
from jaxtyping import Float
import plotly.express as px
import re
import os
import pandas as pd
import json
from typing import Union
import torch
from transformer_lens import HookedTransformer
from utils.store import (
    load_array, save_html, extract_layer_from_string, 
    zero_pad_layer_string, flatten_multiindex, DIRECTION_PATTERN, 
    clean_label, get_model_name, save_json
)
#%%
MODEL_NAME = "gpt2-small"
#%%
def get_directions():
    dir_path = os.path.join('data', MODEL_NAME)
    matching_files = [
        filename 
        for filename in os.listdir(dir_path) 
        if re.match(DIRECTION_PATTERN, filename) 
        and "2d" not in filename and "3d" not in filename and "treebank" not in filename
        and "_ALL" not in filename and "das_simple_train_ADJ" in filename
    ]
    return sorted(matching_files)
#%%
direction_labels = get_directions()
direction_labels
# %%
for label in direction_labels:
    direction = load_array(label, MODEL_NAME)
    save_json(direction, label, MODEL_NAME)

# %%
