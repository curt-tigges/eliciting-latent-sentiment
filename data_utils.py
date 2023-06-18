import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
import yaml

import einops
from fancy_einsum import einsum

from datasets import load_dataset
from transformers import pipeline
import plotly.io as pio
import plotly.express as px

# import pysvelte
from IPython.display import HTML

import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display

import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import transformer_lens
import transformer_lens.utils as utils
import transformer_lens.patching as patching
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

from functools import partial

from torchtyping import TensorType as TT

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"


def read_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    prompts_str, answers_str = content.split("\n\n")
    prompts = prompts_str.split("\n")  # Remove the last empty item
    answers = [
        tuple(answer.split(",")) for answer in answers_str.split(";")[:-1]
    ]  # Remove the last empty item

    return prompts, answers
