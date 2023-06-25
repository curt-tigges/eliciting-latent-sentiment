import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
import yaml

import einops
from fancy_einsum import einsum


import plotly.io as pio
import plotly.express as px

# import pysvelte
from IPython.display import HTML

import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display


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


import pandas as pd
import plotly.express as px


def plot_attention_heads(tensor, title="", top_n=0, range_x=[0, 2.5], threshold=0.02):
    # convert the PyTorch tensor to a numpy array
    values = tensor.cpu().detach().numpy()

    # create a list of labels for each head
    labels = []
    for layer in range(values.shape[0]):
        for head in range(values.shape[1]):
            label = f"Layer {layer}, Head {head}"
            labels.append(label)

    # flatten the values array
    flattened_values = values.flatten()

    if top_n > 0:
        # get the indices of the top N values
        top_indices = flattened_values.argsort()[-top_n:][::-1]

        # filter the flattened values and labels arrays based on the top N indices
        flattened_values = flattened_values[top_indices]
        labels = [labels[i] for i in top_indices]

        # sort the values and labels in descending order
        flattened_values, labels = zip(
            *sorted(zip(flattened_values, labels), reverse=False)
        )

    # create a dataframe with the flattened values and labels
    df = pd.DataFrame({"Logit Diff": flattened_values, "Attention Head": labels})
    flat_value_array = np.array(flattened_values)
    # print sum of all values over threshold
    print(
        f"Total logit diff contribution above threshold: {flat_value_array.sum():.2f}"
    )

    # create the plot
    fig = px.bar(
        df,
        x="Logit Diff",
        y="Attention Head",
        orientation="h",
        range_x=range_x,
        title=title,
    )
    fig.show()


def l_imshow(tensor, renderer=None, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show(renderer)


def l_line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)


def l_scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


def two_lines(tensor1, tensor2, renderer=None, **kwargs):
    px.line(y=[utils.to_numpy(tensor1), utils.to_numpy(tensor2)], **kwargs).show(
        renderer
    )


def get_attn_head_patterns(model, prompt, attn_heads):
    if isinstance(prompt, str):
        prompt = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(prompt, remove_batch_dim=True)

    head_list = []
    head_name_list = []
    for layer, head in attn_heads:
        head_list.append(cache["pattern", layer, "attn"][head])
        head_name_list.append(f"L{layer}H{head}")
    attention_pattern = torch.stack(head_list, dim=0)
    tokens = model.to_str_tokens(prompt)

    return tokens, attention_pattern, head_name_list


def scatter_attention_and_contribution(
    model,
    head,
    prompts,
    io_positions,
    s_positions,
    answer_residual_directions,
    return_vals=False,
    return_fig=False,
):

    df = []

    layer, head_idx = head
    # Get the attention output to the residual stream for the head
    logits, cache = model.run_with_cache(prompts)
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    scaled_residual_stack = cache.apply_ln_to_stack(
        per_head_residual, layer=-1, pos_slice=-1
    )
    head_resid = scaled_residual_stack[layer * model.cfg.n_heads + head_idx]

    # Loop over each prompt
    for i in range(len(answer_residual_directions)):
        # Get attention values
        tokens, attn, names = get_attn_head_patterns(model, prompts[i], [head])

        # For IO
        # Get the attention contribution in the residual directions
        dot = einsum(
            "d_model, d_model -> ", head_resid[i], answer_residual_directions[i][0]
        )

        # Get the attention probability to the IO answer
        prob = attn[0, 14, io_positions[i]]
        df.append([prob, dot, "IO", prompts[i]])

        # For S
        # Get the attention contribution in the residual directions
        dot = einsum(
            "d_model, d_model -> ", head_resid[i], answer_residual_directions[i][1]
        )
        # Get the attention probability to the S answer
        prob = attn[0, 14, s_positions[i]]
        df.append([prob, dot, "S", prompts[i]])

    # Plot the results
    viz_df = pd.DataFrame(
        df, columns=[f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"]
    )
    fig = px.scatter(
        viz_df,
        x=f"Attn Prob on Name",
        y=f"Dot w Name Embed",
        color="Name Type",
        hover_data=["text"],
        color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
        title=f"How Strong {layer}.{head_idx} Writes in the Name Embed Direction Relative to Attn Prob",
    )

    if return_vals:
        return viz_df
    if return_fig:
        return fig
    else:
        fig.show()

def scatter_attention_and_contribution_sentiment(
    model,
    head,
    prompts,
    positions,
    answer_residual_directions,
    return_vals=False,
    return_fig=False,
):

    df = []

    layer, head_idx = head
    # Get the attention output to the residual stream for the head
    logits, cache = model.run_with_cache(prompts)
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    scaled_residual_stack = cache.apply_ln_to_stack(
        per_head_residual, layer=-1, pos_slice=-1
    )
    head_resid = scaled_residual_stack[layer * model.cfg.n_heads + head_idx]

    # Loop over each prompt
    for i in range(len(answer_residual_directions)):
        # Get attention values
        tokens, attn, names = get_attn_head_patterns(model, prompts[i], [head])

        # For IO
        # Get the attention contribution in the residual directions
        dot = einsum(
            "d_model, d_model -> ", head_resid[i], answer_residual_directions[i][0]
        )

        # Get the attention probability to the answer
        prob = attn[0, -1, positions[i]]
        sentiment = "Positive" if i%2==0 else "Negative"
        df.append([prob, dot, f"{sentiment} Sentiment", prompts[i]])

    # Plot the results
    viz_df = pd.DataFrame(
        df, columns=[f"Attn Prob on Word", f"Dot w Sentiment Embed", "Word Type", "text"]
    )
    fig = px.scatter(
        viz_df,
        x=f"Attn Prob on Word",
        y=f"Dot w Sentiment Embed",
        color="Word Type",
        hover_data=["text"],
        color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
        title=f"How Strong {layer}.{head_idx} Writes in the Sentiment Embed Direction Relative to Attn Prob",
    )

    if return_vals:
        return viz_df
    if return_fig:
        return fig
    else:
        fig.show()



def scatter_attention_and_contribution_logic(
    model,
    head,
    prompts,
    answer_residual_directions,
    return_vals=False,
    return_fig=False,
):

    df = []

    layer, head_idx = head
    # Get the attention output to the residual stream for the head
    logits, cache = model.run_with_cache(prompts)
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    scaled_residual_stack = cache.apply_ln_to_stack(
        per_head_residual, layer=-1, pos_slice=-1
    )
    head_resid = scaled_residual_stack[layer * model.cfg.n_heads + head_idx]

    # Loop over each prompt
    for i in range(len(answer_residual_directions)):
        # Get attention values
        tokens, attn, names = get_attn_head_patterns(model, prompts[i], [head])

        # For IO
        # Get the attention contribution in the residual directions
        dot = einsum(
            "d_model, d_model -> ", head_resid[i], answer_residual_directions[i][0]
        )

        # Get the attention probability to the correct answer
        prob = attn[0, 16, 4]
        df.append([prob, dot, "Descriptor", prompts[i]])

        # For S
        # Get the attention contribution in the residual directions
        dot = einsum(
            "d_model, d_model -> ", head_resid[i], answer_residual_directions[i][1]
        )
        # Get the attention probability to the S answer
        prob = attn[0, 16, 2]
        df.append([prob, dot, "S", prompts[i]])

    # Plot the results
    viz_df = pd.DataFrame(
        df, columns=[f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"]
    )
    fig = px.scatter(
        viz_df,
        x=f"Attn Prob on Name",
        y=f"Dot w Name Embed",
        color="Name Type",
        hover_data=["text"],
        color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
        title=f"How Strong {layer}.{head_idx} Writes in the Name Embed Direction Relative to Attn Prob",
    )

    if return_vals:
        return viz_df
    if return_fig:
        return fig
    else:
        fig.show()
