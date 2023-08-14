import os
from functools import partial

import torch
from torchtyping import TensorType as TT
from torch import Tensor
from jaxtyping import Float, Int
from typing import Tuple
import einops

from fancy_einsum import einsum

import plotly.graph_objs as go
import torch
import ipywidgets as widgets
from IPython.display import display
from transformer_lens import ActivationCache, HookedTransformer

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"


# =============== VISUALIZATION UTILS ===============
def visualize_tensor(tensor, labels, zmin=-1.0, zmax=1.0):
    """Visualizes a 3D tensor as a series of heatmaps.

    Args:
        tensor (torch.Tensor): Tensor to visualize.
        labels (List[str]): List of labels for each slice in the tensor.
        zmin (float, optional): Minimum value for the color scale. Defaults to -1.0.
        zmax (float, optional): Maximum value for the color scale. Defaults to 1.0.

    Raises:
        AssertionError: If the number of labels does not match the number of slices in the tensor.
    """
    assert (
        len(labels) == tensor.shape[-1]
    ), "The number of labels should match the number of slices in the tensor."

    def plot_slice(selected_slice):
        """Plots a single slice of the tensor."""
        fig = go.FigureWidget(
            data=go.Heatmap(
                z=tensor[:, :, selected_slice].numpy(),
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu",
            ),
            layout=go.Layout(
                title=f"Slice: {selected_slice} - Step: {labels[selected_slice]}",
                yaxis=dict(autorange="reversed"),
            ),
        )
        return fig

    def on_slider_change(change):
        """Updates the plot when the slider is moved."""
        selected_slice = change["new"]
        fig = plot_slice(selected_slice)
        output.clear_output(wait=True)
        with output:
            display(fig)

    slider = widgets.IntSlider(
        min=0, max=tensor.shape[2] - 1, step=1, value=0, description="Slice:"
    )
    slider.observe(on_slider_change, names="value")
    display(slider)

    output = widgets.Output()
    display(output)

    with output:
        display(plot_slice(0))


# =============== METRIC UTILS ===============

def get_logit_diff(
    logits: Float[Tensor, "batch pos vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"], 
    per_prompt: bool = False,
    per_completion: bool = False,
):
    """
    Gets the difference between the logits of the provided tokens 
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    logits = einops.repeat(
        logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    left_logits: Float[Tensor, "batch n_pairs"] = logits.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    )
    right_logits: Float[Tensor, "batch n_pairs"] = logits.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    )
    if per_completion:
        print(left_logits - right_logits)
    left_logits: Float[Tensor, "batch"] = left_logits.mean(dim=1)
    right_logits: Float[Tensor, "batch"] = right_logits.mean(dim=1)
    if per_prompt:
        return left_logits - right_logits

    return (left_logits - right_logits).mean()

def get_log_probs(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs"],
    per_prompt: bool = False,
) -> Float[Tensor, "batch n_pairs"]:
    
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    assert len(answer_tokens.shape) == 2
    
    # convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # get the log probs for the answer tokens
    log_probs = einops.repeat(
        log_probs, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    answer_log_probs: Float[Tensor, "batch n_pairs"] = log_probs.gather(
        -1, answer_tokens.unsqueeze(-1)
    )
    # average over the answer tokens
    answer_log_probs: Float[Tensor, "batch"] = answer_log_probs.mean(dim=1)
    if per_prompt:
        return answer_log_probs
    else:
        return answer_log_probs.mean()


def log_prob_diff_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs"],
    flipped_log_prob: float,
    clean_log_prob: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()
    
def log_prob_diff_denoising(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch n_pairs 2"],
        flipped_log_prob: float,
        clean_log_prob: float,
        return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()

def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"],
    flipped_logit_diff: float,
    clean_logit_diff: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    ld = ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff))
    if return_tensor:
        return ld
    else:
        return ld.item()


def logit_diff_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float,
        corrupted_logit_diff: float,
        answer_tokens: Float[Tensor, "batch n_pairs 2"],
        return_tensor: bool = False,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = get_logit_diff(logits, answer_tokens)
        ld = ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

        if return_tensor:
            return ld
        else:
            return ld.item()


# =============== LOGIT LENS UTILS ===============


def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    answer_tokens: Int[Tensor, "batch pair correct"], 
    cache: ActivationCache, 
    model: HookedTransformer,
    pos: int = -1,
    biased: bool = False,
):
    scaled_residual_stack: Float[Tensor, "... batch d_model"] = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=pos)
    answer_residual_directions: Float[Tensor, "batch pair correct d_model"] = model.tokens_to_residual_directions(answer_tokens)
    answer_residual_directions = answer_residual_directions.mean(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    batch_logit_diffs: Float[Tensor, "... batch"] = einops.einsum(
        scaled_residual_stack, 
        logit_diff_directions, 
        "... batch d_model, batch d_model -> ... batch",
    )
    if not biased:
        diff_from_unembedding_bias: Float[Tensor, "batch"] = (
            model.b_U[answer_tokens[:, :, 0]] - 
            model.b_U[answer_tokens[:, :, 1]]
        ).mean(dim=1)
        batch_logit_diffs += diff_from_unembedding_bias
    return einops.reduce(batch_logit_diffs, "... batch -> ...", 'mean')


def cache_to_logit_diff(
    cache: ActivationCache,
    answer_tokens: Int[Tensor, "batch pair correct"], 
    pos: int = -1,
):
    final_residual_stream: Float[Tensor, "batch pos d_model"] = cache["resid_post", -1]
    token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, pos, :]
    return residual_stack_to_logit_diff(
        token_residual_stream, 
        answer_tokens=answer_tokens, 
        cache=cache, 
        pos=pos,
    )
    

# =============== PATCHING & KNOCKOUT UTILS ===============
def patch_pos_head_vector(
    orig_head_vector: TT["batch", "pos", "head_index", "d_head"],
    hook,
    pos,
    head_index,
    patch_cache,
):
    """Patches a head vector at a given position and head index.

    Args:
        orig_head_vector (TT["batch", "pos", "head_index", "d_head"]): Original head activation vector.
        hook (Hook): Hook to patch.
        pos (int): Position to patch.
        head_index (int): Head index to patch.
        patch_cache (Dict[str, torch.Tensor]): Patch cache.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: Patched head vector.
    """
    orig_head_vector[:, pos, head_index, :] = patch_cache[hook.name][
        :, pos, head_index, :
    ]
    return orig_head_vector


def patch_head_vector(
    orig_head_vector: TT["batch", "pos", "head_index", "d_head"],
    hook,
    head_index,
    patch_cache,
):
    """Patches a head vector at a given head index.

    Args:
        orig_head_vector (TT["batch", "pos", "head_index", "d_head"]): Original head activation vector.
        hook (Hook): Hook to patch.
        head_index (int): Head index to patch.
        patch_cache (Dict[str, torch.Tensor]): Patch cache.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: Patched head vector.
    """
    orig_head_vector[:, :, head_index, :] = patch_cache[hook.name][:, :, head_index, :]
    return orig_head_vector


def ablate_top_head_hook(
    z: TT["batch", "pos", "head_index", "d_head"], hook, head_idx=0
):
    """Hook to ablate the top head of a given layer.

    Args:
        z (TT["batch", "pos", "head_index", "d_head"]): Attention weights.
        hook ([type]): Hook.
        head_idx (int, optional): Head index to ablate. Defaults to 0.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: Attention weights.
    """
    z[:, -1, head_idx, :] = 0
    return z


def get_knockout_perf_drop(model, heads_to_ablate, clean_tokens, metric):
    """Gets the performance drop for a given model and heads to ablate.

    Args:
        model (nn.Module): Model to knockout.
        heads_to_ablate (List[Tuple[int, int]]): List of tuples of layer and head indices to knockout.
        clean_tokens (Tensor): Clean tokens.
        answer_token_indices (Tensor): Answer token indices.

    Returns:
        Tensor: Performance drop.
    """
    # Adds a hook into global model state
    for layer, head in heads_to_ablate:
        ablate_head_hook = partial(ablate_top_head_hook, head_idx=head)
        model.blocks[layer].attn.hook_z.add_hook(ablate_head_hook)

    ablated_logits, ablated_cache = model.run_with_cache(clean_tokens)
    ablated_logit_diff = metric(ablated_logits)

    return ablated_logit_diff


