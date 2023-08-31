import os
from functools import partial

import torch
from torchtyping import TensorType as TT
from torch import Tensor
from jaxtyping import Float, Int
from typeguard import typechecked
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
@typechecked
def get_logit_diff(
    logits: Float[Tensor, "batch *pos vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"], 
    per_prompt: bool = False,
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
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        final_token_logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    else:
        final_token_logits = logits
    repeated_logits = einops.repeat(
        final_token_logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    left_logits: Float[Tensor, "batch n_pairs"] = repeated_logits.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    )
    right_logits: Float[Tensor, "batch n_pairs"] = repeated_logits.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    )
    left_logits_batch: Float[Tensor, "batch"] = left_logits.mean(dim=1)
    right_logits_batch: Float[Tensor, "batch"] = right_logits.mean(dim=1)
    if per_prompt:
        return left_logits_batch - right_logits_batch

    return (left_logits_batch - right_logits_batch).mean()


def get_prob_diff(
    logits: Float[Tensor, "batch pos vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"], 
    per_prompt: bool = False,
):
    """
    Gets the difference between the softmax probabilities of the provided tokens 
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    logits = einops.repeat(
        logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    probs: Float[Tensor, "batch n_pairs vocab"] = logits.softmax(dim=-1)
    left_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    )
    right_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    )
    left_probs: Float[Tensor, "batch"] = left_probs.mean(dim=1)
    right_probs: Float[Tensor, "batch"] = right_probs.mean(dim=1)
    if per_prompt:
        return left_probs - right_probs

    return (left_probs - right_probs).mean()


def get_log_probs(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    per_prompt: bool = False,
) -> Float[Tensor, "batch n_pairs"]:
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    assert len(answer_tokens.shape) == 2
    
    # convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # get the log probs for the answer tokens
    log_probs_repeated = einops.repeat(
        log_probs, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    answer_log_probs: Float[Tensor, "batch n_pairs"] = log_probs_repeated.gather(
        -1, answer_tokens.unsqueeze(-1)
    )
    # average over the answer tokens
    answer_log_probs_batch: Float[Tensor, "batch"] = answer_log_probs.mean(dim=1)
    if per_prompt:
        return answer_log_probs_batch
    else:
        return answer_log_probs_batch.mean()


def log_prob_diff_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    Linear function of log prob, calibrated so that it equals 0 when performance is
    same as on clean input, and 1 when performance is same as on flipped input.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - clean_value) / (flipped_value  - clean_value))
    if return_tensor:
        return ld
    else:
        return ld.item()
    

def log_prob_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    Linear function of log prob, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_value) / (clean_value  - flipped_value))
    if return_tensor:
        return ld
    else:
        return ld.item()


def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    ld = ((patched_logit_diff - flipped_value) / (clean_value  - flipped_value))
    if return_tensor:
        return ld
    else:
        return ld.item()


def prob_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
) -> float:
    '''
    Linear function of prob diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_prob_diff(logits, answer_tokens)
    ld = ((patched_logit_diff - flipped_value) / (clean_value  - flipped_value)).item()
    if return_tensor:
        return ld
    else:
        return ld.item()
    

def logit_flip_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    Moves in discrete jumps based on whether logit diffs are closer to clean or corrupted.
    '''
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_logit_diff(logits, answer_tokens, per_prompt=True)
    clean_distances = (clean_value - patched_logit_diff).abs()
    corrupt_distances = (flipped_value - patched_logit_diff).abs()
    lf = (clean_distances < corrupt_distances).float().mean()
    if return_tensor:
        return lf
    else:
        return lf.item()


def logit_diff_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch *n_pairs 2"],
    flipped_value: float,
    clean_value: float,
    return_tensor: bool = False,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    ld = ((patched_logit_diff - clean_value) / (clean_value - flipped_value))

    if return_tensor:
        return ld
    else:
        return ld.item()
    



# =============== LOGIT LENS UTILS ===============


def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache, 
    answer_tokens: Int[Tensor, "batch pair correct"], 
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
    model: HookedTransformer,
    pos: int = -1,
):
    final_residual_stream: Float[Tensor, "batch pos d_model"] = cache["resid_post", -1]
    token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, pos, :]
    return residual_stack_to_logit_diff(
        token_residual_stream, 
        answer_tokens=answer_tokens, 
        model=model,
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


def create_cache_for_dir_patching(
    clean_cache: ActivationCache, 
    corrupted_cache: ActivationCache, 
    sentiment_dir: Float[Tensor, "d_model"],
    model: HookedTransformer,
) -> ActivationCache:
    '''
    We patch the sentiment direction from corrupt to clean
    '''
    cache_dict = dict()
    for act_name, clean_value in clean_cache.items():
        is_result = act_name.endswith('result')
        is_resid = (
            act_name.endswith('resid_pre') or
            act_name.endswith('resid_post') or
            act_name.endswith('attn_out') or
            act_name.endswith('mlp_out')
        )
        if is_resid:
            clean_value = clean_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s d, d -> b s'
            )
            clean_proj = einops.einsum(
                clean_value, sentiment_dir, 'b s d, d -> b s'
            )
            sentiment_dir_broadcast = einops.repeat(
                sentiment_dir, 'd -> b s d', 
                b=corrupt_value.shape[0], 
                s=corrupt_value.shape[1], 
            )
            proj_diff = einops.repeat(
                clean_proj - corrupt_proj, 
                'b s -> b s d', 
                d=corrupt_value.shape[-1]
            )
            sentiment_adjustment = proj_diff * sentiment_dir_broadcast
            cache_dict[act_name] = (
                corrupt_value + sentiment_adjustment
            )
        elif is_result:
            clean_value = clean_value.to(device)
            corrupt_value = corrupted_cache[act_name].to(device)
            corrupt_proj = einops.einsum(
                corrupt_value, sentiment_dir, 'b s h d, d -> b s h'
            )
            clean_proj = einops.einsum(
                clean_value, sentiment_dir, 'b s h d, d -> b s h'
            )
            sentiment_dir_broadcast = einops.repeat(
                sentiment_dir, 'd -> b s h d', 
                b=corrupt_value.shape[0], 
                s=corrupt_value.shape[1], 
                h=corrupt_value.shape[2]
            )
            proj_diff = einops.repeat(
                clean_proj - corrupt_proj, 
                'b s h -> b s h d', 
                d=corrupt_value.shape[3]
            )
            sentiment_adjustment = proj_diff * sentiment_dir_broadcast
            cache_dict[act_name] = (
                corrupt_value + sentiment_adjustment
            )
        else:
            cache_dict[act_name] = clean_value

    return ActivationCache(cache_dict, model)