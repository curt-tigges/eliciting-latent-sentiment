import os
from functools import partial

import torch
from torchtyping import TensorType as TT
from torch import Tensor
from jaxtyping import Float
from typing import Tuple
import einops

import transformer_lens.patching as patching
from fancy_einsum import einsum

import plotly.graph_objs as go
import torch
import ipywidgets as widgets
from IPython.display import display


if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"

# =============== DATA UTILS ===============
def set_up_data(model, prompts, answers):
    """Sets up data for a given model, prompts, and answers.

    Args:
        model (HookedTransformer): Model to set up data for.
        prompts (List[str]): List of prompts to use.
        answers (List[List[str]]): List of answers to use.

    Returns:
        Tuple[List[str], List[str], torch.Tensor]: Clean tokens, corrupted tokens, and answer token indices.
    """
    clean_tokens = model.to_tokens(prompts)
    # Swap each adjacent pair of tokens
    corrupted_tokens = clean_tokens[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]

    answer_token_indices = torch.tensor(
        [
            [model.to_single_token(answers[i][j]) for j in range(2)]
            for i in range(len(answers))
        ],
        device=model.cfg.device,
    )

    return clean_tokens, corrupted_tokens, answer_token_indices


def read_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    prompts_str, answers_str = content.split("\n\n")
    prompts = prompts_str.split("\n")  # Remove the last empty item
    answers = [
        tuple(answer.split(",")) for answer in answers_str.split(";")[:-1]
    ]  # Remove the last empty item

    return prompts, answers


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
def get_logit_diff(logits, answer_token_indices, per_prompt=False):
    """Gets the difference between the logits of the provided tokens (e.g., the correct and incorrect tokens in IOI)

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if per_prompt:
        print(correct_logits - incorrect_logits)

    return (correct_logits - incorrect_logits).mean()


def get_logit_diff_multi(
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
        print(left_logits - right_logits)

    return (left_logits - right_logits).mean()


# =============== LOGIT LENS UTILS ===============


def residual_stack_to_logit_diff(residual_stack, logit_diff_directions, prompts, cache):
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / len(prompts)


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


def path_patching(
    model,
    patch_tokens,
    orig_tokens,
    sender_heads,
    receiver_hooks,
    sender_positions=-1,
    receiver_positions=-1,
):
    """Patches a model using the provided patch tokens.

    Args:
        model (nn.Module): Model to patch.
        patch_tokens (Tokens): Patch tokens.
        orig_tokens (Tokens): Original tokens.
        sender_heads (List[Tuple[int, int]]): List of tuples of layer and head indices to patch.
        receiver_hooks (List[Tuple[str, int]]): List of tuples of hook names and head indices to patch.
        positions (int, optional): Positions to patch. Defaults to -1.

    Returns:
        nn.Module: Patched model.
    """

    def patch_positions(z, source_act, hook, positions=["end"], verbose=False):
        for pos in positions:
            z[torch.arange(orig_tokens.N), orig_tokens.word_idx[pos]] = source_act[
                torch.arange(patch_tokens.N), patch_tokens.word_idx[pos]
            ]
        return z

    # process arguments
    sender_hooks = []
    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_z", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]
    receiver_hook_heads = [x[1] for x in receiver_hooks]
    # Forward pass A (in https://arxiv.org/pdf/2211.00593.pdf)
    source_logits, sender_cache = model.run_with_cache(patch_tokens)

    # Forward pass B
    target_logits, target_cache = model.run_with_cache(orig_tokens)

    # Forward pass C
    # Cache the receiver hooks
    # (adding these hooks first means we save values BEFORE they are overwritten)
    receiver_cache = model.add_caching_hooks(lambda x: x in receiver_hook_names)

    # "Freeze" intermediate heads to their orig_tokens values
    # q, k, and v will get frozen, and then if it's a sender head, this will get undone
    # z, attn_out, and the MLP will all be recomputed and added to the residual stream
    # however, the effect of the change on the residual stream will be overwritten by the
    # freezing for all non-receiver components
    pass_c_hooks = []
    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)
                if (hook_name, head_idx) not in receiver_hooks:
                    # print(f"Freezing {hook_name}")
                    hook = partial(
                        patch_head_vector, head_index=head_idx, patch_cache=target_cache
                    )
                    pass_c_hooks.append((hook_name, hook))
                else:
                    pass
                    # print(f"Not freezing {hook_name}")

    # These hooks will overwrite the freezing, for the sender heads
    # We also carry out pass C
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
            hook_name,
            head_idx,
        )
        hook = partial(
            patch_pos_head_vector,
            pos=sender_positions,
            head_index=head_idx,
            patch_cache=sender_cache,
        )
        pass_c_hooks.append((hook_name, hook))

    receiver_logits = model.run_with_hooks(orig_tokens, fwd_hooks=pass_c_hooks)
    # Add (or return) all the hooks needed for forward pass D
    pass_d_hooks = []

    for hook_name, head_idx in receiver_hooks:
        # for pos in positions:
        # if torch.allclose(
        #     receiver_cache[hook_name][torch.arange(orig_tokens.N), orig_tokens.word_idx[pos]],
        #     target_cache[hook_name][torch.arange(orig_tokens.N), orig_tokens.word_idx[pos]],
        # ):
        #     warnings.warn("Torch all close for {}".format(hook_name))
        hook = partial(
            patch_pos_head_vector,
            pos=receiver_positions,
            head_index=head_idx,
            patch_cache=receiver_cache,
        )
        pass_d_hooks.append((hook_name, hook))

    return pass_d_hooks


def get_path_patching_results(
    model,
    clean_tokens,
    patch_tokens,
    metric,
    step_metric,
    receiver_heads,
    receiver_type="hook_q",
    sender_heads=None,
    position=-1,
):
    """Gets the path patching results for a given model.

    Args:
        model (nn.Module): Model to patch.
        step_logit_diff (Tensor): Logit difference for the particular step/revision.
        receiver_heads (List[Tuple[int, int]]): List of tuples of layer and head indices to patch.
        receiver_type (str, optional): Type of receiver. Defaults to "hook_q".
        sender_heads (List[Tuple[int, int]], optional): List of tuples of layer and head indices to patch. Defaults to None.
        position (int, optional): Positions to patch. Defaults to -1.

    Returns:
        Tensor: Path patching results.
    """
    metric_delta_results = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device="cuda:0"
    )

    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            pass_d_hooks = path_patching(
                model=model,
                patch_tokens=patch_tokens,
                orig_tokens=clean_tokens,
                sender_heads=[(layer, head_idx)],
                receiver_hooks=[
                    (f"blocks.{layer_idx}.attn.{receiver_type}", head_idx)
                    for layer_idx, head_idx in receiver_heads
                ],
                positions=position,
            )
            path_patched_logits = model.run_with_hooks(
                clean_tokens, fwd_hooks=pass_d_hooks
            )
            patched_metric = metric(path_patched_logits)
            metric_delta_results[layer, head_idx] = (
                -(step_metric - patched_metric) / step_metric
            )
    return metric_delta_results


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


# =========================== CIRCUITS OVER TIME ===========================
def get_chronological_circuit_performance(
    model_hf_name,
    model_tl_name,
    cache_dir,
    ckpts,
    clean_tokens,
    corrupted_tokens,
    answer_token_indices,
):
    """Gets the performance of a model over time.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        clean_tokens (Tensor): Clean tokens.
        corrupted_tokens (Tensor): Corrupted tokens.
        answer_token_indices (Tensor): Answer token indices.

    Returns:
        dict: Dictionary of performance over time.
    """
    logit_diff_vals = []
    clean_ld_baselines = []
    corrupted_ld_baselines = []

    metric = partial(get_logit_diff, answer_token_indices=answer_token_indices)

    previous_model = None

    for ckpt in ckpts:

        # Get model
        if previous_model is not None:
            clear_gpu_memory(previous_model)

        print(f"Loading model for step {ckpt}...")
        model = load_model(model_hf_name, model_tl_name, f"step{ckpt}", cache_dir)

        # Get metric values
        print("Getting metric values...")
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = metric(clean_logits).item()
        corrupted_logit_diff = metric(corrupted_logits).item()

        clean_ld_baselines.append(clean_logit_diff)
        corrupted_ld_baselines.append(corrupted_logit_diff)
        print(f"Logit diff: {clean_logit_diff}")
        logit_diff_vals.append(clean_logit_diff)

        previous_model = model

    return {
        "logit_diffs": torch.tensor(logit_diff_vals),
        "clean_baselines": torch.tensor(clean_ld_baselines),
        "corrupted_baselines": torch.tensor(corrupted_ld_baselines),
    }


def get_chronological_circuit_data(
    model_hf_name,
    model_tl_name,
    cache_dir,
    ckpts,
    circuit,
    clean_tokens,
    corrupted_tokens,
    answer_token_indices,
):
    """Extracts data from different circuit components over time.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        circuit (dict): Circuit dictionary.
        clean_tokens (Tensor): Clean tokens.
        corrupted_tokens (Tensor): Corrupted tokens.
        answer_token_indices (Tensor): Answer token indices.

    Returns:
        dict: Dictionary of data over time.
    """
    logit_diff_vals = []
    clean_ld_baselines = []
    corrupted_ld_baselines = []
    attn_head_vals = []
    value_patch_vals = []
    circuit_vals = {key: [] for key in circuit.keys()}
    knockout_drops = {key: [] for key in circuit.keys()}

    metric = partial(get_logit_diff, answer_token_indices=answer_token_indices)

    previous_model = None

    for ckpt in ckpts:

        # Get model
        if previous_model is not None:
            clear_gpu_memory(previous_model)

        print(f"Loading model for step {ckpt}...")
        model = load_model(model_hf_name, model_tl_name, f"step{ckpt}", cache_dir)

        # Get metric values (relative to final performance)
        print("Getting metric values...")
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = metric(clean_logits).item()
        corrupted_logit_diff = metric(corrupted_logits).item()

        clean_ld_baselines.append(clean_logit_diff)
        corrupted_ld_baselines.append(corrupted_logit_diff)

        logit_diff_vals.append(clean_logit_diff)

        # Get attention pattern patching metrics
        print("Getting attention pattern patching metrics...")
        attn_head_out_all_pos_act_patch_results = (
            patching.get_act_patch_attn_head_pattern_all_pos(
                model, corrupted_tokens, clean_cache, metric
            )
        )
        attn_head_vals.append(attn_head_out_all_pos_act_patch_results)

        # Get value patching metrics
        print("Getting value patching metrics...")
        value_patch_results = patching.get_act_patch_attn_head_v_all_pos(
            model, corrupted_tokens, clean_cache, metric
        )
        value_patch_vals.append(value_patch_results)

        # Get path patching metrics for specific circuit parts
        for key in circuit.keys():
            # Get path patching results
            print(f"Getting path patching metrics for {key}...")
            path_patching_results = get_path_patching_results(
                model,
                clean_tokens,
                corrupted_tokens,
                metric,
                clean_logit_diff,
                circuit[key].heads,
                receiver_type=circuit[key].receiver_type,
                position=circuit[key].position,
            )
            circuit_vals[key].append(path_patching_results)

            # Get knockout performance drop
            print(f"Getting knockout performance drop for {key}...")
            knockout_drops[key].append(
                get_knockout_perf_drop(model, circuit[key].heads, clean_tokens, metric)
            )

        previous_model = model

    return {
        "logit_diffs": torch.tensor(logit_diff_vals),
        "clean_baselines": torch.tensor(clean_ld_baselines),
        "corrupted_baselines": torch.tensor(corrupted_ld_baselines),
        "attn_head_vals": torch.stack(attn_head_vals, dim=-1),
        "value_patch_vals": torch.stack(value_patch_vals, dim=-1),
        "circuit_vals": circuit_vals,
        "knockout_drops": knockout_drops,
    }
