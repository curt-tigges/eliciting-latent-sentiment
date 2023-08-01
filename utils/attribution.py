from jaxtyping import Float
from typing import Optional, Union, List, Literal, Tuple
import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
import einops
from circuitsvis.attention import attention_heads, attention_patterns


def concat_lists(*lists):
    return [x for l in lists for x in l]


def diagonalise(tensor: Tensor, dim: int):
    '''
    Returns a tensor with dimension `dim` duplicated, diagonal elements along these two dims
    are given by `tensor`, and off-diags are zero.
    '''
    shape = tensor.shape
    new_shape = shape[:dim] + (shape[dim],) + shape[dim:]
    new_tensor = t.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    indices = concat_lists(
        [slice(None)] * dim,
        [range(shape[dim]), range(shape[dim])],
        [slice(None)] * (len(shape) - dim - 1)
    )
    new_tensor[indices] = tensor
    return new_tensor


def from_cache(
    cache: ActivationCache,
    model: HookedTransformer,
    tokens: Union[List[str], Float[Tensor, "seq"]],
    resid_directions: Optional[Union[Float[Tensor, "*seq d_model"], Float[Tensor, "*seq d_vocab"]]] = None,
    seq_pos: Optional[Union[int, List[int]]] = None,
    heads: Optional[List[Tuple[int, int]]] = None,
    mode: Literal["large", "small"] = "small",
    include_b_U_attribution: bool = False, # This always points towards more common tokens I guess, so it does a lot of the heavy lifting for us (when we aren't using a baseline). Interesting!
):
    '''
    resid_directions
        if not None, then these are the directions we do attribution with (unless it has length d_vocab, in which case this is interpreted as something like logit_diff, i.e. it might have 1s and -1s in the elements).
    tokens
        if not None, then we use these unembeddings as our resid_directions
    seq_pos
        if not None, then we only do attribution for this position (set all other
        positions to be zero)
    '''
    t.cuda.empty_cache()
    assert not(cache.has_batch_dim), "Only supports batch dim = 1 (otherwise things get too big and messy!)"
    seq_len = len(tokens)

    if seq_pos is None:
        seq_pos = list(range(seq_len))
    elif isinstance(seq_pos, int):
        seq_pos = [seq_pos]
    assert isinstance(seq_pos, list) and all(isinstance(i, int) for i in seq_pos), "seq_pos must be None, int, or list of ints"

    # ! Get MLP & other decomps
    if 'hook_pos_embed' in cache.keys():
        embed_results: Float[Tensor, "2 seqQ d_model"] = t.stack([
            cache["embed"], cache["pos_embed"]
        ]) / cache["scale"]
        embed_results: Float[Tensor, "2 seqQ seqK d_model"] = diagonalise(embed_results, dim=1)[:, seq_pos]
    mlp_results: Float[Tensor, "layer seqQ d_model"] = t.stack([
        cache["mlp_out", layer] for layer in range(model.cfg.n_layers)
    ]) / cache["scale"]
    mlp_results: Float[Tensor, "layer seqQ seqK d_model"] = diagonalise(mlp_results, dim=1)[:, seq_pos]
    mlp_labels = [f"MLP{i}" for i in range(model.cfg.n_layers)]

    # ! Get attention biases
    attn_biases: Float[Tensor, "layer seqQ seqK d_model"] = diagonalise(
        einops.repeat(model.b_O, "layer d_model -> layer seqQ d_model", seqQ=seq_len), dim=1
    )[:, seq_pos] / cache["scale"][seq_pos]
    attn_bias_labels = [f"Attn bias [{i}]" for i in range(model.cfg.n_layers)]

    # ! Get attention decomposition (this is harder because we have to decompose by source position)
    attn_results = []
    attn_labels = []
    # TODO - could save memory if I didn't have things with `d_model` dimension much; I multiply along this straight away not @ end
    for layer in range(model.cfg.n_layers):
        pattern: Float[Tensor, "nheads seqQ seqK"] = cache["pattern", layer][:, seq_pos]
        v: Float[Tensor, "seqK nheads d_head"] = cache["v", layer]
        v_post: Float[Tensor, "seqK nheads d_model"] = einops.einsum(
            v, model.W_O[layer],
            "seqK nheads d_head, nheads d_head d_model -> nheads seqK d_model",
        )
        results_pre: Float[Tensor, "nheads seqQ seqK d_model"] = einops.einsum(
            v_post, pattern,
            "nheads seqK d_model, nheads seqQ seqK -> nheads seqQ seqK d_model",
        )
        # Apply final layernorm (needs to be by query position, not by key position)
        results_pre /= einops.repeat(
            cache["scale"][seq_pos],
            "seqQ d_model -> seqQ seqK d_model", seqK=seq_len
        )
        attn_results.append(results_pre)
        attn_labels.extend([f"{layer}.{head}" for head in range(model.cfg.n_heads)])

    

    if 'hook_pos_embed' in cache.keys():
        labels = ["embed", "pos_embed"] + mlp_labels + attn_bias_labels + attn_labels
        full_decomp: Float[Tensor, "component seqQ seqK d_model"] = t.cat([
            embed_results, mlp_results, attn_biases, t.cat(attn_results)
        ])
    else:
        labels = mlp_labels + attn_bias_labels + attn_labels
        full_decomp: Float[Tensor, "component seqQ seqK d_model"] = t.cat([
            mlp_results, attn_biases, t.cat(attn_results)
        ])

    # Get the residual stream directions
    if isinstance(tokens[0], str):
        token_ids = model.to_tokens(tokens, prepend_bos=False).squeeze()
        token_str = tokens
    else:
        token_ids = tokens
        token_str = model.to_str_tokens(tokens)
    
    if resid_directions is None:
        assert seq_pos == list(range(seq_len)), "If resid_directions is None, then seq_pos must be None (we're doing attribution per token, in the direction of the correct token's unembedding)."
        seq_pos = list(range(seq_len - 1))
        resid_directions: Float[Tensor, "seqQ d_model"] = model.W_U.T[token_ids[1:]]
        full_decomp: Float[Tensor, "component seqQ seqK d_model"] = full_decomp[:, :-1, :-1]
        b_U_attribution: Float[Tensor, "seqQ"] = model.b_U[token_ids[1:]]
        b_U_attribution: Float[Tensor, "1 seqQ seqK"] = t.diag(b_U_attribution)[seq_pos].unsqueeze(0)
        token_str = token_str[:-1]
    else:
        if resid_directions.ndim == 1:
            resid_directions = einops.repeat(resid_directions, "d -> seqQ d", seqQ=len(seq_pos))
        assert (resid_directions.ndim == 2) and (resid_directions.size(0) <= seq_len)

        if resid_directions.size(1) == model.cfg.d_model:
            b_U_attribution = None
        elif resid_directions.size(1) == model.cfg.d_vocab:
            # In this case, we need bias attribution, and we need to redefine resid directions
            b_U_attribution: Float[Tensor, "1 seqQ seqK"] = t.zeros(1, len(seq_pos), seq_len, device=model.b_U.device)
            b_U_attribution[:, :, seq_pos] = einops.einsum(
                model.b_U, resid_directions,
                "d_vocab, seqQ d_vocab -> seqQ"
            )
            resid_directions = einops.einsum(
                model.W_U, resid_directions,
                "d_model d_vocab, seq d_vocab -> seq d_model"
            )
        else:
            raise ValueError(f"resid_directions must have shape (*seq_len, d_model) or (*seq_len, d_vocab), but the last dimension doesn't match. Shape is {resid_directions.shape}")

    print(full_decomp.shape, resid_directions.shape)
    full_attribution = einops.einsum(
        full_decomp, resid_directions,
        "component seqQ seqK d_model, seqQ d_model -> component seqQ seqK"
    )
    print(full_attribution.shape)
    if (b_U_attribution is not None) and include_b_U_attribution:
        full_attribution = t.concat([full_attribution, b_U_attribution])
        labels.append("b_U")

    # get number of MLPs, attn biases, and attn heads
    n_mlp = len(mlp_labels)
    n_attn_bias = len(attn_bias_labels)
    n_attn_heads = len(attn_labels)
    n_embed = 2 if 'hook_pos_embed' in cache.keys() else 0

    full_attribution_max = einops.reduce(full_attribution[(n_embed+n_mlp+n_attn_bias):].abs(), "c sQ sK -> 1 1 1", "max") # or 1 sQ 1
    print(f"Max attribution: {full_attribution_max}")
    full_attribution_scaled_positive = (full_attribution * (full_attribution > 0).float()) / full_attribution_max
    full_attribution_scaled_negative = (-full_attribution * (full_attribution < 0).float()) / full_attribution_max

    # Now finally, we (annoyingly) need to pad back to the original length
    components, seqQ, seqK = full_attribution.shape
    full_attribution_padded = t.zeros((components, seqK, seqK), device=full_attribution.device) #.fill_(float("-inf"))
    full_attribution_padded_scaled_positive = full_attribution_padded.clone()
    full_attribution_padded_scaled_negative = full_attribution_padded.clone()
    # error point
    # ----------------
    full_attribution_padded_scaled_positive[:, (seq_pos if len(seq_pos) > 1 else slice(None))] = full_attribution_scaled_positive
    full_attribution_padded_scaled_negative[:, (seq_pos if len(seq_pos) > 1 else slice(None))] = full_attribution_scaled_negative

    # Include only specific heads if we want
    if heads is not None:
        

        # get the number of heads per layer
        heads_per_layer = n_attn_heads // model.cfg.n_layers

        # get the layer and head indices
        layer_indices = [head[0] for head in heads]
        head_indices = [head[1] for head in heads]

        # get the indices of the heads in the full attribution matrix
        head_indices = [layer * heads_per_layer + head + n_mlp + n_attn_bias + n_embed for layer, head in heads]
        #head_indices = [layer * heads_per_layer + head for layer, head in heads]
        print(labels)

        # slice the full attribution matrix to get only the heads we want
        print(f"Initial shape: {full_attribution_padded_scaled_positive.shape}")
        full_attribution_padded_scaled_positive = full_attribution_padded_scaled_positive[head_indices, :, :]
        full_attribution_padded_scaled_negative = full_attribution_padded_scaled_negative[head_indices, :, :]

        print(f"Final shape: {full_attribution_padded_scaled_positive.shape}")
        # get the labels for the heads we want
        labels = [labels[i] for i in head_indices]


    if mode == "small":
        html_pos = attention_patterns(
            attention = full_attribution_padded_scaled_positive,
            tokens = token_str,
            attention_head_names = labels,
        )
        html_neg = attention_patterns(
            attention = full_attribution_padded_scaled_negative,
            tokens = token_str,
            attention_head_names = labels,
        )
        return html_pos, html_neg
    else:
        html_all = attention_heads(
            attention = full_attribution_padded_scaled_positive - full_attribution_padded_scaled_negative,
            tokens = token_str,
            attention_head_names = labels,
        )
        return html_all