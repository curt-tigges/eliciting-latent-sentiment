from transformer_lens import ActivationCache
from jaxtyping import Float, Bool
from torch import Tensor
import einops

def residual_sentiment_sim_by_head(
    cache: ActivationCache,
    sentiment_directions: Float[Tensor, "batch d_model"],
    centre_residuals: bool = True,
    normalise_residuals: bool = False,
    layers: int = 12,
    heads: int = 12,
) -> Float[Tensor, "layer head"]:
    batch_size = sentiment_directions.shape[0]
    residual_stack: Float[
        Tensor, "components batch d_model"
    ] = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=False)
    residual_stack: Float[
        Tensor, "components batch d_model"
    ] = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    if centre_residuals:
        residual_stack -= einops.reduce(
            residual_stack, 
            "components batch d_model -> components 1 d_model", 
            "mean"
        )
    if normalise_residuals:
        residual_stack = (
            residual_stack.T /
            residual_stack.norm(dim=-1).T
        ).T
        sentiment_directions = (
            sentiment_directions.T /
            sentiment_directions.norm(dim=-1).T
        ).T
    component_means: Float[Tensor, "components"] = einops.einsum(
        residual_stack, sentiment_directions, 
        "components batch d_model, batch d_model -> components"
    ) / batch_size
    return einops.rearrange(
        component_means, 
        "(layer head) -> layer head", 
        layer=layers, 
        head=heads,
    )

def residual_sentiment_sim_by_pos(
    cache: ActivationCache,
    sentiment_directions: Float[Tensor, "batch d_model"],
    seq_len: int,
    centre_residuals: bool = True,
    normalise_residuals: bool = False,
) -> Float[Tensor, "components"]:
    batch_size = sentiment_directions.shape[0]
    residual_stack: Float[
        Tensor, "components batch pos d_model"
    ] = cache.stack_head_results(layer=-1, return_labels=False)
    for pos in range(seq_len):
        residual_stack[:, :, pos, :] = cache.apply_ln_to_stack(
            residual_stack[:, :, pos, :], layer=-1, pos_slice=pos
        )
    if centre_residuals:
        residual_stack -= einops.reduce(
            residual_stack, 
            "components batch pos d_model -> components 1 pos d_model", 
            "mean"
        )
    if normalise_residuals: # for cosine similarity
        residual_stack = (
            residual_stack.T /
            residual_stack.norm(dim=-1).T
        ).T
        sentiment_directions = (
            sentiment_directions.T /
            sentiment_directions.norm(dim=-1).T
        ).T
    component_means: Float[Tensor, "components"] = einops.einsum(
        residual_stack, sentiment_directions, 
        "components batch pos d_model, batch d_model -> components pos"
    ) / batch_size
    return component_means


def residual_flip_dir_by_pos(
    cache: ActivationCache,
    is_positive: Bool[Tensor, "batch"],
    seq_len: int,
) -> Float[Tensor, "components pos d_model"]:
    # we consider corrupt -> clean flips
    stack: Float[
        Tensor, "components batch pos d_model"
    ] = cache.stack_head_results(layer=-1, return_labels=False)
    for pos in range(seq_len):
        stack[:, :, pos, :] = cache.apply_ln_to_stack(
            stack[:, :, pos, :], layer=-1, pos_slice=pos
        )
    flip_dirs: Float[Tensor, "components pos d_model"] = (
        stack[:, is_positive, :, :] - stack[:, ~is_positive, :, :]
    ).mean(dim=1)
    return flip_dirs