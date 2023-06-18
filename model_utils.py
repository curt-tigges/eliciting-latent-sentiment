import os
import torch
import gc

from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"


def clear_gpu_memory(model):
    """Clears GPU memory by deleting model and running garbage collection.

    Args:
        model (torch.nn.Module): Model to clear from GPU memory.
    """
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_hf_name, model_tl_name, revision, cache_dir, fp16=False):
    """Loads a model from HuggingFace and wraps it in a TransformerLens.

    Args:
        model_hf_name (str): Name of model on HuggingFace.
        model_tl_name (str): Name of model in TransformerLens. This can be different from model_hf_name; check the
            TransformerLens documentation for more information.
        revision (str): Revision of model on HuggingFace.
        cache_dir (str): Directory to cache model in.

    Returns:
        HookedTransformer: Model wrapped in TransformerLens.
    """

    cache_model_name = model_hf_name
    if cache_model_name.startswith("EleutherAI/"):
        cache_model_name = cache_model_name[
            11:
        ]  # Remove "EleutherAI/" from model_hf_name

    cache_dir = cache_dir + f"/{cache_model_name}/{revision}"

    print(cache_dir)
    # Download model from HuggingFace
    source_model = AutoModelForCausalLM.from_pretrained(
        model_hf_name, revision=revision, cache_dir=cache_dir
    ).half()

    # source_model.to("cpu")

    # Load model into TransformerLens
    model = HookedTransformer.from_pretrained(
        model_tl_name,
        hf_model=source_model,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        # move_state_dict_to_device=False,
        # device="cpu",
    )

    # model.cfg.device = device
    # model.to(device)
    clear_gpu_memory(source_model)

    return model


def load_model_tl(model_tl_name, revision, cache_dir, fp16=False):
    """Loads a model from HuggingFace and wraps it in a TransformerLens.

    Args:
        model_hf_name (str): Name of model on HuggingFace.
        model_tl_name (str): Name of model in TransformerLens. This can be different from model_hf_name; check the
            TransformerLens documentation for more information.
        revision (str): Revision of model on HuggingFace.
        cache_dir (str): Directory to cache model in.

    Returns:
        HookedTransformer: Model wrapped in TransformerLens.
    """

    cache_model_name = model_tl_name
    if cache_model_name.startswith("EleutherAI/"):
        cache_model_name = cache_model_name[
            11:
        ]  # Remove "EleutherAI/" from model_hf_name

    cache_dir = cache_dir + f"/{cache_model_name}/{revision}"
    print(f"Saving model to {cache_dir}")

    # Load model into TransformerLens
    model = HookedTransformer.from_pretrained(
        model_tl_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        checkpoint_value=revision,
        model_kwargs={"torch_dtype": "float16", "cache_dir": cache_dir},
    )

    return model
