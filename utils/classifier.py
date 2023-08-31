import einops
from typing import List, Union
import jaxtyping
from typeguard import typechecked
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from utils.store import load_pickle


class HookedClassifier(HookedTransformer):

    def __init__(
        self, 
        base_model: HookedTransformer, 
        class_layer_weights: jaxtyping.Float[Tensor, "num_classes d_model"]
    ):
        super().__init__(base_model.cfg)
        self.base_model = base_model
        self.device = self.base_model.cfg.device
        self.class_layer_weights = class_layer_weights.to(self.device)

    @typechecked
    def forward(
        self, 
        input: Union[str, List[str], jaxtyping.Int[Tensor, 'batch pos']], 
        return_type: str = 'logits'
    ) -> jaxtyping.Float[Tensor, "batch *num_classes"]:
        _, cache = self.base_model.run_with_cache(input, return_type=None)
        last_token_act: jaxtyping.Float[
            Tensor, "batch d_model"
        ] = cache['ln_final.hook_normalized'][:, -1, :]
        logits: jaxtyping.Float[Tensor, "batch num_classes"] = einops.einsum(
            self.class_layer_weights,
            last_token_act,
            "batch d_model, num_classes d_model -> batch num_classes"
        )
        if return_type == 'logits':
            return logits
        elif return_type == 'prediction':
            return logits.argmax(dim=-1)
        elif return_type == 'probabilities':
            return torch.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

    @classmethod
    def from_pretrained(
        cls,
        classifier_path: str,
        class_layer_path: str,
        model_name: str,
        **from_pretrained_kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(classifier_path)
        class_layer_weights = load_pickle(class_layer_path, model_name)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=model,
            **from_pretrained_kwargs
        )
        return cls(model, torch.tensor(class_layer_weights['score.weight']))