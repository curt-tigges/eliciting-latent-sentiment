import einops
from typing import Callable, Dict, List, Tuple, Union
import jaxtyping
from jaxtyping import Float
from typeguard import typechecked
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.HookedTransformer import Loss
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
        return_type: str = 'logits',
        return_cache_object=False,
        names_filter: Callable = None,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end=True,
        clear_contexts=False,
    ) -> Union[
            Tuple[jaxtyping.Float[Tensor, "batch *num_classes"], ActivationCache], 
            jaxtyping.Float[Tensor, "batch *num_classes"]
        ]:
        if names_filter is None:
            names_filter = lambda _: True
        new_names_filter = lambda name: names_filter(name) or name == 'ln_final.hook_normalized'
        if fwd_hooks or bwd_hooks:
            with self.base_model.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as base_model:
                _, cache = base_model.run_with_cache(
                    input, return_type=None, names_filter=new_names_filter
                )
        else:
            _, cache = self.base_model.run_with_cache(
                input, return_type=None, names_filter=new_names_filter
            )
        last_token_act: jaxtyping.Float[
            Tensor, "batch d_model"
        ] = cache['ln_final.hook_normalized'][:, -1, :]
        logits: jaxtyping.Float[Tensor, "batch num_classes"] = einops.einsum(
            self.class_layer_weights,
            last_token_act,
            "num_classes d_model, batch d_model -> batch num_classes"
        )
        if return_type == 'logits':
            out = logits
        elif return_type == 'prediction':
            out = logits.argmax(dim=-1)
        elif return_type == 'probabilities':
            out = torch.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
        if return_cache_object:
            return out, cache
        return out
        
    def run_with_cache(
        self, *model_args, return_cache_object=True, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        return self.forward(
            *model_args, return_cache_object=return_cache_object, **kwargs
        )
    
    def run_with_hooks(
        self,
        *model_args,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        return self.forward(
            *model_args,
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
            **model_kwargs,
        )


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