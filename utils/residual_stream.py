from typing import List
from jaxtyping import Int, Float
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from utils.prompts import get_dataset


class ResidualStreamDataset:

    def __init__(
        self, 
        prompt_strings: List[str], 
        prompt_tokens: Int[Tensor, "batch pos"], 
        binary_labels: Int[Tensor, "batch"],
        model: HookedTransformer,
        prompt_type: str,
    ) -> None:
        assert len(prompt_strings) == len(prompt_tokens)
        assert len(prompt_strings) == len(binary_labels)
        self.prompt_strings = prompt_strings
        self.prompt_tokens = prompt_tokens
        self.binary_labels = binary_labels
        self.model = model
        self.prompt_type = prompt_type
        example_str_tokens = model.to_str_tokens(prompt_strings[0])
        self.example = [f"{i}:{tok}" for i, tok in enumerate(example_str_tokens)]
        self.placeholder_dict = prompt_type.get_placeholder_positions(example_str_tokens)
        label_positions = [pos for _, positions in self.placeholder_dict.items() for pos in positions]
        self.str_labels = [
            ''.join([model.to_str_tokens(prompt)[pos] for pos in label_positions]) 
            for prompt in prompt_strings
        ]

    def __len__(self) -> int:
        return len(self.prompt_strings)
    
    def __eq__(self, other: object) -> bool:
        return set(self.prompt_strings) == set(other.prompt_strings)
    
    def embed(self, position_type: str, layer: int) -> Float[Tensor, "batch d_model"]:
        assert 0 <= layer <= self.model.cfg.n_layers
        assert position_type in self.placeholder_dict.keys(), (
            f"Position type {position_type} not found in {self.placeholder_dict.keys()} "
            f"for prompt type {self.prompt_type}"
        )
        embed_position = self.placeholder_dict[position_type][-1]
        hook = 'resid_pre'
        if layer == self.model.cfg.n_layers:
            hook = 'resid_post'
            layer -= 1
        _, cache = self.model.run_with_cache(
            self.prompt_tokens, return_type=None, names_filter = lambda name: hook in name
        )
        out: Float[Tensor, "batch pos d_model"] = cache[hook, layer]
        return out[:, embed_position, :].cpu().detach().numpy()
    
    @classmethod
    def get_dataset(
        cls,
        model: HookedTransformer,
        device: torch.device,
        prompt_type: str = "simple_train"
    ) -> 'ResidualStreamDataset':
        all_prompts, answer_tokens, clean_tokens, _ = get_dataset(model, device, prompt_type=prompt_type)
        clean_labels = answer_tokens[:, 0, 0] == answer_tokens[0, 0, 0]
        
        assert len(all_prompts) == len(answer_tokens)
        assert len(all_prompts) == len(clean_tokens)
        return cls(
            all_prompts,
            clean_tokens,
            clean_labels,
            model,
            prompt_type,
        )
    

    def _get_labels_by_class(self, label: int) -> List[str]:
        return [
            string.strip() for string, one_hot in zip(self.str_labels, self.binary_labels) if one_hot == label
        ]
    
    def get_positive_negative_labels(self):
        return (
            self._get_labels_by_class(1),
            self._get_labels_by_class(0),
        )
    
    