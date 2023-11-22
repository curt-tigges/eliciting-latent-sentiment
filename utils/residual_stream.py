from typing import Callable, List, Optional, Sequence, Tuple, Union
import einops
from jaxtyping import Int, Float, Bool
from typeguard import typechecked
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
from tqdm.auto import tqdm
from utils.prompts import ReviewScaffold, get_dataset, PromptType


def get_resid_name(layer: int, model: HookedTransformer) -> Tuple[str, int]:
    resid_type = "resid_pre"
    if layer == model.cfg.n_layers:
        resid_type = "resid_post"
        layer -= 1
    return get_act_name(resid_type, layer), layer


class ResidualStreamDataset:
    @typechecked
    def __init__(
        self,
        prompt_strings: List[str],
        prompt_tokens: Int[Tensor, "batch pos"],
        binary_labels: Bool[Tensor, "batch"],
        model: HookedTransformer,
        prompt_type: PromptType,
    ) -> None:
        assert len(prompt_strings) == len(prompt_tokens)
        assert len(prompt_strings) == len(binary_labels)
        self.prompt_strings = prompt_strings
        self.prompt_tokens = prompt_tokens
        self._binary_labels = binary_labels
        self.model = model
        self.prompt_type = prompt_type
        example_str_tokens: List[str] = model.to_str_tokens(prompt_strings[0])  # type: ignore
        self.example = [f"{i}:{tok}" for i, tok in enumerate(example_str_tokens)]
        self.placeholder_dict = prompt_type.get_placeholder_positions(
            example_str_tokens
        )
        label_positions = [
            pos for _, positions in self.placeholder_dict.items() for pos in positions
        ]
        self.str_labels = [
            "".join([model.to_str_tokens(prompt)[pos] for pos in label_positions])  # type: ignore
            for prompt in prompt_strings
        ]

    @property
    def binary_labels(self) -> Bool[Tensor, "batch"]:
        return self._binary_labels.cpu().detach()

    def __len__(self) -> int:
        return len(self.prompt_strings)

    def __eq__(self, other: "ResidualStreamDataset") -> bool:
        return set(self.prompt_strings) == set(other.prompt_strings)

    def get_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        assert batch_size is not None, "get_dataloader: must specify batch size"
        token_answer_dataset = TensorDataset(
            self.prompt_tokens,
        )
        token_answer_dataloader = DataLoader(
            token_answer_dataset, batch_size=batch_size
        )
        return token_answer_dataloader

    def run_with_cache(
        self,
        names_filter: Callable,
        batch_size: int,
        requires_grad: bool = True,
        device: Optional[torch.device] = None,
        disable_tqdm: Optional[bool] = None,
        leave_tqdm: bool = False,
    ):
        """
        Note that variable names here assume denoising, i.e. corrupted -> clean
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        model = self.model.eval().requires_grad_(False)
        assert batch_size is not None, "run_with_cache: must specify batch size"
        if model.cfg.device != device:
            model = model.to(device)
        act_dict = dict()
        dataloader = self.get_dataloader(batch_size=batch_size)
        buffer_initialized = False
        total_samples = len(dataloader.dataset)
        if disable_tqdm is None:
            disable_tqdm = len(dataloader) > 1
        bar = tqdm(dataloader, disable=disable_tqdm, leave=leave_tqdm)
        bar.set_description(
            f"Running with cache: model={model.cfg.model_name}, "
            f"batch_size={batch_size}"
        )
        for idx, (prompt_tokens,) in enumerate(bar):
            prompt_tokens = prompt_tokens.to(device)
            with torch.inference_mode():
                # forward pass
                _, fwd_cache = model.run_with_cache(
                    prompt_tokens, names_filter=names_filter, return_type=None
                )
                fwd_cache.to("cpu")

                # Initialise the buffer tensors if necessary
                if not buffer_initialized:
                    for k, v in fwd_cache.items():
                        act_dict[k] = torch.zeros(
                            (total_samples, *v.shape[1:]),
                            dtype=self.model.dtype,
                            device="cpu",
                        )
                    buffer_initialized = True

                # Fill the buffer tensors
                start_idx = idx * batch_size
                end_idx = start_idx + prompt_tokens.size(0)
                for k, v in fwd_cache.items():
                    act_dict[k][start_idx:end_idx] = v
        act_cache = ActivationCache(
            {
                k: v.detach().clone().requires_grad_(requires_grad)
                for k, v in act_dict.items()
            },
            model=model,
        )
        act_cache.to("cpu")
        torch.set_grad_enabled(was_grad_enabled)
        model = model.train().requires_grad_(requires_grad)

        return None, act_cache

    @typechecked
    def embed(
        self,
        position_type: Union[str, None],
        layer: int,
        batch_size: int = 64,
        seed: int = 0,
    ) -> Float[Tensor, "batch d_model"]:
        """
        Returns a dataset of embeddings at the specified position and layer.
        Useful for training classifiers on the residual stream.
        """
        assert self.model.tokenizer is not None, "embed: model must have tokenizer"
        torch.manual_seed(seed)
        assert 0 <= layer <= self.model.cfg.n_layers
        assert position_type is None or position_type in self.placeholder_dict.keys(), (
            f"Position type {position_type} not found in {self.placeholder_dict.keys()} "
            f"for prompt type {self.prompt_type}"
        )
        hook, _ = get_resid_name(layer, self.model)
        _, cache = self.run_with_cache(
            names_filter=lambda name: hook == name, batch_size=batch_size
        )
        out: Float[Tensor, "batch pos d_model"] = cache[hook]
        if position_type is None:
            # Step 1: Identify non-zero positions in the tensor
            non_pad_mask: Bool[Tensor, "batch pos"] = (
                self.prompt_tokens != self.model.tokenizer.pad_token_id
            )

            # Step 2: Check if values at these positions are not constant across batches
            non_constant_mask: Bool[Tensor, "pos"] = (
                self.prompt_tokens != self.prompt_tokens[0]
            ).any(dim=0)
            valid_positions: Bool[Tensor, "batch pos"] = (
                non_pad_mask & non_constant_mask
            )

            # Step 3: Randomly sample from these positions for each batch
            embed_position: Int[Tensor, "batch"] = (
                torch.multinomial(valid_positions.float(), 1)
                .squeeze()
                .to(device=out.device)
            )
            return out[torch.arange(len(out)), embed_position, :].detach().cpu()
        else:
            embed_pos: int = self.placeholder_dict[position_type][-1]
            return out[:, embed_pos, :].detach().cpu()

    @classmethod
    def get_dataset(
        cls,
        model: HookedTransformer,
        device: torch.device,
        prompt_type: PromptType = PromptType.SIMPLE_TRAIN,
        scaffold: Optional[ReviewScaffold] = None,
    ) -> Union[None, "ResidualStreamDataset"]:
        """
        N.B. labels assume that first batch corresponds to 1
        """
        if prompt_type == PromptType.NONE:
            return None
        clean_corrupt_data = get_dataset(
            model, device, prompt_type=prompt_type, scaffold=scaffold
        )
        clean_labels = (
            clean_corrupt_data.answer_tokens[:, 0, 0]
            == clean_corrupt_data.answer_tokens[0, 0, 0]
        )

        assert len(clean_corrupt_data.all_prompts) == len(
            clean_corrupt_data.answer_tokens
        )
        assert len(clean_corrupt_data.all_prompts) == len(
            clean_corrupt_data.clean_tokens
        )
        return cls(
            clean_corrupt_data.all_prompts,
            clean_corrupt_data.clean_tokens,
            clean_labels,
            model,
            prompt_type,
        )

    def _get_labels_by_class(self, label: int) -> List[str]:
        return [
            string.strip()
            for string, one_hot in zip(self.str_labels, self.binary_labels)
            if one_hot == label
        ]

    def get_positive_negative_labels(self):
        return (
            self._get_labels_by_class(1),
            self._get_labels_by_class(0),
        )
