import random
import numpy as np
import yaml
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_attention_mask
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Float, Int, Bool
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, TypeVar
from typeguard import typechecked
import einops
from enum import Enum
import re
from tqdm.auto import tqdm
from collections import UserList
from utils.store import load_pickle
from utils.circuit_analysis import (
    get_logit_diff, get_prob_diff, center_logit_diffs, 
    get_accuracy_from_logit_diffs,
    get_final_non_pad_token,
)

T = TypeVar("T")


class ReviewScaffold(Enum):
    PLAIN = 'plain'
    CLASSIFICATION = 'classification'
    CONTINUATION = 'continuation'


def extract_placeholders(text: str) -> List[str]:
    # Use regex to find all instances of {SOME_TEXT}
    matches = re.findall(r'\{(\w+)\}', text)
    return matches


def dedup_list(input_list: list) -> list:
    seen = set()
    return [x for x in input_list if x not in seen and not seen.add(x)]


def interleave_list(lst: list) -> list:
    """
    Reorders a list to interleave its first and second halves.
    If the list has an odd length, the extra element will be included in the first half.
    """
    midpoint = len(lst) // 2 + len(lst) % 2  # This ensures the first half gets the extra element if length is odd
    first_half = lst[:midpoint]
    second_half = lst[midpoint:]
    
    result = []
    for i in range(midpoint):
        result.append(first_half[i])
        if i < len(second_half):  # Make sure we don't go out of bounds for the second half
            result.append(second_half[i])

    return result


def filter_words_by_length(model: HookedTransformer, words: list, length: int, verbose=False) -> list:
    if verbose:
        print("Filtering words by length")
    new_words = []
    for word in words:
        tkn = model.to_str_tokens(word, prepend_bos=False)
        if len(tkn) == length:
            new_words.append(word)
    if verbose:
        print(f"Count of words: {len(words)}")

    return new_words


def truncate_words_by_length(model: HookedTransformer, words: list, length: int, verbose=False) -> list:
    if verbose:
        print("Truncating words by length")
    new_words = []
    for word in words:
        tkn = model.to_str_tokens(word, prepend_bos=False)
        trunc = ''.join(tkn[:length])
        new_words.append(trunc)
    return new_words


class CircularList(UserList[T]):
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]
        return super(CircularList, self).__getitem__(index % len(self))


class PromptsConfig:

    def __init__(self) -> None:
        with open("prompts.yaml", "r") as f:
            prompts_dict = yaml.safe_load(f)
        self._prompts_dict = prompts_dict
        
    def get(
        self, 
        key: str, 
        model: HookedTransformer, 
        filter_length: Optional[int] = None, 
        truncate_length: Optional[int] = None,
        drop_duplicates: bool = True,
        prepend_space: bool = True, 
        verbose: bool = False,
    ) -> CircularList:
        assert filter_length is not None or truncate_length is not None, (
            "Must specify at least one of filter_length or truncate_length"
        )
        words: list = self._prompts_dict[key]
        if prepend_space:
            words = [" " + word.strip() for word in words]
        if filter_length is not None:
            words = filter_words_by_length(model, words, filter_length, verbose=verbose)
        if truncate_length is not None:
            words = truncate_words_by_length(model, words, truncate_length, verbose=verbose)
        if drop_duplicates:
            words = dedup_list(words)
        return CircularList(words)
    


class PromptType(Enum):
    SIMPLE = "simple"
    SIMPLE_TRAIN = "simple_train"
    SIMPLE_TEST = "simple_test"
    SIMPLE_RES = "simple_res"
    SIMPLE_PRODUCT = "simple_product"
    SIMPLE_BOOK = "simple_book"
    COMPLETION = "completion"
    COMPLETION_2 = "completion_2"
    COMPLETION_3 = "completion_3"
    CLASSIFICATION = "classification"
    CLASSIFICATION_2 = "classification_2"
    CLASSIFICATION_3 = "classification_3"
    CLASSIFICATION_4 = "classification_4"
    RES_CLASS_1 = "res_class_1"
    SIMPLE_ADVERB = "simple_adverb"
    SIMPLE_FRENCH = "simple_french"
    PROPER_NOUNS = "proper_nouns"
    MEDICAL = "medical"
    MULTI_SUBJECT_1 = "multi_subject_1"
    TREEBANK_TRAIN = "treebank_train"
    TREEBANK_DEV = "treebank_dev"
    TREEBANK_TEST = "treebank_test"
    NONE = "none"

    def is_treebank(self):
        return self in [
            PromptType.TREEBANK_TRAIN,
            PromptType.TREEBANK_DEV,
            PromptType.TREEBANK_TEST,
            PromptType.NONE,
        ]

    def get_format_string(self):
        if self.is_treebank():
            return None
        prompt_strings = {
            PromptType.SIMPLE_BOOK: "The novel was{VRB} by the book club members. The comments were universally{ADJ} \nConclusion: You will be very",
            PromptType.SIMPLE_RES: "This restaurant is a{NOUN}, you are going to{VRB} it. \nConclusion: This restaurant is very",
            PromptType.SIMPLE_PRODUCT: "I am{FEEL} about this purchase, the product is{ADJ}. \nConclusion: I feel very",
            PromptType.SIMPLE: "I thought this movie was{ADJ}, I{VRB} it. \nConclusion: This movie is",
            PromptType.SIMPLE_TRAIN: "I thought this movie was{ADJ}, I{VRB} it. \nConclusion: This movie is",
            PromptType.SIMPLE_TEST: "I thought this movie was{ADJ}, I{VRB} it. \nConclusion: This movie is",
            PromptType.SIMPLE_ADVERB: "The traveller{ADV} walked to their destination. The traveller felt very",
            PromptType.SIMPLE_FRENCH: "Je pensais que ce film était{ADJ}, je l'ai{VRB}. \nConclusion: Ce film est très",
            PromptType.PROPER_NOUNS: "When I hear the name{NOUN}, I feel very",
            PromptType.MEDICAL: "I heard the doctor use the word{MED} and I felt very",
            PromptType.COMPLETION: "I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was just very",
            PromptType.COMPLETION_2: "I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall it was just very",
            PromptType.COMPLETION_3: "I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was just very {ADJ4}. I felt it was",
            PromptType.CLASSIFICATION: "Review Text: 'I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was just very {ADJ4}.' \nReview Sentiment:",
            PromptType.CLASSIFICATION_2: "Review Text: I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was just very {ADJ4}. \nReview Sentiment:",
            PromptType.CLASSIFICATION_3: "Review Text: I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was just very {ADJ4}. Review Sentiment:",
            PromptType.CLASSIFICATION_4: "I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was just very {ADJ4}. Review Sentiment:",
            PromptType.RES_CLASS_1: "This restaurant was{ADJ1}, I{VRB} it. The food was{ADJ2}, and the service was{ADJ3}. Overall it was just",
            PromptType.MULTI_SUBJECT_1: (
                "Review A: 'I thought this movie was{SUBJ1_ADJ1}, I {SUBJ1_VRB} it. The acting was{SUBJ1_ADJ2}, the plot was{SUBJ1_ADJ3}, "
                "and overall the movie was just very {SUBJ1_ADJ4}.'\n"
                "Review B: 'I thought this movie was{SUBJ2_ADJ1}, I {SUBJ2_VRB} it. The acting was{SUBJ2_ADJ2}, the plot was{SUBJ2_ADJ3}, "
                "and overall the movie was just very {SUBJ2_ADJ4}.'\n"
                "Review {SUBJ} Sentiment:"
            )
        }
        return prompt_strings[self]
    
    def get_placeholders(self) -> List[str]:
        '''
        Example output: ['ADJ', 'VRB']
        '''
        if self.is_treebank():
            return []
        formatter = self.get_format_string()
        return extract_placeholders(formatter)
    
    @typechecked
    def get_placeholder_positions(self, token_list: List[str]) -> Dict[str, List[int]]:
        '''
        Identifies placeholder positions in a list of string tokens.
        Handles whether the placeholder is a single token or multi-token.
        Example output: {'ADJ': [4, 5], 'VRB': [8]}
        '''
        format_string = self.get_format_string()
        if format_string is None or self.is_treebank():
            return {}
        format_idx = 0
        curr_sub_token = None
        out = dict()
        for token_index, token in enumerate(token_list):
            if format_string[format_idx] == '{':
                curr_sub_token = format_string[format_idx + 1:format_string.find('}', format_idx)]
            if format_string.find(token, format_idx) >= 0:
                format_idx = format_string.find(token, format_idx) + len(token)
            elif curr_sub_token is not None:
                out[curr_sub_token] = out.get(curr_sub_token, []) + [token_index]
        return out
    

prompt_config = PromptsConfig()


def get_prompts(
    model: HookedTransformer,
    prompt_type: PromptType = PromptType.SIMPLE, 
) -> Tuple[Dict[str, List[str]], Dict[str, CircularList[str]]]:
    # Define output types
    pos_prompts: List[str]
    neg_prompts: List[str]
    neutral_prompts: Union[List[str], None]

    # Read lists from config
    pos_answers: CircularList[str] = prompt_config.get("positive_answer_tokens", model, filter_length=1)
    neg_answers: CircularList[str] = prompt_config.get("negative_answer_tokens", model, filter_length=1)
    positive_adjectives: CircularList[str] = prompt_config.get("positive_core_adjectives", model, filter_length=1)
    negative_adjectives: CircularList[str] = prompt_config.get("negative_core_adjectives", model, filter_length=1)
    neutral_adjectives: CircularList[str] = prompt_config.get("neutral_core_adjectives", model, filter_length=1)
    positive_verbs: CircularList[str] = prompt_config.get("positive_verbs", model, filter_length=1)
    negative_verbs: CircularList[str] = prompt_config.get("negative_verbs", model, filter_length=1)
    neutral_verbs: CircularList[str] = prompt_config.get("neutral_verbs", model, filter_length=1)
    positive_top_adjectives: CircularList[str] = prompt_config.get("positive_top_adjectives", model, filter_length=1)
    negative_top_adjectives: CircularList[str] = prompt_config.get("negative_top_adjectives", model, filter_length=1)
    neutral_top_adjectives: CircularList[str] = prompt_config.get("neutral_top_adjectives", model, filter_length=1)

    # Get prompt type/format
    prompt_type = PromptType(prompt_type)
    formatter = prompt_type.get_format_string()
    assert formatter is not None

    if prompt_type == PromptType.SIMPLE:
        n_prompts = min(len(positive_adjectives), len(negative_adjectives))
        pos_prompts = [formatter.format(ADJ=positive_adjectives[i], VRB=positive_verbs[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADJ=negative_adjectives[i], VRB=negative_verbs[i]) for i in range(n_prompts)]
        neutral_prompts = [formatter.format(ADJ=neutral_adjectives[i], VRB=neutral_verbs[i]) for i in range(len(neutral_adjectives))]
    elif prompt_type == PromptType.SIMPLE_TRAIN:
        n_prompts = min(len(positive_adjectives), len(negative_adjectives))
        positive_adjectives = prompt_config.get("positive_adjectives_train", model, filter_length=1)
        negative_adjectives = prompt_config.get("negative_adjectives_train", model, filter_length=1)
        neutral_prompts = None
        pos_prompts = [formatter.format(ADJ=positive_adjectives[i], VRB=positive_verbs[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADJ=negative_adjectives[i], VRB=negative_verbs[i]) for i in range(n_prompts)]
    elif prompt_type == PromptType.SIMPLE_TEST:
        positive_adjectives = prompt_config.get("positive_adjectives_test", model, filter_length=1)
        negative_adjectives = prompt_config.get("negative_adjectives_test", model, filter_length=1)
        n_prompts = min(len(positive_adjectives), len(negative_adjectives))
        positive_adjectives = prompt_config.get("positive_adjectives_test", model, filter_length=1)
        negative_adjectives = prompt_config.get("negative_adjectives_test", model, filter_length=1)
        neutral_prompts = None
        pos_prompts = [formatter.format(ADJ=positive_adjectives[i], VRB=positive_verbs[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADJ=negative_adjectives[i], VRB=negative_verbs[i]) for i in range(n_prompts)]
    elif prompt_type == PromptType.SIMPLE_BOOK:
        positive_adjectives = prompt_config.get("positive_comment_adjectives", model, filter_length=1)
        negative_adjectives = prompt_config.get("negative_comment_adjectives", model, filter_length=1)
        n_prompts = min(len(positive_adjectives), len(negative_adjectives), len(negative_verbs), len(positive_verbs))
        pos_prompts = [formatter.format(ADJ=positive_adjectives[i], VRB=positive_verbs[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADJ=negative_adjectives[i], VRB=negative_verbs[i]) for i in range(n_prompts)]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_answer_adjectives", model, filter_length=1)
        neg_answers = prompt_config.get("negative_answer_adjectives", model, filter_length=1)
    elif prompt_type == PromptType.SIMPLE_RES:
        positive_nouns = prompt_config.get("positive_nouns", model, filter_length=1)
        negative_nouns = prompt_config.get("negative_nouns", model, filter_length=1)
        positive_infinitives = prompt_config.get("positive_infinitives", model, filter_length=1)
        negative_infinitives = prompt_config.get("negative_infinitives", model, filter_length=1)
        n_prompts = min(len(positive_nouns), len(negative_nouns), len(negative_infinitives), len(positive_infinitives))
        pos_prompts = [formatter.format(NOUN=positive_nouns[i], VRB=positive_infinitives[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(NOUN=negative_nouns[i], VRB=negative_infinitives[i]) for i in range(n_prompts)]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_very_answers", model, filter_length=1)
        neg_answers = prompt_config.get("negative_very_answers", model, filter_length=1)
    elif prompt_type == PromptType.SIMPLE_PRODUCT:
        positive_feelings = prompt_config.get("positive_feelings", model, filter_length=1)
        negative_feelings = prompt_config.get("negative_feelings", model, filter_length=1)
        n_prompts = min(len(positive_feelings), len(negative_feelings), len(negative_adjectives), len(positive_adjectives))
        pos_prompts = [formatter.format(ADJ=positive_adjectives[i], FEEL=positive_feelings[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADJ=negative_adjectives[i], FEEL=negative_feelings[i]) for i in range(n_prompts)]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_moods", model, filter_length=1)
        neg_answers = prompt_config.get("negative_moods", model, filter_length=1)
    elif prompt_type == PromptType.SIMPLE_ADVERB:
        positive_adverbs = prompt_config.get("positive_adverbs", model, filter_length=2)
        negative_adverbs = prompt_config.get("negative_adverbs", model, filter_length=2)
        n_prompts = min(len(positive_adverbs), len(negative_adverbs))
        pos_prompts = [formatter.format(ADV=positive_adverbs[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADV=negative_adverbs[i]) for i in range(n_prompts)]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_moods", model, filter_length=1)
        neg_answers = prompt_config.get("negative_moods", model, filter_length=1)
    elif prompt_type == PromptType.SIMPLE_FRENCH:
        positive_french_adj = prompt_config.get("positive_french_adjectives", model, filter_length=3)
        negative_french_adj = prompt_config.get("negative_french_adjectives", model, filter_length=3)
        positive_french_verbs = prompt_config.get("positive_french_verbs", model, filter_length=3)
        negative_french_verbs = prompt_config.get("negative_french_verbs", model, filter_length=3)
        n_prompts = min(len(positive_french_adj), len(negative_french_adj))
        pos_prompts = [formatter.format(ADJ=positive_french_adj[i], VRB=positive_french_verbs[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(ADJ=negative_french_adj[i], VRB=negative_french_verbs[i]) for i in range(n_prompts)]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_french_answers", model, truncate_length=1)
        neg_answers = prompt_config.get("negative_french_answers", model, truncate_length=1)
    elif prompt_type == PromptType.PROPER_NOUNS:
        positive_proper = prompt_config.get("positive_proper_nouns", model, filter_length=1)
        negative_proper = prompt_config.get("negative_proper_nouns", model, filter_length=1)
        n_prompts = min(len(positive_proper), len(negative_proper))
        pos_prompts = [formatter.format(NOUN=positive_proper[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(NOUN=negative_proper[i]) for i in range(n_prompts)]
        neutral_prompts = None
    elif prompt_type == PromptType.MEDICAL:
        positive_medical = prompt_config.get("positive_medical", model, filter_length=1)
        negative_medical = prompt_config.get("negative_medical", model, filter_length=1)
        n_prompts = min(len(positive_medical), len(negative_medical))
        pos_prompts = [formatter.format(MED=positive_medical[i]) for i in range(n_prompts)]
        neg_prompts = [formatter.format(MED=negative_medical[i]) for i in range(n_prompts)]
        neutral_prompts = None
    elif prompt_type in (
        PromptType.COMPLETION, PromptType.COMPLETION_2, PromptType.RES_CLASS_1
    ):
        n_prompts = min(len(positive_adjectives), len(negative_adjectives))
        pos_prompts = [
            formatter.format(ADJ1=positive_adjectives[i], ADJ2=positive_adjectives[i + 1], ADJ3=positive_adjectives[i + 2], VRB=positive_verbs[i])
            for i in range(n_prompts)
        ]
        neg_prompts = [
            formatter.format(ADJ1=negative_adjectives[i], ADJ2=negative_adjectives[i + 1], ADJ3=negative_adjectives[i + 2], VRB=negative_verbs[i])
            for i in range(n_prompts)
        ]
        neutral_prompts = [
            formatter.format(ADJ1=neutral_adjectives[i], ADJ2=neutral_adjectives[i + 1], ADJ3=neutral_adjectives[i + 2], VRB=neutral_verbs[i])
            for i in range(n_prompts)
        ]
    elif prompt_type in (
        PromptType.CLASSIFICATION, PromptType.CLASSIFICATION_2, PromptType.CLASSIFICATION_3, PromptType.CLASSIFICATION_4
    ):
        n_prompts = min(len(positive_adjectives), len(negative_adjectives))
        pos_prompts = [
            formatter.format(ADJ1=positive_adjectives[i], ADJ2=positive_adjectives[i + 1], ADJ3=positive_adjectives[i + 2], ADJ4=positive_top_adjectives[i], VRB=positive_verbs[i])
            for i in range(n_prompts)
        ]
        neg_prompts = [
            formatter.format(ADJ1=negative_adjectives[i], ADJ2=negative_adjectives[i + 1], ADJ3=negative_adjectives[i + 2], ADJ4=negative_top_adjectives[i], VRB=negative_verbs[i])
            for i in range(n_prompts)
        ]
        neutral_prompts = [
            formatter.format(ADJ1=neutral_adjectives[i], ADJ2=neutral_adjectives[i + 1], ADJ3=neutral_adjectives[i + 2], ADJ4=neutral_top_adjectives[i], VRB=neutral_verbs[i])
            for i in range(n_prompts)
        ]
        pos_answers = CircularList([" Positive"])
        neg_answers = CircularList([" Negative"])
    elif prompt_type == PromptType.MULTI_SUBJECT_1:
        n_prompts = min(len(positive_adjectives), len(negative_adjectives))
        pos_prompts = [
            formatter.format(
                SUBJ1_ADJ1=positive_adjectives[i], SUBJ1_ADJ2=positive_adjectives[i + 1], SUBJ1_ADJ3=positive_adjectives[i + 2], SUBJ1_ADJ4=positive_top_adjectives[i], SUBJ1_VRB=positive_verbs[i],
                SUBJ2_ADJ1=negative_adjectives[i], SUBJ2_ADJ2=negative_adjectives[i + 1], SUBJ2_ADJ3=negative_adjectives[i + 2], SUBJ2_VRB=negative_verbs[i], SUBJ2_ADJ4=negative_top_adjectives[i],
                SUBJ="A",
            ) for i in range(n_prompts)
        ] + [
            formatter.format(
                SUBJ1_ADJ1=negative_adjectives[i], SUBJ1_ADJ2=negative_adjectives[i + 1], SUBJ1_ADJ3=negative_adjectives[i + 2], SUBJ1_VRB=negative_verbs[i], SUBJ1_ADJ4=negative_top_adjectives[i],
                SUBJ2_ADJ1=positive_adjectives[i], SUBJ2_ADJ2=positive_adjectives[i + 1], SUBJ2_ADJ3=positive_adjectives[i + 2], SUBJ2_VRB=positive_verbs[i], SUBJ2_ADJ4=positive_top_adjectives[i],
                SUBJ="B",
            ) for i in range(n_prompts)
        ]
        neg_prompts = [
            formatter.format(
                SUBJ1_ADJ1=positive_adjectives[i], SUBJ1_ADJ2=positive_adjectives[i + 1], SUBJ1_ADJ3=positive_adjectives[i + 2], SUBJ1_VRB=positive_verbs[i], SUBJ1_ADJ4=positive_top_adjectives[i],
                SUBJ2_ADJ1=negative_adjectives[i], SUBJ2_ADJ2=negative_adjectives[i + 1], SUBJ2_ADJ3=negative_adjectives[i + 2], SUBJ2_VRB=negative_verbs[i], SUBJ2_ADJ4=negative_top_adjectives[i],
                SUBJ="B",
            ) for i in range(n_prompts)
        ] + [
            formatter.format(
                SUBJ1_ADJ1=negative_adjectives[i], SUBJ1_ADJ2=negative_adjectives[i + 1], SUBJ1_ADJ3=negative_adjectives[i + 2], SUBJ1_VRB=negative_verbs[i], SUBJ1_ADJ4=negative_top_adjectives[i],
                SUBJ2_ADJ1=positive_adjectives[i], SUBJ2_ADJ2=positive_adjectives[i + 1], SUBJ2_ADJ3=positive_adjectives[i + 2], SUBJ2_VRB=positive_verbs[i], SUBJ2_ADJ4=positive_top_adjectives[i],
                SUBJ="A",
            ) for i in range(n_prompts)
        ]
        pos_prompts = interleave_list(pos_prompts)
        neg_prompts = interleave_list(neg_prompts)
        neutral_prompts = None
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    # check length match
    assert len(pos_prompts) == len(neg_prompts), (
        f"Number of positive prompts ({len(pos_prompts)}) "
        f"does not match number of negative prompts ({len(neg_prompts)}). "
        f"Please check the prompts.yaml file. \n"
        f"Prompt type: {prompt_type}\n"
        f"Full list of positive prompts: {pos_prompts}. \n"
        f"Full list of negative prompts: {neg_prompts}."
    )

    # create output dicts
    prompt_dict = dict(
        positive=pos_prompts,
        negative=neg_prompts,
        neutral=neutral_prompts,
    )
    answer_dict = dict(
        positive=pos_answers,
        negative=neg_answers,
    )
    return prompt_dict, answer_dict


class CleanCorruptedDataset(torch.utils.data.Dataset):
    clean_tokens: Int[Tensor, "batch pos"]
    corrupted_tokens: Int[Tensor, "batch pos"]
    answer_tokens: Int[Tensor, "batch pair correct"]

    @typechecked
    def __init__(
        self, 
        clean_tokens: Int[Tensor, "batch pos"], 
        corrupted_tokens: Int[Tensor, "batch pos"],
        answer_tokens: Int[Tensor, "batch *pair correct"],
        all_prompts: List[str], 
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        assert len(clean_tokens) == len(corrupted_tokens)
        assert len(clean_tokens) == len(answer_tokens)
        super().__init__()
        if answer_tokens.ndim == 2:
            answer_tokens = answer_tokens.unsqueeze(1)
        self.clean_tokens = clean_tokens
        self.corrupted_tokens = corrupted_tokens
        self.answer_tokens = answer_tokens
        self.all_prompts = all_prompts
        self.tokenizer = tokenizer

    def get_subset(self, indices: List[int]):
        return CleanCorruptedDataset(
            self.clean_tokens[indices],
            self.corrupted_tokens[indices],
            self.answer_tokens[indices],
            [self.all_prompts[i] for i in indices],
            self.tokenizer,
        )
    
    def get_num_pad_tokens(self) -> Int[Tensor, "batch"]:
        assert self.tokenizer is not None
        return (
            self.clean_tokens == self.tokenizer.pad_token_id
        ).sum(dim=1)
    
    def get_num_non_pad_tokens(self) -> Int[Tensor, "batch"]:
        assert self.tokenizer is not None
        return (
            self.clean_tokens == self.tokenizer.pad_token_id
        ).sum(dim=1)
    
    def restrict_by_padding(self, min_tokens: int, max_tokens: int):
        num_pad_tokens = self.get_num_non_pad_tokens()
        indices = torch.where(
            (num_pad_tokens >= min_tokens) & (num_pad_tokens <= max_tokens)
        )[0].tolist()
        return self.get_subset(indices)
    
    def shuffle(self, seed: int = 0):
        random.seed(seed)
        indices = list(range(len(self)))
        random.shuffle(indices)
        return self.get_subset(indices)

    def to(self, device: torch.device):
        self.clean_tokens = self.clean_tokens.to(device)
        self.corrupted_tokens = self.corrupted_tokens.to(device)
        self.answer_tokens = self.answer_tokens.to(device)
        return self

    def __len__(self):
        return self.clean_tokens.shape[0]
    
    def __getitem__(self, idx):
        return (
            self.clean_tokens[idx], 
            self.corrupted_tokens[idx], 
            self.answer_tokens[idx],
        )
    
    def get_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        assert batch_size is not None, "get_dataloader: must specify batch size"
        token_answer_dataset = TensorDataset(
            self.corrupted_tokens, 
            self.clean_tokens, 
            self.answer_tokens
        )
        token_answer_dataloader = DataLoader(token_answer_dataset, batch_size=batch_size)
        return token_answer_dataloader
    
    def forward(
        self,
        model: HookedTransformer, 
        batch_size: int,
        requires_grad: bool = True,
        device: Optional[torch.device] = None,
        disable_tqdm: Optional[bool] = None,
        dtype: torch.dtype = torch.float32,
        center: bool = True,
    ):
        return self.run_with_cache(
            model=model,
            names_filter=lambda _: False,
            batch_size=batch_size,
            requires_grad=requires_grad,
            device=device,
            disable_tqdm=disable_tqdm,
            dtype=dtype,
            center=center,
        )
    
    def run_with_cache(
        self, 
        model: HookedTransformer, 
        names_filter: Callable, 
        batch_size: int,
        requires_grad: bool = True,
        device: Optional[torch.device] = None,
        disable_tqdm: Optional[bool] = None,
        leave_tqdm: bool = False,
        dtype: torch.dtype = torch.float32,
        center: bool = True,
    ):
        """
        Note that variable names here assume denoising, i.e. corrupted -> clean
        """
        assert batch_size is not None, "run_with_cache: must specify batch size"
        model.tokenizer = self.tokenizer
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        model.requires_grad_(False)
        corrupted_dict = dict()
        clean_dict = dict()
        dataloader = self.get_dataloader(batch_size=batch_size)
        corrupted_logit_diffs = []
        clean_logit_diffs = []
        corrupted_prob_diffs = []
        clean_prob_diffs = []
        buffer_initialized = False

        # Initialise arrays
        total_samples = len(dataloader.dataset)
        clean_logit_diffs = torch.zeros(total_samples, dtype=dtype, device='cpu')
        corrupted_logit_diffs = torch.zeros(total_samples, dtype=dtype, device='cpu')
        clean_prob_diffs = torch.zeros(total_samples, dtype=dtype, device='cpu')
        corrupted_prob_diffs = torch.zeros(total_samples, dtype=dtype, device='cpu')

        corrupted_dict = dict()
        clean_dict = dict()
        if disable_tqdm is None:
            disable_tqdm = len(dataloader) == 1
        bar = tqdm(dataloader, disable=disable_tqdm, leave=leave_tqdm)
        bar.set_description(
            f"Running with cache: model={model.cfg.model_name}, "
            f"batch_size={batch_size}"
        )
        for idx, (corrupted_tokens, clean_tokens, answer_tokens) in enumerate(bar):
            corrupted_tokens = corrupted_tokens.to(device)
            clean_tokens = clean_tokens.to(device)
            answer_tokens = answer_tokens.to(device)
            clean_logits: Float[Tensor, "batch pos d_vocab"]
            corrupted_logits: Float[Tensor, "batch pos d_vocab"]
            with torch.inference_mode():
                # update bar description
                bar.set_description(
                    "run_with_cache: corrupted forward pass"
                )
                # corrupted forward pass
                corrupted_logits, corrupted_cache = model.run_with_cache(
                    corrupted_tokens, names_filter=names_filter
                ) # type: ignore
                corrupted_cache.to('cpu')

                # clean forward pass
                bar.set_description(
                    "run_with_cache: clean forward pass"
                )
                clean_logits, clean_cache = model.run_with_cache(
                    clean_tokens, names_filter=names_filter
                ) # type: ignore
                clean_cache.to('cpu')

                # Initialise the buffer tensors if necessary
                if not buffer_initialized:
                    bar.set_description(
                        f"run_with_cache: initializing {len(corrupted_cache)} "
                        f"buffers of size {total_samples}"
                    )
                    for k, v in corrupted_cache.items():
                        corrupted_dict[k] = torch.zeros(
                            (total_samples, *v.shape[1:]), 
                            dtype=dtype, 
                            device='cpu'
                        )
                        clean_dict[k] = torch.zeros(
                            (total_samples, *v.shape[1:]), 
                            dtype=dtype, 
                            device='cpu'
                        )
                    buffer_initialized = True

                bar.set_description(
                    "run_with_cache: filling buffer"
                )
                # Fill the buffer tensors
                start_idx = idx * batch_size
                end_idx = start_idx + corrupted_tokens.size(0)
                for k, v in corrupted_cache.items():
                    corrupted_dict[k][start_idx:end_idx] = v
                for k, v in clean_cache.items():
                    clean_dict[k][start_idx:end_idx] = v
                # Fill logit and prob diff tensors
                clean_logit_diffs[start_idx:end_idx] = get_logit_diff(
                    clean_logits, answer_tokens, per_prompt=True,
                    tokens=clean_tokens, tokenizer=model.tokenizer
                ).detach().cpu()
                corrupted_logit_diffs[start_idx:end_idx] = get_logit_diff(
                    corrupted_logits, answer_tokens, per_prompt=True,
                    tokens=clean_tokens, tokenizer=model.tokenizer
                ).detach().cpu()
                clean_prob_diffs[start_idx:end_idx] = get_prob_diff(
                    clean_logits, answer_tokens, per_prompt=True,
                    tokens=clean_tokens, tokenizer=model.tokenizer
                ).detach().cpu()
                corrupted_prob_diffs[start_idx:end_idx] = get_prob_diff(
                    corrupted_logits, answer_tokens, per_prompt=True,
                    tokens=clean_tokens, tokenizer=model.tokenizer
                ).detach().cpu()
        corrupted_cache = ActivationCache(
            {k: v.detach().clone().requires_grad_(requires_grad) for k, v in corrupted_dict.items()}, 
            model=model
        )
        clean_cache = ActivationCache(
            {k: v.detach().clone().requires_grad_(requires_grad) for k, v in clean_dict.items()}, 
            model=model
        )
        corrupted_cache.to('cpu')
        clean_cache.to('cpu')
        torch.set_grad_enabled(was_grad_enabled)
        model = model.train().requires_grad_(requires_grad)

        return CleanCorruptedCacheResults(
            dataset=self,
            corrupted_cache=corrupted_cache,
            clean_cache=clean_cache,
            corrupted_logit_diffs=corrupted_logit_diffs,
            clean_logit_diffs=clean_logit_diffs,
            corrupted_prob_diffs=corrupted_prob_diffs,
            clean_prob_diffs=clean_prob_diffs,
            center=center,
        )


class CleanCorruptedCacheResults:

    def __init__(
        self, 
        dataset: CleanCorruptedDataset,
        corrupted_cache: ActivationCache,
        clean_cache: ActivationCache,
        corrupted_logit_diffs: Float[Tensor, "batch"],
        clean_logit_diffs: Float[Tensor, "batch"],
        corrupted_prob_diffs: Float[Tensor, "batch"],
        clean_prob_diffs: Float[Tensor, "batch"],
        center: bool = True,
    ) -> None:
        self.dataset = dataset
        self.corrupted_cache = corrupted_cache
        self.clean_cache = clean_cache
        self.corrupted_logit_diffs = corrupted_logit_diffs
        self.clean_logit_diffs = clean_logit_diffs
        self.corrupted_prob_diffs = corrupted_prob_diffs
        self.clean_prob_diffs = clean_prob_diffs
        if center:
            self.center_logit_diffs()
        else:
            self.corrupted_logit_bias = 0
            self.clean_logit_bias = 0
        self.set_accuracy()

    def __str__(self) -> str:
        return (
            f"CleanCorruptedCacheResults(\n"
            f"  corrupted_logit_diff={self.corrupted_logit_diff:.2f},\n"
            f"  clean_logit_diff={self.clean_logit_diff:.2f},\n"
            f"  corrupted_prob_diff={self.corrupted_prob_diff:.2f},\n"
            f"  clean_prob_diff={self.clean_prob_diff:.2f},\n"
            f"  clean_accuracy={self.clean_accuracy:.1%},\n"
            f"  corrupted_accuracy={self.corrupted_accuracy:.1%},\n"
            f"  length={len(self.dataset)}"
            f")"
        )
    
    def set_accuracy(self):
        self.corrupted_logit_diff = torch.mean(self.corrupted_logit_diffs)
        self.clean_logit_diff = torch.mean(self.clean_logit_diffs)
        self.corrupted_prob_diff = torch.mean(self.corrupted_prob_diffs)
        self.clean_prob_diff = torch.mean(self.clean_prob_diffs)
        self.clean_accuracy = get_accuracy_from_logit_diffs(self.clean_logit_diffs)
        self.corrupted_accuracy = get_accuracy_from_logit_diffs(self.corrupted_logit_diffs)
    
    def center_logit_diffs(self):
        answer_tokens = self.dataset.answer_tokens
        self.corrupted_logit_diffs, self.corrupted_logit_bias = center_logit_diffs(
            self.corrupted_logit_diffs, answer_tokens
        )
        self.clean_logit_diffs, self.clean_logit_bias = center_logit_diffs(
            self.clean_logit_diffs, answer_tokens
        )


def get_dataset(
    model: HookedTransformer, 
    device: torch.device,
    n_pairs: Optional[int] = None,
    prompt_type: PromptType = PromptType.SIMPLE,
    comparison: Tuple[str, str] = ("positive", "negative"),
    scaffold: Optional[ReviewScaffold] = None,
) -> CleanCorruptedDataset:
    prompt_type = PromptType(prompt_type)
    if prompt_type in (
        PromptType.TREEBANK_TRAIN, PromptType.TREEBANK_TEST, PromptType.TREEBANK_DEV
    ):
        assert scaffold is not None
        return get_pickle_dataset(model, prompt_type, scaffold).to(device)
    prompts_dict, answers_dict = get_prompts(
        model, prompt_type
    )
    if n_pairs is None:
        n_pairs = min(
            len(answers_dict[comparison[0]]), 
            len(answers_dict[comparison[1]]), 
        )
    assert n_pairs <= len(answers_dict[comparison[0]])
    n_prompts = min(
        len(prompts_dict[comparison[0]]), 
        len(prompts_dict[comparison[1]]), 
    )
    batch_size = n_prompts * 2
    all_prompts = []
    answer_tokens = torch.empty(
        (batch_size, n_pairs, 2), 
        device=device, 
        dtype=torch.long
    )
    prompt_len = None
    for i in range(n_prompts):
        all_prompts.append(prompts_dict[comparison[0]][i])
        all_prompts.append(prompts_dict[comparison[1]][i])
        for pair_idx in range(n_pairs):
            answer_tokens[i * 2, pair_idx, 0] = model.to_single_token(answers_dict[comparison[0]][pair_idx])
            answer_tokens[i * 2, pair_idx, 1] = model.to_single_token(answers_dict[comparison[1]][pair_idx])
            answer_tokens[i * 2 + 1, pair_idx, 0] = model.to_single_token(answers_dict[comparison[1]][pair_idx])
            answer_tokens[i * 2 + 1, pair_idx, 1] = model.to_single_token(answers_dict[comparison[0]][pair_idx])
        if prompt_len is None:
            prompt_len = len(model.to_tokens(all_prompts[-1], prepend_bos=True))
        else:
            assert prompt_len == len(model.to_tokens(all_prompts[-1], prepend_bos=True))
    prompts_tokens: Float[Tensor, "batch pos"] = model.to_tokens(
        all_prompts, prepend_bos=True
    )
    clean_tokens = prompts_tokens.to(device)
    corrupted_tokens = model.to_tokens(
        all_prompts[1:] + [all_prompts[0]], prepend_bos=True
    ).to(device)
    assert model.tokenizer is not None
    assert (clean_tokens[:, -1] != model.tokenizer.bos_token_id).all(), (
        "Last token in prompt should not be BOS token, "
        "this suggests inconsistent prompt lengths."
    )
    
    return CleanCorruptedDataset(
        all_prompts=all_prompts, 
        answer_tokens=answer_tokens, 
        clean_tokens=clean_tokens, 
        corrupted_tokens=corrupted_tokens,
        tokenizer=model.tokenizer,
    )


def get_pickle_dataset(
    model: HookedTransformer,
    prompt_type: PromptType,
    scaffold: ReviewScaffold
):
    return load_pickle(
        prompt_type.value + '_' + scaffold.value,
        model
    )


def get_onesided_datasets(
    model: HookedTransformer, 
    device: torch.device,
    n_answers: int = 1,
    prompt_type: PromptType = PromptType.SIMPLE,
    dataset_sentiments: list = ["positive", "negative"],
    answer_sentiment: str = "positive",
):
    '''
    answer_tokens:
        list of the token (ie an integer) corresponding to each answer, 
        in the format (correct_token, incorrect_token)
    '''
    prompt_str_dict, answers_dict = get_prompts(
        model, prompt_type
    )
    prompts_dict = {
        key: model.to_tokens(values, prepend_bos=True)
        for key, values in prompt_str_dict.items()
    }
    for prompt_k, prompt_v in prompts_dict.items():
        assert prompt_v.shape[1] == prompts_dict["positive"].shape[1], (
            f"{prompt_k} prompt has seq len {prompt_v.shape[1]} "
            f"while positive has seq len {prompts_dict['positive'].shape[1]}"
        )
    
    n_prompts = min([prompts_dict[s].shape[0] for s in dataset_sentiments])
    prompt_return_dict = {
        s:prompts_dict[s][:n_prompts] for s in dataset_sentiments
    }

    answers = answers_dict[answer_sentiment]
    assert len(answers) >= n_answers
    answers = torch.tensor([int(model.to_single_token(a)) for a in answers[:n_answers]])
    answer_list = [answers for _ in range(n_prompts)]
    answer_tokens = torch.stack(answer_list, dim=0).to(device)

    return (
        prompt_return_dict, 
        answer_tokens
    )

def get_ccs_dataset(
    model: HookedTransformer,
    device: torch.device,
    prompt_type: PromptType = PromptType.CLASSIFICATION_4,
    pos_answers: List[str] = [" Positive"],
    neg_answers: List[str] = [" Negative"],
) -> Tuple[
    Float[Tensor, "batch q_and_a"], 
    Float[Tensor, "batch q_and_a"], 
    List[List[str]],
    List[List[str]],
    Int[Tensor, "batch"],
    Bool[Tensor, "batch"],
]:
    clean_corrupt_data: CleanCorruptedDataset = get_dataset(
        model, device, n_pairs=1, prompt_type=prompt_type, 
    )
    answer_tokens: Int[Tensor, "batch 2"] = clean_corrupt_data.answer_tokens.squeeze(1)
    possible_answers = answer_tokens[0]
    possible_answers_repeated: Int[Tensor, "batch 2"] = einops.repeat(
        possible_answers, "answers -> batch answers", 
        batch=clean_corrupt_data.clean_tokens.shape[0]
    )
    # concatenate clean_tokens and answer_tokens along new dimension
    pos_tokens: Float[Tensor, "batch q_and_a"] = torch.cat(
        (clean_corrupt_data.clean_tokens, possible_answers_repeated[:, :1]), dim=1
    )
    neg_tokens: Float[Tensor, "batch q_and_a"] = torch.cat(
        (clean_corrupt_data.clean_tokens, possible_answers_repeated[:, -1:]), dim=1
    )
    gt_labels: Int[Tensor, "batch"] = (
        pos_tokens[:, -1] == answer_tokens[:, 0]
    ).to(torch.int64) # 1 for positive, 0 for negative
    truncated: Bool[Tensor, "batch"] = torch.zeros(
        gt_labels.shape[0], device=device, dtype=torch.bool
    )
    pos_prompts = [
        [prompt, answer] 
        for prompt in clean_corrupt_data.all_prompts 
        for answer in pos_answers
    ]
    neg_prompts = [
        [prompt, answer]
        for prompt in clean_corrupt_data.all_prompts
        for answer in neg_answers
    ]
    assert len(pos_prompts) == len(pos_tokens)
    return neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, truncated
