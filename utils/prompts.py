import yaml
from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Dict, List, Tuple, Union
import einops
from enum import Enum


def filter_words_by_length(model, words: list, length: int, verbose=False) -> list:
    if verbose:
        print("Filtering words by length")
    new_words = []
    for a in words:
        tkn = model.to_str_tokens(a, prepend_bos=False)
        if len(tkn) == length:
            new_words.append(a)
    if verbose:
        print(f"Count of words: {len(words)}")

    return new_words


def truncate_words_by_length(model, words: list, length: int, verbose=False) -> list:
    if verbose:
        print("Truncating words by length")
    new_words = []
    for a in words:
        tkn = model.to_str_tokens(a, prepend_bos=False)
        new_words.append(tkn[:length])
    return new_words


class CircularList(list):
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
        self, key: str, model: HookedTransformer, 
        filter_length: int = None, truncate_length: int = None,
        prepend_space: bool = True, verbose: bool = False,
    ) -> CircularList:
        words: list = self._prompts_dict[key]
        if prepend_space:
            words = [" " + word.strip() for word in words]
        if filter_length is not None:
            words = filter_words_by_length(model, words, filter_length, verbose=verbose)
        if truncate_length is not None:
            words = truncate_words_by_length(model, words, truncate_length, verbose=verbose)
        return CircularList(words)


class PromptType(Enum):
    SIMPLE = "simple"
    COMPLETION = "completion"
    COMPLETION_2 = "completion_2"
    CLASSIFICATION = "classification"
    CLASSIFICATION_2 = "classification_2"
    CLASSIFICATION_3 = "classification_3"
    CLASSIFICATION_4 = "classification_4"
    RES_CLASS_1 = "res_class_1"
    SIMPLE_MOOD = "simple_mood"
    SIMPLE_ADVERB = "simple_adverb"
    SIMPLE_FRENCH = "simple_french"
    PROPER_NOUNS = "proper_nouns"
    MEDICAL = "medical"


prompt_config = PromptsConfig()


def get_prompts(
    model: HookedTransformer,
    prompt_type: str = "simple", 
) -> Tuple[Dict[str, CircularList[str]], Dict[str, CircularList[str]]]:
    
    pos_prompts: CircularList[str]
    neg_prompts: CircularList[str]
    neutral_prompts: CircularList[str]
    pos_answers: CircularList[str] = prompt_config.get("positive_answer_tokens", model, filter_length=1)
    neg_answers: CircularList[str] = prompt_config.get("negative_answer_tokens", model, filter_length=1)
    positive_adjectives: CircularList[str] = prompt_config.get("positive_core_adjectives", model, filter_length=1)
    negative_adjectives: CircularList[str] = prompt_config.get("negative_core_adjectives", model, filter_length=1)
    neutral_adjectives: CircularList[str] = prompt_config.get("neutral_core_adjectives", model, filter_length=1)
    prompt_type = PromptType(prompt_type)

    if prompt_type == PromptType.SIMPLE:
        pos_prompts = [
            f"I thought this movie was{positive_adjectives[i]}, I loved it. \nConclusion: This movie is" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{negative_adjectives[i]}, I hated it. \nConclusion: This movie is" for i in range(len(negative_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{neutral_adjectives[i]}, I watched it. \nConclusion: This movie is" for i in range(len(neutral_adjectives))
        ]
    elif prompt_type == PromptType.SIMPLE_MOOD:
        walking_synonyms: CircularList[str]  = prompt_config.get("walk_synonyms", model)
        positive_feelings: CircularList[str] = prompt_config.get("positive_feelings", model)
        negative_feelings: CircularList[str] = prompt_config.get("negative_feelings", model)
        positive_adverbs: CircularList[str] = prompt_config.get("positive_adverbs", model)
        negative_adverbs: CircularList[str] = prompt_config.get("negative_adverbs", model)
        n_prompts = min(len(positive_adverbs), len(walking_synonyms), len(positive_feelings), len(negative_adverbs), len(negative_feelings))
        pos_prompts = [
            f"The traveller was{positive_adverbs[i]}{walking_synonyms[i % len(walking_synonyms)]} to their destination. Upon arrival, they felt{positive_feelings[i]}. \nConclusion: The traveller is" for i in range(n_prompts)
        ]
        neg_prompts = [
            f"The traveller was{negative_adverbs[i]}{walking_synonyms[i% len(walking_synonyms)]} to their destination. Upon arrival, they felt{negative_feelings[i]}. \nConclusion: The traveller is" for i in range(n_prompts)
        ]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_moods", model)
        neg_answers = prompt_config.get("negative_moods", model)
    elif prompt_type == PromptType.SIMPLE_ADVERB:
        positive_adverbs = prompt_config.get("positive_adverbs", model)
        negative_adverbs = prompt_config.get("negative_adverbs", model)
        n_prompts = min(len(positive_adverbs), len(negative_adverbs))
        pos_prompts = [
            f"The traveller{positive_adverbs[i]} walked to their destination. The traveller felt very" for i in range(n_prompts)
        ]
        neg_prompts = [
            f"The traveller{negative_adverbs[i]} walked to their destination. The traveller felt very" for i in range(n_prompts)
        ]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_moods", model)
        neg_answers = prompt_config.get("negative_moods", model)
    elif prompt_type == PromptType.SIMPLE_FRENCH:
        positive_french = prompt_config.get("positive_french", model)
        negative_french = prompt_config.get("negative_french", model)
        n_prompts = min(len(positive_french), len(negative_french))
        pos_prompts = [
            f"Je pensais que ce film était{positive_french[i]}, je l'ai adoré. \nConclusion: Ce film est" for i in range(n_prompts)
        ]
        neg_prompts = [
            f"Je pensais que ce film était{negative_french[i]}, je l'ai haï. \nConclusion: Ce film est" for i in range(n_prompts)
        ]
        neutral_prompts = None
        pos_answers = prompt_config.get("positive_french_answers", model, truncate_length=1)
        neg_answers = prompt_config.get("negative_french_answers", model, truncate_length=1)
    elif prompt_type == PromptType.PROPER_NOUNS:
        positive_proper = prompt_config.get("positive_proper_nouns", model)
        negative_proper = prompt_config.get("negative_proper_nouns", model)
        n_prompts = min(len(positive_proper), len(negative_proper))
        pos_prompts = [
            f"When I hear the name{positive_proper[i]}, I feel very" for i in range(n_prompts)
        ]
        neg_prompts = [
            f"When I hear the name{negative_proper[i]}, I feel very" for i in range(n_prompts)
        ]
        neutral_prompts = None
    elif prompt_type == PromptType.MEDICAL:
        positive_medical = prompt_config.get("positive_medical", model)
        negative_medical = prompt_config.get("negative_medical", model)
        n_prompts = min(len(positive_medical), len(negative_medical))
        pos_prompts = [
            f"I heard the doctor use the word{positive_medical[i]} and I felt very" for i in range(n_prompts)
        ]
        neg_prompts = [
            f"I heard the doctor use the word{negative_medical[i]} and I felt very" for i in range(n_prompts)
        ]
        neutral_prompts = None
    elif prompt_type == PromptType.COMPLETION:
        pos_prompts = [
            f"I thought this movie was{positive_adjectives[i]}, I loved it. The acting was{positive_adjectives[i + 1]}, the plot was{positive_adjectives[i + 2]}, and overall the movie was just very" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{negative_adjectives[i]}, I hated it. The acting was{negative_adjectives[i + 1]}, the plot was{negative_adjectives[i + 2]}, and overall the movie was just very" for i in range(len(negative_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{neutral_adjectives[i]}, I watched it. The acting was{neutral_adjectives[i + 1]}, the plot was{neutral_adjectives[i + 2]}, and overall the movie was just very" for i in range(len(neutral_adjectives))
        ]
    elif prompt_type == PromptType.COMPLETION_2:
        pos_prompts = [
            f"I thought this movie was{positive_adjectives[i]}, I loved it. The acting was{positive_adjectives[i + 1]}, the plot was{positive_adjectives[i + 2]}, and overall it was just very good. I felt it was" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{negative_adjectives[i]}, I hated it. The acting was{negative_adjectives[i + 1]}, the plot was{negative_adjectives[i + 2]}, and overall it was just very bad. I felt it was" for i in range(len(positive_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{neutral_adjectives[i]}, I watched it. The acting was{neutral_adjectives[i + 1]}, the plot was{neutral_adjectives[i + 2]}, and overall it was just very average. I felt it was" for i in range(len(positive_adjectives))
        ]
    elif prompt_type == PromptType.CLASSIFICATION:
        pos_prompts = [
            f"Review Text: 'I thought this movie was{positive_adjectives[i]}, I loved it. The acting was{positive_adjectives[i + 1]}, the plot was{positive_adjectives[i + 2]}, and overall the movie was just very good.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: 'I thought this movie was{negative_adjectives[i]}, I hated it. The acting was{negative_adjectives[i + 1]}, the plot was{negative_adjectives[i + 2]}, and overall the movie was just very bad.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: 'I thought this movie was{neutral_adjectives[i]}, I watched it. The acting was{neutral_adjectives[i + 1]}, the plot was{neutral_adjectives[i + 2]}, and overall the movie was just very average.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.CLASSIFICATION_2:
        pos_prompts = [
            f"Review Text: I thought this movie was{positive_adjectives[i]}, I loved it. The acting was{positive_adjectives[i + 1]}, the plot was{positive_adjectives[i + 2]}, and overall the movie was just very good. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: I thought this movie was{negative_adjectives[i]}, I hated it. The acting was{negative_adjectives[i + 1]}, the plot was{negative_adjectives[i + 2]}, and overall the movie was just very bad. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: I thought this movie was{neutral_adjectives[i]}, I watched it. The acting was{neutral_adjectives[i + 1]}, the plot was{neutral_adjectives[i + 2]}, and overall the movie was just very average. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.CLASSIFICATION_3:
        pos_prompts = [
            f"Review Text: I thought this movie was{positive_adjectives[i]}, I loved it. The acting was{positive_adjectives[i + 1]}, the plot was{positive_adjectives[i + 2]}, and overall the movie was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: I thought this movie was{negative_adjectives[i]}, I hated it. The acting was{negative_adjectives[i + 1]}, the plot was{negative_adjectives[i + 2]}, and overall the movie was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: I thought this movie was{neutral_adjectives[i]}, I watched it. The acting was{neutral_adjectives[i + 1]}, the plot was{neutral_adjectives[i + 2]}, and overall the movie was just very average. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.CLASSIFICATION_4:
        pos_prompts = [
            f"I thought this movie was{positive_adjectives[i]}, I loved it. The acting was{positive_adjectives[i + 1]}, the plot was{positive_adjectives[i + 2]}, and overall the movie was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"I thought this movie was{negative_adjectives[i]}, I hated it. The acting was{negative_adjectives[i + 1]}, the plot was{negative_adjectives[i + 2]}, and overall the movie was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"I thought this movie was{neutral_adjectives[i]}, I watched it. The acting was{neutral_adjectives[i + 1]}, the plot was{neutral_adjectives[i + 2]}, and overall the movie was just very average. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.RES_CLASS_1:
        pos_prompts = [
            f"This restaurant was{positive_adjectives[i]}, I loved it. The food was{positive_adjectives[i + 1]}, and the service was{positive_adjectives[i + 2]}. Overall it was just great. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"This restaurant was{negative_adjectives[i]}, I hated it. The food was{negative_adjectives[i + 1]}, and the service was{negative_adjectives[i + 2]}. Overall it was just awful. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = None
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

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


def get_dataset(
    model: HookedTransformer, 
    device: torch.device,
    n_pairs: int = 1,
    prompt_type: str = "simple",
    comparison: Tuple[str, str] = ("positive", "negative"),
) -> Tuple[
    List[str], # all_prompts
    Float[Tensor, "batch n_pairs 2"], # answer tokens
    Float[Tensor, "batch pos"], # clean tokens
    Float[Tensor, "batch pos"], # corrupted tokens
]:
    '''
    answer_tokens:
        list of the token (ie an integer) corresponding to each answer, 
        in the format (correct_token, incorrect_token)
    '''
    prompt_type = PromptType(prompt_type)
    prompts_dict, answers_dict = get_prompts(
        model, prompt_type
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
    for i in range(n_prompts):
        all_prompts.append(prompts_dict[comparison[0]][i])
        all_prompts.append(prompts_dict[comparison[1]][i])
        for pair_idx in range(n_pairs):
            answer_tokens[i * 2, pair_idx, 0] = model.to_single_token(answers_dict[comparison[0]][pair_idx])
            answer_tokens[i * 2, pair_idx, 1] = model.to_single_token(answers_dict[comparison[1]][pair_idx])
            answer_tokens[i * 2 + 1, pair_idx, 0] = model.to_single_token(answers_dict[comparison[1]][pair_idx])
            answer_tokens[i * 2 + 1, pair_idx, 1] = model.to_single_token(answers_dict[comparison[0]][pair_idx])
    prompts_tokens: Float[Tensor, "batch pos"] = model.to_tokens(
        all_prompts, prepend_bos=True
    )
    clean_tokens = prompts_tokens.to(device)
    corrupted_tokens = model.to_tokens(
        all_prompts[1:] + [all_prompts[0]], prepend_bos=True
    ).to(device)
    
    return (
        all_prompts, answer_tokens, clean_tokens, corrupted_tokens
    )

def get_onesided_datasets(
    model: HookedTransformer, 
    device: torch.device,
    n_answers: int = 1,
    prompt_type: str = "simple",
    dataset_sentiments: list = ["positive", "negative"],
    answer_sentiment: str = "positive",
):
    '''
    answer_tokens:
        list of the token (ie an integer) corresponding to each answer, 
        in the format (correct_token, incorrect_token)
    '''
    assert prompt_type in ["simple", "completion", "classification"]
    
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

    answers = answers or answers_dict[answer_sentiment]
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
    prompt_type: str = "classification_4",
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
    all_prompts, answer_tokens, clean_tokens, _ = get_dataset(
        model, device, n_pairs=1, prompt_type=prompt_type, 
        pos_answers=pos_answers, neg_answers=neg_answers,
    )
    answer_tokens: Int[Tensor, "batch 2"] = answer_tokens.squeeze(1)
    clean_tokens.shape
    possible_answers = answer_tokens[0]
    possible_answers_repeated: Int[Tensor, "batch 2"] = einops.repeat(
        possible_answers, "answers -> batch answers", batch=clean_tokens.shape[0]
    )
    # concatenate clean_tokens and answer_tokens along new dimension
    pos_tokens: Float[Tensor, "batch q_and_a"] = torch.cat(
        (clean_tokens, possible_answers_repeated[:, :1]), dim=1
    )
    neg_tokens: Float[Tensor, "batch q_and_a"] = torch.cat(
        (clean_tokens, possible_answers_repeated[:, -1:]), dim=1
    )
    gt_labels: Int[Tensor, "batch"] = (
        pos_tokens[:, -1] == answer_tokens[:, 0]
    ).to(torch.int64) # 1 for positive, 0 for negative
    truncated: Bool[Tensor, "batch"] = torch.zeros(
        gt_labels.shape[0], device=device, dtype=torch.bool
    )
    pos_prompts = [
        [prompt, answer] 
        for prompt in all_prompts 
        for answer in pos_answers
    ]
    neg_prompts = [
        [prompt, answer]
        for prompt in all_prompts
        for answer in neg_answers
    ]
    assert len(pos_prompts) == len(pos_tokens)
    return neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, truncated


def embed_and_mlp0(
    tokens: Union[str, List[str], Int[Tensor, "batch pos"]],
    transformer: HookedTransformer,
):
    if isinstance(tokens, str):
        tokens = transformer.to_tokens(tokens)
    elif isinstance(tokens, list) and isinstance(tokens[0], str):
        tokens = transformer.to_tokens(tokens)
    block0 = transformer.blocks[0]
    resid_mid = transformer.embed(tokens)
    mlp_out = block0.mlp((resid_mid))
    resid_post = resid_mid + mlp_out
    return block0.ln2(resid_post)
