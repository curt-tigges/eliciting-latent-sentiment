from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from jaxtyping import Float
from typing import Tuple
import einops

pos_adj = [
    ' perfect', ' fantastic',' delightful',' cheerful',' marvelous',' good',' remarkable',' wonderful',
    ' fabulous',' outstanding',' awesome',' exceptional',' incredible',' extraordinary',
    ' amazing',' lovely',' brilliant',' charming',' terrific',' superb',' spectacular',' great',' splendid',
    ' beautiful',' joyful',' positive',#' excellent'
    ]

neg_adj = [
    ' dreadful',' bad',' dull',' depressing',' miserable',' tragic',' nasty',' inferior',' horrific',' terrible',
    ' ugly',' disgusting',' disastrous',' horrendous',' annoying',' boring',' offensive',' frustrating',' wretched',' dire',
    ' awful',' unpleasant',' horrible',' mediocre',' disappointing',' inadequate'
    ]


pos_tokens = [" amazing", " good", " pleasant", " wonderful", " great"]
neg_tokens = [" terrible", " bad", " unpleasant", " horrendous", " awful"]


def get_adjective(adjective_list, index):
    """ Returns the adjective at the given index, looping around the list if necessary
    """
    return adjective_list[index % len(adjective_list)]


def remove_pythia_double_token_words(model, words: list = None) -> list:
    """ Removes words that are double tokens in Pythia
    """
    print("Using Pythia model")
    new_words = []
    for a in words:
        tkn = model.to_str_tokens(a)
        if len(tkn)==2:
            new_words.append(a)

    print(f"Count of words: {len(words)}")

    return new_words


def get_prompts(
        prompt_type: str = "simple", 
        positive_adjectives: list = pos_adj, 
        negative_adjectives: list = neg_adj
) -> Tuple[list, list]:
    assert prompt_type in ["simple", "completion", "classification"]
    pos_prompts, neg_prompts = [], []
    if prompt_type == "simple":
        pos_prompts = [
            f"I thought this movie was{positive_adjectives[i]}, I loved it. \nConclusion: This movie is" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{negative_adjectives[i]}, I hated it. \nConclusion: This movie is" for i in range(len(negative_adjectives))
        ]
    elif prompt_type == "completion":
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very" for i in range(len(negative_adjectives))
        ]
    elif prompt_type == "classification":
        pos_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]

    return pos_prompts, neg_prompts


def get_dataset(
    model: HookedTransformer, 
    device: torch.device,
    n_pairs: int = 1,
    prompt_type: str = "simple",
    pos_answers: list = pos_tokens,
    neg_answers: list = neg_tokens,
) -> Tuple[
    Float[Tensor, "batch pos"],
    Float[Tensor, "batch n_pairs 2"],
    Float[Tensor, "batch pos"],
    Float[Tensor, "batch pos"],
]:
    '''
    answer_tokens:
        list of the token (ie an integer) corresponding to each answer, 
        in the format (correct_token, incorrect_token)
    '''
    assert n_pairs <= len(pos_tokens)
    assert prompt_type in ["simple", "completion", "classification"]
    
    if "pythia" in model.cfg.model_name:
        positive_adjectives = remove_pythia_double_token_words(model, pos_adj)
        negative_adjectives = remove_pythia_double_token_words(model, neg_adj)
    else:
        positive_adjectives = pos_adj
        negative_adjectives = neg_adj

    pos_prompts, neg_prompts = get_prompts(prompt_type, positive_adjectives, negative_adjectives)

    batch_size = len(pos_prompts) * 2
    all_prompts = []
    answer_tokens = torch.empty(
        (batch_size, n_pairs, 2), 
        device=device, 
        dtype=torch.long
    )
    for i in range(min(len(pos_prompts), len(neg_prompts))):

        all_prompts.append(pos_prompts[i])
        all_prompts.append(neg_prompts[i])
        
        for pair_idx in range(n_pairs):
            pos_token = model.to_single_token(pos_answers[pair_idx])
            neg_token = model.to_single_token(neg_answers[pair_idx])
            answer_tokens[i * 2, pair_idx, 0] = pos_token
            answer_tokens[i * 2, pair_idx, 1] = neg_token
            answer_tokens[i * 2 + 1, pair_idx, 0] = neg_token
            answer_tokens[i * 2 + 1, pair_idx, 1] = pos_token
    
    prompts_tokens: Float[Tensor, "batch pos"] = model.to_tokens(all_prompts, prepend_bos=True)
    
    clean_tokens = prompts_tokens.to(device)
    corrupted_tokens = model.to_tokens(all_prompts[1:] + [all_prompts[0]], prepend_bos=True).to(device)
    
    return (
        all_prompts, answer_tokens, clean_tokens, corrupted_tokens
    )

def get_logit_diff(
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
        return left_logits - right_logits
    return (left_logits - right_logits).mean()

def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    flipped_logit_diff: float,
    clean_logit_diff: float,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 
    0 when performance is same as on flipped input, and 
    1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return (
        (patched_logit_diff - flipped_logit_diff) / 
        (clean_logit_diff  - flipped_logit_diff)
    ).item()