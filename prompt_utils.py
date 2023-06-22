from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from jaxtyping import Float
from typing import Tuple

positive_adjectives = [
    ' perfect', ' fantastic',' delightful',' cheerful',' marvelous',' good',' remarkable',' wonderful',
    ' fabulous',' outstanding',' awesome',' exceptional',' incredible',' extraordinary',
    ' amazing',' lovely',' brilliant',' charming',' terrific',' superb',' spectacular',' great',' splendid',
    ' beautiful',' joyful',' positive',' excellent'
    ]

negative_adjectives = [
    ' dreadful',' bad',' dull',' depressing',' miserable',' tragic',' nasty',' inferior',' horrific',' terrible',
    ' ugly',' disgusting',' disastrous',' horrendous',' annoying',' boring',' offensive',' frustrating',' wretched',' dire',
    ' awful',' unpleasant',' horrible',' mediocre',' disappointing',' inadequate'
    ]

pos_prompts = [
    f"I thought this movie was{positive_adjectives[i]}, I loved it. \nConclusion: This movie is" for i in range(len(positive_adjectives)-1)
]
neg_prompts = [
    f"I thought this movie was{negative_adjectives[i]}, I hated it. \nConclusion: This movie is" for i in range(len(negative_adjectives)-1)
]

def get_dataset(
    model: HookedTransformer, device: torch.device
) -> Tuple[
    Float[Tensor, "batch pos"],
    Float[Tensor, "batch 2"],
    Float[Tensor, "batch pos"],
    Float[Tensor, "batch pos"],
]:
    '''
    answer_tokens:
        list of the token (ie an integer) corresponding to each answer, 
        in the format (correct_token, incorrect_token)
    '''
    all_prompts = []
    answer_tokens = []
    for i in range(len(pos_prompts)-1):

        all_prompts.append(pos_prompts[i])
        all_prompts.append(neg_prompts[i])
        
        answer_tokens.append(
            (
                model.to_single_token(" amazing"),
                model.to_single_token(" terrible"),
            )
        )

        answer_tokens.append(
            (
                model.to_single_token(" terrible"),
                model.to_single_token(" amazing"),
            )
        )
    answer_tokens: Float[Tensor, "batch 2"] = torch.tensor(
        answer_tokens
    ).to(device)
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