from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Tuple, Union
import einops
from enum import Enum

# deprecated for now--those with the weakest positive response are eliminated
# pos_adj = [
#     ' perfect', ' fantastic',' delightful',' cheerful',' marvelous',' good',' remarkable',' wonderful',
#     ' fabulous',' outstanding',' awesome',' exceptional',' incredible',' extraordinary',
#     ' amazing',' lovely',' brilliant',' charming',' terrific',' superb',' spectacular',' great',' splendid',
#     ' beautiful',' joyful',' positive',#' excellent'
#     ]

pos_adj = [
    ' perfect', ' fantastic',' marvelous',' good',' remarkable',' wonderful',
    ' fabulous',' outstanding',' awesome',' exceptional',' incredible',' extraordinary',
    ' amazing',' lovely',' brilliant',' terrific',' superb',' spectacular',' great',
    ' beautiful'
    ]

# neg_adj = [
#     ' dreadful',' bad',' dull',' depressing',' miserable',' tragic',' nasty',' inferior',' horrific',' terrible',
#     ' ugly',' disgusting',' disastrous',' horrendous',' annoying',' boring',' offensive',' frustrating',' wretched',' dire',
#     ' awful',' unpleasant',' horrible',' mediocre',' disappointing',' inadequate'
#     ]

neg_adj = [
    ' dreadful',' bad',' miserable',' horrific',' terrible',
    ' disgusting',' disastrous',' horrendous',' offensive',' wretched',
    ' awful',' unpleasant',' horrible',' mediocre',' disappointing'
    ]
neutral_adj = [
    ' average',' normal',' standard',' typical',' common',' ordinary',
    ' regular',' usual',' familiar',' generic',' conventional',
    ' fine', ' okay', ' ok', ' decent', ' fair', ' satisfactory', 
    ' adequate', ' alright',

]
neutral_core_adj = [
    ' ok', ' okay', ' OK', ' alright', ' fine', ' neutral', ' acceptable', ' fair', 
    ' standard', ' reasonable', ' average'
]

pos_tokens = [" great"," amazing", " awesome", " good", " perfect"]
neg_tokens = [" terrible", " awful", " bad", " horrible", " disgusting"]
neutral_tokens = [
    " OK", " mediocre", "fine", "okay", "alright", "ok", "decent",
]

pos_adverbs = [
    # 3 tokens in gpt2-small with prepended space
    'excitedly',
    'joyfully',
    'gleefully',
    'merrily',
    'cheerfully',
    'delightedly',
    'joyously',
    'triumphantly',
    'blissfully',
    'euphorically',
    'radiantly',
    'buoyantly',
    'jokingly',
    'humorously',
    'comically',
    'amusingly',
    'playfully',
    'sportively',
    'lightheartedly',
    'gleefully'
]
neg_adverbs = [
    'fearfully',
    'frightfully',
    'scaredly',
    'scarily',
    'fearingly',
    'fearfully',
    'frightenedly',
    'horrendously',
    'horrifically',
    'horrifyingly',
    'dreadfully',
    'grimly',
    'gruesomely',
    'morbidly',
    'disgustingly',
    'vilely',
    'obscenely',
    'odiously',
    'outrageously',
    'monstrously',
    'devilishly',
    'demonically',
    'sadistically',
    'viciously',
    'vilely',
    'wickedly',
    'perversely',
    'corruptly',
    'sinfully',
    'unjustly',
    'wickedly',
    'shamefully',
    'disgracefully',
    'scandalously',
    'shamelessly'
]
walk_synonyms = [
    'strolling',
    'marching',
    'striding',
    'hiking',
    'trekking',
    'ambling',
    'roaming',
    'wandering',
    'trudging',
    'meandering',
    'pacing',
    'tramping',
    'slogging',
    'treading',
    'rambling'
]
pos_feelings = [
    # 2 tokens in gpt2-small with prepended space
    'happy',
    'glad',
    'cheerful',
    'joyful',
    'delighted',
    'pleased',
    'content',
    'satisfied',
    'thankful',
    'merry',
    'ecstatic',
    'thrilled',
    'sunny',
    'radiant',
    'upbeat',
    'vibrant',
    'optimistic',
    'grateful',
    'inspired',
    'festive',
    'blessed',
    'elevated',
    'sparkling',
    'lively',
    'enthusiastic',
    'ecstatic',
    'eager',
    'radiant',
    'animated'

]
neg_feelings = [
    'sad',
    'unhappy',
    'depressed',
    'miserable',
    'down',
    'desolate',
    'wretched',
    'gloomy',
    'dismal',
    'melancholy',
    'blue',
    'miserable',
    'pessimistic',
    'sour',
    'cynical',
    'disgruntled',
    'unhappy',
    'dissatisfied',
    'restless',
    'uncomfortable',
    'uneasy',
    'worried',
    'disturbed',
    'bothered',
    'distressed',
    'agitated',
    'restless',
    'anxious',
    'nervous'
]

def get_adjective(adjective_list, index):
    """ Returns the adjective at the given index, looping around the list if necessary
    """
    return adjective_list[index % len(adjective_list)]


def remove_pythia_double_token_words(model, words: list = None, verbose=True) -> list:
    """ Removes words that are double tokens in Pythia
    """
    print("Using Pythia model")
    new_words = []
    for a in words:
        tkn = model.to_str_tokens(a)
        if len(tkn)==2:
            new_words.append(a)
    if verbose:
        print(f"Count of words: {len(words)}")

    return new_words


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


def get_prompts(
    prompt_type: str = "simple", 
    positive_adjectives: list = pos_adj, 
    negative_adjectives: list = neg_adj,
    neutral_adjectives: list = neutral_adj,
    positive_adverbs: list = pos_adverbs,
    negative_adverbs: list = neg_adverbs,
    walking_synonyms: list = walk_synonyms,
    positive_feelings: list = pos_feelings,
    negative_feelings: list = neg_feelings,
) -> Tuple[list, list]:
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
        n_prompts = min(len(positive_adverbs), len(walking_synonyms), len(positive_feelings), len(negative_adverbs), len(negative_feelings))
        pos_prompts = [
            f"The traveller was {positive_adverbs[i]} {walking_synonyms[i % len(walking_synonyms)]} to their destination. Upon arrival, they felt {positive_feelings[i]}. \nConclusion: The traveller is" for i in range(n_prompts)
        ]
        neg_prompts = [
            f"The traveller was {negative_adverbs[i]} {walking_synonyms[i% len(walking_synonyms)]} to their destination. Upon arrival, they felt {negative_feelings[i]}. \nConclusion: The traveller is" for i in range(n_prompts)
        ]
        neutral_prompts = None
    elif prompt_type == PromptType.COMPLETION:
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very" for i in range(len(negative_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very" for i in range(len(neutral_adjectives))
        ]
    elif prompt_type == PromptType.COMPLETION_2:
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall it was just very good. I felt it was" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall it was just very bad. I felt it was" for i in range(len(positive_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall it was just very average. I felt it was" for i in range(len(positive_adjectives))
        ]
    elif prompt_type == PromptType.CLASSIFICATION:
        pos_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.CLASSIFICATION_2:
        pos_prompts = [
            f"Review Text: I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.CLASSIFICATION_3:
        pos_prompts = [
            f"Review Text: I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.CLASSIFICATION_4:
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == PromptType.RES_CLASS_1:
        pos_prompts = [
            f"This restaurant was{get_adjective(positive_adjectives, i)}, I loved it. The food was{get_adjective(positive_adjectives, i+1)}, and the service was{get_adjective(positive_adjectives, i+2)}. Overall it was just great. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"This restaurant was{get_adjective(negative_adjectives, i)}, I hated it. The food was{get_adjective(negative_adjectives, i+1)}, and the service was{get_adjective(negative_adjectives, i+2)}. Overall it was just awful. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = None
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    return pos_prompts, neg_prompts, neutral_prompts


def get_dataset(
    model: HookedTransformer, 
    device: torch.device,
    n_pairs: int = 1,
    prompt_type: str = "simple",
    pos_answers: list = None,
    neg_answers: list = None,
    neutral_answers: list = neutral_tokens,
    comparison: Tuple[str, str] = ("positive", "negative"),
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
    prompt_type = PromptType(prompt_type)
    # FIXME: the below should be moved to a separate function like get_prompts()
    if prompt_type == PromptType.SIMPLE_MOOD:
        pos_answers = [" happy", "excited", " delighted", " pleased", "smiling", " satisfied", " glad"]
        neg_answers = [" sad", " scared", " terrible", " terrified", " morbid", " guilty", " suffering"]
    elif pos_answers is None or neg_answers is None:
        pos_answers = pos_tokens
        neg_answers = neg_tokens
    assert n_pairs <= len(pos_answers)

    
    if "pythia" in model.cfg.model_name:
        positive_adjectives = remove_pythia_double_token_words(model, pos_adj)
        negative_adjectives = remove_pythia_double_token_words(model, neg_adj)
        neutral_adjectives = remove_pythia_double_token_words(
            model, neutral_adj
        )
        pos_answers = remove_pythia_double_token_words(model, pos_answers)
        neg_answers = remove_pythia_double_token_words(model, neg_answers)
        neutral_answers = remove_pythia_double_token_words(model, neutral_answers)
    else:
        positive_adjectives = pos_adj
        negative_adjectives = neg_adj
        neutral_adjectives = neutral_adj

    pos_prompts, neg_prompts, neutral_prompts = get_prompts(
        prompt_type, positive_adjectives, negative_adjectives, neutral_adjectives
    )
    prompts_dict = {
        "positive": pos_prompts,
        "negative": neg_prompts,
        "neutral": neutral_prompts,
    }
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
            pos_token = model.to_single_token(pos_answers[pair_idx])
            neg_token = model.to_single_token(neg_answers[pair_idx])
            neut_token = model.to_single_token(neutral_answers[pair_idx])
            tokens_dict = {
                'positive': pos_token, 
                'negative': neg_token, 
                'neutral': neut_token,
            }
            answer_tokens[i * 2, pair_idx, 0] = tokens_dict[comparison[0]]
            answer_tokens[i * 2, pair_idx, 1] = tokens_dict[comparison[1]]
            answer_tokens[i * 2 + 1, pair_idx, 0] = tokens_dict[comparison[1]]
            answer_tokens[i * 2 + 1, pair_idx, 1] = tokens_dict[comparison[0]]
    
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
    answers: list = None,
    positive_adjectives: list = pos_adj, 
    negative_adjectives: list = neg_adj,
    neutral_adjectives: list = neutral_core_adj,
):
    '''
    answer_tokens:
        list of the token (ie an integer) corresponding to each answer, 
        in the format (correct_token, incorrect_token)
    '''
    assert n_answers <= len(pos_tokens)
    assert prompt_type in ["simple", "completion", "classification"]
    
    if "pythia" in model.cfg.model_name:
        positive_adjectives = remove_pythia_double_token_words(
            model, positive_adjectives
        )
        negative_adjectives = remove_pythia_double_token_words(
            model, negative_adjectives
        )
        neutral_adjectives = remove_pythia_double_token_words(
            model, neutral_core_adj
        )

    pos_prompts, neg_prompts, neutral_prompts = get_prompts(
        prompt_type, positive_adjectives, negative_adjectives, neutral_adjectives
    )
    prompts_dict = {
        "positive": model.to_tokens(pos_prompts, prepend_bos=True),
        "negative": model.to_tokens(neg_prompts, prepend_bos=True),
        "neutral": model.to_tokens(neutral_prompts, prepend_bos=True)
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

    answers_dict = {
        'positive': pos_tokens, 
        'negative': neg_tokens, 
        'neutral': neutral_tokens,
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
