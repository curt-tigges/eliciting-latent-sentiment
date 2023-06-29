from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from jaxtyping import Float
from typing import Tuple
import einops

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
    negative_adjectives: list = neg_adj,
    neutral_adjectives: list = neutral_adj,
) -> Tuple[list, list]:
    assert prompt_type in ["simple", "completion", "completion_2", "completion_3", "classification", "classification_2", "classification_3", "classification_4"]
    if prompt_type == "simple":
        pos_prompts = [
            f"I thought this movie was{positive_adjectives[i]}, I loved it. \nConclusion: This movie is" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{negative_adjectives[i]}, I hated it. \nConclusion: This movie is" for i in range(len(negative_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{neutral_adjectives[i]}, I watched it. \nConclusion: This movie is" for i in range(len(neutral_adjectives))
        ]

    elif prompt_type == "completion":
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very" for i in range(len(negative_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very" for i in range(len(neutral_adjectives))
        ]
    elif prompt_type == "completion_2":
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall it was just very good. I felt it was" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall it was just very bad. I felt it was" for i in range(len(positive_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall it was just very average. I felt it was" for i in range(len(positive_adjectives))
        ]
    elif prompt_type == "completion_3":
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall it was just very good. This film was really" for i in range(len(positive_adjectives))
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall it was just very bad. This film was really" for i in range(len(positive_adjectives))
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall it was just very average. This film was really" for i in range(len(positive_adjectives))
        ]
    elif prompt_type == "classification":
        pos_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: 'I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average.' \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == "classification_2":
        pos_prompts = [
            f"Review Text: I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average. \nReview Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == "classification_3":
        pos_prompts = [
            f"Review Text: I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"Review Text: I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"Review Text: I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    elif prompt_type == "classification_4":
        pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall the movie was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall the movie was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        neutral_prompts = [
            f"I thought this movie was{get_adjective(neutral_adjectives, i)}, I watched it. The acting was{get_adjective(neutral_adjectives, i+1)}, the plot was{get_adjective(neutral_adjectives, i+2)}, and overall the movie was just very average. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    return pos_prompts, neg_prompts, neutral_prompts


def get_dataset(
    model: HookedTransformer, 
    device: torch.device,
    n_pairs: int = 1,
    prompt_type: str = "simple",
    pos_answers: list = pos_tokens,
    neg_answers: list = neg_tokens,
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
    assert n_pairs <= len(pos_tokens)
    
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


def get_contrastive_prompts(
    prompt_type: str = "contrastive_tasks", 
    positive_adjectives: list = pos_adj, 
    negative_adjectives: list = neg_adj,
    neutral_adjectives: list = neutral_adj,
) -> Tuple[list, list, list, list]:
    assert prompt_type in ["contrastive_tasks"]
    if prompt_type == "contrastive_tasks":
        task_1_pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall it was just very good. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        task_1_neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall it was just very bad. Review Sentiment:" for i in range(len(positive_adjectives)-1)
        ]
        task_2_pos_prompts = [
            f"I thought this movie was{get_adjective(positive_adjectives, i)}, I loved it. The acting was{get_adjective(positive_adjectives, i+1)}, the plot was{get_adjective(positive_adjectives, i+2)}, and overall it was just very good. This film was really" for i in range(len(positive_adjectives))
        ]
        task_2_neg_prompts = [
            f"I thought this movie was{get_adjective(negative_adjectives, i)}, I hated it. The acting was{get_adjective(negative_adjectives, i+1)}, the plot was{get_adjective(negative_adjectives, i+2)}, and overall it was just very bad. This film was really" for i in range(len(positive_adjectives))
        ]
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    return (
        task_1_pos_prompts, task_1_neg_prompts, task_2_pos_prompts, task_2_neg_prompts
    )


def get_task_contrast_dataset(
    model: HookedTransformer, 
    device: torch.device,
    n_pairs: int = 1,
    prompt_type: str = "contrastive_tasks",
    answers_dict: dict = {},
) -> Tuple[
    list,
    list,
    Float[Tensor, "batch n_pairs 2"],
    Float[Tensor, "batch pos"],
    Float[Tensor, "batch pos"],
]:
    assert set(answers_dict.keys()) == set(["task_1_positive", "task_1_negative", "task_2_positive", "task_2_negative"])
    
    if "pythia" in model.cfg.model_name:
        positive_adjectives = remove_pythia_double_token_words(model, pos_adj)
        negative_adjectives = remove_pythia_double_token_words(model, neg_adj)
        neutral_adjectives = remove_pythia_double_token_words(
            model, neutral_adj
        )
        answers_dict["task_1_positive"] = remove_pythia_double_token_words(model, answers_dict["task_1_positive"])
        answers_dict["task_1_negative"] = remove_pythia_double_token_words(model, answers_dict["task_1_negative"])
        answers_dict["task_2_positive"] = remove_pythia_double_token_words(model, answers_dict["task_2_positive"])
        answers_dict["task_2_negative"] = remove_pythia_double_token_words(model, answers_dict["task_2_negative"])
    else:
        positive_adjectives = pos_adj
        negative_adjectives = neg_adj
        neutral_adjectives = neutral_adj

    task_1_pos_prompts, task_1_neg_prompts, task_2_pos_prompts, task_2_neg_prompts = get_contrastive_prompts(
        prompt_type, positive_adjectives, negative_adjectives, neutral_adjectives
    )
    
    n_prompts = min(
        len(task_1_pos_prompts), 
        len(task_1_neg_prompts),
        len(task_2_pos_prompts),
        len(task_2_neg_prompts), 
    )
    batch_size = n_prompts * 4
    #task_1_prompts = []
    #task_2_prompts = []
    all_prompts = []
    answer_tokens = torch.empty(
        (batch_size, n_pairs, 2), 
        device=device, 
        dtype=torch.long
    )
    
    for i in range(n_prompts):

        # task_1_prompts.append(task_1_pos_prompts[i])
        # task_1_prompts.append(task_1_neg_prompts[i])
        # task_2_prompts.append(task_2_pos_prompts[i])
        # task_2_prompts.append(task_2_neg_prompts[i])
        
        #task1neg
        all_prompts.append(task_1_neg_prompts[i]) 
        #task2pos
        all_prompts.append(task_2_pos_prompts[i])
        #task2neg
        all_prompts.append(task_2_neg_prompts[i])
        #task1pos
        all_prompts.append(task_1_pos_prompts[i])

        
        for pair_idx in range(n_pairs):
            task_1_pos_token = model.to_single_token(answers_dict["task_1_positive"][pair_idx])
            task_1_neg_token = model.to_single_token(answers_dict["task_1_negative"][pair_idx])
            task_2_pos_token = model.to_single_token(answers_dict["task_2_positive"][pair_idx])
            task_2_neg_token = model.to_single_token(answers_dict["task_2_negative"][pair_idx])
     
            # replace this with code that assigns the correct tokens to the answer_tokens tensor
            # answer_tokens[i * 2, pair_idx, 0] = tokens_dict[comparison[0]]
            # answer_tokens[i * 2, pair_idx, 1] = tokens_dict[comparison[1]]
            # answer_tokens[i * 2 + 1, pair_idx, 0] = tokens_dict[comparison[1]]
            # answer_tokens[i * 2 + 1, pair_idx, 1] = tokens_dict[comparison[0]]
            answer_tokens[i * 4, pair_idx, 0] = task_1_neg_token
            answer_tokens[i * 4, pair_idx, 1] = task_2_neg_token
            answer_tokens[i * 4 + 1, pair_idx, 0] = task_2_pos_token
            answer_tokens[i * 4 + 1, pair_idx, 1] = task_1_pos_token
            answer_tokens[i * 4 + 2, pair_idx, 0] = task_2_neg_token
            answer_tokens[i * 4 + 2, pair_idx, 1] = task_1_neg_token
            answer_tokens[i * 4 + 3, pair_idx, 0] = task_1_pos_token
            answer_tokens[i * 4 + 3, pair_idx, 1] = task_2_pos_token
    
    clean_tokens = model.to_tokens(
        all_prompts, prepend_bos=True
    ).to(device)

    flipped_prompts = all_prompts[-2:] + all_prompts[:-2]
    corrupted_tokens = model.to_tokens(
        flipped_prompts, prepend_bos=True
    ).to(device)

    #all_prompts = task_1_prompts+task_2_prompts
    
    return (
        all_prompts, flipped_prompts, answer_tokens, clean_tokens, corrupted_tokens
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
