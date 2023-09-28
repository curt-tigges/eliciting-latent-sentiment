import numpy as np
import pandas as pd
import os
from enum import Enum
from datasets import Dataset, DatasetDict
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Literal, Tuple
from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
import einops
from utils.circuit_analysis import get_logit_diff
from utils.prompts import CleanCorruptedDataset, ReviewScaffold
from utils.store import save_pickle, to_csv


def get_merged_dataframe(root: str = "stanfordSentimentTreebank"):
    phrase_ids = pd.read_csv(os.path.join(root, "dictionary_fixed.txt"), sep="|", header=None, names=["phrase", "phrase_id"])
    phrase_ids.head()
    
    phrase_labels = pd.read_csv(os.path.join(root, "sentiment_labels.txt"), sep="|").rename(
        columns={"phrase ids": "phrase_id", "sentiment values": "sentiment_value"}
    )
    # round sentiment value to integer
    phrase_labels.sentiment_value = phrase_labels.sentiment_value.apply(lambda x: int(round(x)))
    
    phrases_df = pd.merge(
        phrase_ids, phrase_labels, on="phrase_id", how="inner", validate="one_to_one"
    ).drop_duplicates('phrase', keep='first')
    
    sentence_ids = pd.read_csv(os.path.join(root, "datasetSentences_fixed.txt"), sep="\t")
    
    sentence_splits = pd.read_csv(os.path.join(root, "datasetSplit.txt"), sep=",")
    
    sentences_df = pd.merge(
        sentence_ids, sentence_splits, on="sentence_index", how="inner", validate="one_to_one"
    ).drop_duplicates('sentence', keep='first')
    
    # eliminate duplicates in phrase column in phrases_df
    phrases_df = phrases_df.drop_duplicates('phrase', keep='first')
    
    sentence_phrase_df = pd.merge(
        sentences_df, phrases_df, 
        left_on="sentence", right_on="phrase", 
        how="inner", validate="one_to_one"
    )
    sentence_phrase_df['split'] = sentence_phrase_df['splitset_label'].map({1: 'train', 2: 'test', 3: 'dev'})
    return sentence_phrase_df


def pair_by_num_tokens_and_split(df: pd.DataFrame):
    # Sort and group by num_tokens and split
    grouped = df.groupby(['num_tokens', 'split'])

    # Create new dataframe to store paired rows
    column_names = ['num_tokens', 'split'] + [
        f'{col}_{pos_neg}' for pos_neg in ('pos', 'neg') 
        for col in df.columns if col not in ('num_tokens', 'split', 'sentiment_value')
    ]
    
    paired_data = []
    
    # For each group, pair the rows
    for (_, split_value), group in grouped:
        pos_rows = group[group['sentiment_value'] == 1].to_dict(orient='records')
        neg_rows = group[group['sentiment_value'] == 0].to_dict(orient='records')

        # Pair up rows
        for pos, neg in zip(pos_rows, neg_rows):
            new_data = {'num_tokens': pos['num_tokens'], 'split': split_value}
            for col in df.columns:
                if col not in ['num_tokens', 'split', 'sentiment_value']:
                    new_data[f'{col}_pos'] = pos[col]
                    new_data[f'{col}_neg'] = neg[col]
            paired_data.append(new_data)

    paired_df = pd.DataFrame(paired_data, columns=column_names)
    return paired_df


def convert_to_dataset_dict(df: pd.DataFrame) -> DatasetDict:
    # Filter DataFrame based on 'split' column
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    dev_df = df[df['split'] == 'dev']

    # Convert filtered DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df.drop(columns=['split']))
    test_dataset = Dataset.from_pandas(test_df.drop(columns=['split']))
    dev_dataset = Dataset.from_pandas(dev_df.drop(columns=['split']))

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'dev': dev_dataset
    })
    # Display the DatasetDict
    print(dataset_dict)
    # remove dataset columns
    dataset_dict = dataset_dict.remove_columns(['sentence_index', 'splitset_label', 'phrase', 'phrase_id', '__index_level_0__'])
    dataset_dict = dataset_dict.rename_column('sentiment_value', 'label')
    dataset_dict = dataset_dict.rename_column('sentence', 'text')
    dataset_dict.save_to_disk('data/sst2')
    return dataset_dict


def apply_scaffold_to_prompts(prompts: List[str], scaffold: str):
    if scaffold == ReviewScaffold.PLAIN:
        return prompts
    elif scaffold == ReviewScaffold.CLASSIFICATION:
        return [
            f"Review Text: {prompt} Review Sentiment:"
            for prompt in prompts
        ]
    elif scaffold == ReviewScaffold.CONTINUATION:
        return [
            f"{prompt} Overall the movie was just very"
            for prompt in prompts
        ]
    else:
        raise ValueError(f"Invalid scaffold: {scaffold}")
    

def construct_answer_tokens(
    scaffold: str, 
    sentiment: Int[np.ndarray, "batch"],
    model: HookedTransformer
) -> Float[Tensor, "half_length 1 2"]:
    assert np.isin(sentiment, [0, 1]).all()
    if scaffold == ReviewScaffold.PLAIN:
        positive = torch.tensor([0, 1])
        negative = torch.tensor([1, 0])
    elif scaffold == ReviewScaffold.CLASSIFICATION:
        positive = torch.tensor([
            model.to_single_token(' Positive'), model.to_single_token(' Negative'),
        ], device="cpu")
        negative = torch.tensor([
            model.to_single_token(' Negative'), model.to_single_token(' Positive'),
        ], device="cpu")
    elif scaffold == ReviewScaffold.CONTINUATION:
        positive = torch.tensor([
            model.to_single_token(' good'), model.to_single_token(' bad'),
        ], device="cpu")
        negative = torch.tensor([
            model.to_single_token(' bad'), model.to_single_token(' good'),
        ], device="cpu")
    else:
        raise ValueError(f"Invalid scaffold: {scaffold}")
    positive = einops.repeat(
        positive,
        "correct -> batch correct",
        batch=len(sentiment),
    )
    negative = einops.repeat(
        negative,
        "correct -> batch correct",
        batch=len(sentiment),
    )
    sentiment = einops.repeat(
        torch.tensor(sentiment, dtype=torch.long, device="cpu"),
        "batch -> batch 2",
    )
    out = torch.where(
        sentiment == 1,
        positive,
        negative,
    )
    assert out.shape == (len(sentiment), 2)
    out = out.unsqueeze(1)
    return out


def filter_by_logit_diff(
    model: HookedTransformer,
    input_tokens: Int[Tensor, "batch pos"],
    answer_tokens: Int[Tensor, "batch pair correct"],
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    min_logit_diff: float = 0,
) -> Bool[Tensor, "batch"]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.ones(input_tokens.shape[0], dtype=torch.bool)
    bar = tqdm(range(0, len(input_tokens), batch_size))
    bar.set_description(
        "Running forward pass to filter by logit diff on "
        f"{input_tokens.shape} data"
    )
    mean_logit_diff = 0
    for start_idx in bar:
        end_idx = start_idx + batch_size
        batch_tokens = input_tokens[start_idx:end_idx].to(device)
        batch_answers = answer_tokens[start_idx:end_idx].to(device)
        logits = model(
            batch_tokens, 
            return_type="logits"
        )
        logit_diff = get_logit_diff(
            logits,
            batch_answers,
            per_prompt=True,
            tokens=batch_tokens,
            tokenizer=model.tokenizer,
        )
        mean_logit_diff += logit_diff.sum().item()
        mask[start_idx:end_idx] = logit_diff > min_logit_diff
    mean_logit_diff /= len(input_tokens)
    print(
        f"Average logit diff prior: {mean_logit_diff:.3f}\n"
        f"Filtering from {len(input_tokens)} to {mask.sum()} "
    )
    return mask


def apply_scaffold_to_data_frame(
    df: pd.DataFrame, 
    scaffold: ReviewScaffold, 
    model: HookedTransformer,
    padding_side: Optional[Literal['left', 'right']] = None,
):
    
    prompts = apply_scaffold_to_prompts(df.phrase.tolist(), scaffold)
    if padding_side is not None:
        model.tokenizer.padding_side = padding_side
    input_tokens = model.to_tokens(prompts)
    answer_tokens = construct_answer_tokens(
        scaffold, df.sentiment_value.values, model
    )
    return input_tokens, answer_tokens


def alternating_array(n):
    # Create an array of length n with all 1s
    ones_array = np.ones(n, dtype=int)
    
    # Create an array of length n with all 0s
    zeros_array = np.zeros(n, dtype=int)
    
    # Initialize the final alternating array
    alternating = np.empty(2 * n, dtype=int)
    
    # Fill the alternating array with 1s and 0s
    alternating[::2] = ones_array
    alternating[1::2] = zeros_array
    
    return alternating
    

def create_dataset_for_split(
    full_df: pd.DataFrame,
    split: str,
    model: HookedTransformer,
    scaffold: ReviewScaffold = ReviewScaffold.PLAIN,
    padding_side: Optional[Literal['left', 'right']] = None,
    min_logit_diff: float = 0,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
):
    split_df = full_df.loc[full_df.split == split]
    input_raw, answers_raw = apply_scaffold_to_data_frame(
        split_df, scaffold, model, padding_side=padding_side
    )
    mask = filter_by_logit_diff(
        model, input_raw, answers_raw, 
        min_logit_diff=min_logit_diff,
        device=device,
        batch_size=batch_size,
    )
    mask_df = split_df[mask.numpy()]
    paired_df = pair_by_num_tokens_and_split(mask_df)
    to_csv(paired_df, f'treebank_{split}_{scaffold.value}', model)
    clean_prompts = [
        item 
        for pair in zip(
            paired_df.phrase_pos.tolist(), 
            paired_df.phrase_neg.tolist()
        ) 
        for item in pair
    ] # pos neg alternating
    corrupt_prompts = clean_prompts[1:] + [clean_prompts[0]]
    clean_prompts = apply_scaffold_to_prompts(clean_prompts, scaffold)
    corrupt_prompts = apply_scaffold_to_prompts(corrupt_prompts, scaffold)
    if padding_side is not None:
        model.tokenizer.padding_side = padding_side
    clean_tokens = model.to_tokens(clean_prompts)
    corrupted_tokens = model.to_tokens(corrupt_prompts)
    answer_tokens = construct_answer_tokens(
        scaffold, 
        alternating_array(len(paired_df)),
        model
    )
    dataset = CleanCorruptedDataset(
        clean_tokens=clean_tokens.cpu(),
        corrupted_tokens=corrupted_tokens.cpu(),
        answer_tokens=answer_tokens.cpu(),
        all_prompts=clean_prompts,
        tokenizer=model.tokenizer,
    )
    dataset.shuffle()
    save_pickle(dataset, f'treebank_{split}_{scaffold.value}', model)
    print(
        f"split: {split}\n"
        f"scaffold: {scaffold.value}\n"
        f"split size: {len(split_df)}\n"
        f"filtered split size: {len(mask_df)}\n"
        f"paired size: {len(paired_df)}\n"
        f"dataset shape: {clean_tokens.shape}\n"
        f"min logit diff: {min_logit_diff}\n"
    )


def create_datasets_for_model(
    model: HookedTransformer,
    sentence_phrase_df: pd.DataFrame,
    padding_side: Optional[Literal['left', 'right']] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
):
    sentence_phrase_df['str_tokens'] = ''
    for idx, row in sentence_phrase_df.iterrows():
        str_tokens = model.to_str_tokens(row['sentence'])
        sentence_phrase_df.loc[idx, 'str_tokens'] = '|'.join(str_tokens)
        sentence_phrase_df.loc[idx, 'num_tokens'] = len(str_tokens)
    sentence_phrase_df.num_tokens = sentence_phrase_df.num_tokens.astype(int)
    for split in ('train', 'dev', 'test'):
        for scaffold in ReviewScaffold:
            if scaffold != ReviewScaffold.CLASSIFICATION or split != 'test':
                continue # FIXME: remove this line
            create_dataset_for_split(
                sentence_phrase_df, split, model, scaffold, 
                padding_side=padding_side,
                device=device,
                batch_size=batch_size,
                min_logit_diff=0,
            )