import pandas as pd
import os
from enum import Enum
from datasets import Dataset, DatasetDict
from typing import List
from transformer_lens import HookedTransformer
import torch
from utils.prompts import CleanCorruptedDataset
from utils.store import save_pickle


class ReviewScaffold(Enum):
    PLAIN = 'plain'
    CLASSIFICATION = 'classification'
    CONTINUATION = 'continuation'


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
    

def construct_answer_tokens(scaffold: str, length: int, model: HookedTransformer):
    if scaffold == ReviewScaffold.PLAIN:
        return torch.tensor([(1, 0)] * length + [(0, 1)] * length).unsqueeze(1)
    elif scaffold == ReviewScaffold.CLASSIFICATION:
        return torch.tensor(
            [(model.to_single_token(' Positive'), model.to_single_token(' Negative'))] * length +
            [(model.to_single_token(' Negative'), model.to_single_token(' Positive'))] * length
        )
    elif scaffold == ReviewScaffold.CONTINUATION:
        return torch.tensor(
            [(model.to_single_token(' good'), model.to_single_token(' bad'))] * length +
            [(model.to_single_token(' bad'), model.to_single_token(' good'))] * length
        )
    else:
        raise ValueError(f"Invalid scaffold: {scaffold}")
    

def create_dataset_for_split(
    full_df: pd.DataFrame,
    split: str,
    model: HookedTransformer,
    scaffold: str = ReviewScaffold.PLAIN,
):
    df = full_df.loc[full_df.split == split]
    clean_prompts = df.phrase_pos.tolist() + df.phrase_neg.tolist()
    corrupt_prompts = df.phrase_neg.tolist() + df.phrase_pos.tolist()
    clean_prompts = apply_scaffold_to_prompts(clean_prompts, scaffold)
    corrupt_prompts = apply_scaffold_to_prompts(corrupt_prompts, scaffold)
    clean_tokens = model.to_tokens(clean_prompts)
    corrupted_tokens = model.to_tokens(corrupt_prompts)
    answer_tokens = construct_answer_tokens(scaffold, len(clean_prompts), model)
    dataset = CleanCorruptedDataset(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        answer_tokens=answer_tokens,
        all_prompts=clean_prompts
    )
    save_pickle(dataset, f'treebank_{split}_{scaffold.value}', model)
    print(split, scaffold.value, clean_tokens.shape, corrupted_tokens.shape, answer_tokens.shape, len(clean_prompts))


def create_dataset_for_model(
    model: HookedTransformer,
    sentence_phrase_df: pd.DataFrame,
):
    for idx, row in sentence_phrase_df.iterrows():
        str_tokens = model.to_str_tokens(row['sentence'])
        sentence_phrase_df.loc[idx, 'str_tokens'] = '|'.join(str_tokens)
        sentence_phrase_df.loc[idx, 'num_tokens'] = len(str_tokens)
    sentence_phrase_df.num_tokens = sentence_phrase_df.num_tokens.astype(int)
    paired_df = pair_by_num_tokens_and_split(sentence_phrase_df)
    for split in ('train', 'dev', 'test'):
        for scaffold in ReviewScaffold:
            create_dataset_for_split(paired_df, split, model, scaffold)