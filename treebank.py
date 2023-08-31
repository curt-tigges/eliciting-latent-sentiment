# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# !ls

# %%
# %cd eliciting-latent-sentiment

# %%
# !pip install circuitsvis

# %%
import pandas as pd
from datasets import Dataset, DatasetDict
import os
from transformer_lens import HookedTransformer
import torch
from utils.prompts import CleanCorruptedDataset
from utils.store import save_pickle, load_pickle, save_dataset_dict
# %%
ROOT = "stanfordSentimentTreebank"
phrase_ids = pd.read_csv(os.path.join(ROOT, "dictionary_fixed.txt"), sep="|", header=None, names=["phrase", "phrase_id"])
phrase_ids.head()
# %%
phrase_labels = pd.read_csv(os.path.join(ROOT, "sentiment_labels.txt"), sep="|").rename(
    columns={"phrase ids": "phrase_id", "sentiment values": "sentiment_value"}
)
# round sentiment value to integer
phrase_labels.sentiment_value = phrase_labels.sentiment_value.apply(lambda x: int(round(x)))
print(phrase_labels.sentiment_value.value_counts())
phrase_labels.head()
# %%
phrases_df = pd.merge(phrase_ids, phrase_labels, on="phrase_id", how="inner", validate="one_to_one")
phrases_df.head()
# %%
sentence_ids = pd.read_csv(os.path.join(ROOT, "datasetSentences_fixed.txt"), sep="\t")
sentence_ids.head()

# %%
sentence_splits = pd.read_csv(os.path.join(ROOT, "datasetSplit.txt"), sep=",")
sentence_splits.head()
# %%
sentences_df = pd.merge(
    sentence_ids, sentence_splits, on="sentence_index", how="inner", validate="one_to_one"
).drop_duplicates('sentence', keep='first')
sentences_df.head()
# %%
# eliminate duplicates in phrase column in phrases_df
phrases_df = phrases_df.drop_duplicates('phrase', keep='first')

# %%
sentence_phrase_df = pd.merge(
    sentences_df, phrases_df, 
    left_on="sentence", right_on="phrase", 
    how="inner", validate="one_to_one"
)
sentence_phrase_df['split'] = sentence_phrase_df['splitset_label'].map({1: 'train', 2: 'test', 3: 'dev'})
sentence_phrase_df.head()
# %%
len(sentence_ids), len(sentences_df), len(sentence_phrase_df)
# %%
sentence_phrase_df.split.value_counts()
# %%
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

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
def create_dataset_for_split(
    full_df: pd.DataFrame,
    split: str,
    model: HookedTransformer,
):
    df = full_df.loc[full_df.split == split]
    clean_prompts = df.phrase_pos.tolist() + df.phrase_neg.tolist()
    corrupt_prompts = df.phrase_neg.tolist() + df.phrase_pos.tolist()
    clean_tokens = model.to_tokens(clean_prompts)
    corrupted_tokens = model.to_tokens(corrupt_prompts)
    answer_tokens = torch.tensor([(1, 0)] * len(df) + [(0, 1)] * len(df)).unsqueeze(1)
    dataset = CleanCorruptedDataset(
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        answer_tokens=answer_tokens,
        all_prompts=clean_prompts
    )
    save_pickle(dataset, f'treebank-{split}', model)
    print(split, clean_tokens.shape, corrupted_tokens.shape, answer_tokens.shape, len(clean_prompts))
# %%
def create_dataset_for_model(
    model: HookedTransformer,
    sentence_phrase_df: pd.DataFrame = sentence_phrase_df,
):
    for idx, row in sentence_phrase_df.iterrows():
        str_tokens = model.to_str_tokens(row['sentence'])
        sentence_phrase_df.loc[idx, 'str_tokens'] = '|'.join(str_tokens)
        sentence_phrase_df.loc[idx, 'num_tokens'] = len(str_tokens)
    sentence_phrase_df.num_tokens = sentence_phrase_df.num_tokens.astype(int)
    paired_df = pair_by_num_tokens_and_split(sentence_phrase_df)
    create_dataset_for_split(paired_df, 'train', model)
    create_dataset_for_split(paired_df, 'dev', model)
    create_dataset_for_split(paired_df, 'test', model)
# %%
MODELS = [
    'gpt2-small',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
]
for model in MODELS:
    model = HookedTransformer.from_pretrained(model, device=device)
    create_dataset_for_model(model)

# %%
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
    save_dataset_dict(dataset_dict, 'sst2', model)
    dataset_dict.save_to_disk('sst2')
    return dataset_dict

# %%
convert_to_dataset_dict(sentence_phrase_df)
#%%