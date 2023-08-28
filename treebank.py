#%%
import pandas as pd
import os
from transformer_lens import HookedTransformer
import torch
from utils.prompts import CleanCorruptedDataset
from utils.store import save_pickle, load_pickle
#%%
ROOT = "stanfordSentimentTreebank"
phrase_ids = pd.read_csv(os.path.join(ROOT, "dictionary.txt"), sep="|", header=None, names=["phrase", "phrase_id"])
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
sentence_ids = pd.read_csv(os.path.join(ROOT, "datasetSentences.txt"), sep="\t")
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HookedTransformer.from_pretrained('gpt2-small', device=device)
# %%
for idx, row in sentence_phrase_df.iterrows():
    str_tokens = model.to_str_tokens(row['sentence'])
    sentence_phrase_df.loc[idx, 'str_tokens'] = '|'.join(str_tokens)
    sentence_phrase_df.loc[idx, 'num_tokens'] = len(str_tokens)
# %%
sentence_phrase_df.num_tokens = sentence_phrase_df.num_tokens.astype(int)
sentence_phrase_df.num_tokens.value_counts()
# %%
sentence_phrase_df.head()
# %%
def pair_by_num_tokens(df: pd.DataFrame):
    # Sort and group by num_tokens
    grouped = df.groupby('num_tokens')

    # Create new dataframe to store paired rows
    column_names = ['num_tokens'] + [
        f'{col}_{pos_neg}' for pos_neg in ('pos', 'neg') 
        for col in df.columns if col not in ('num_tokens', 'sentiment_value')
    ]
    paired_data = []
    # For each group, pair the rows
    for _, group in grouped:
        pos_rows = group[group['sentiment_value'] == 1].to_dict(orient='records')
        neg_rows = group[group['sentiment_value'] == 0].to_dict(orient='records')

        # Pair up rows
        for pos, neg in zip(pos_rows, neg_rows):
            new_data = {'num_tokens': pos['num_tokens']}
            for col in df.columns:
                if col not in ['num_tokens', 'sentiment_value']:
                    new_data[f'{col}_pos'] = pos[col]
                    new_data[f'{col}_neg'] = neg[col]
            paired_data.append(new_data)

    paired_df = pd.DataFrame(paired_data, columns=column_names)
    return paired_df
# %%
paired_df = pair_by_num_tokens(sentence_phrase_df)
paired_df
# %%
# first half positive to negative, second half negative to positive
clean_prompts = paired_df.phrase_pos.tolist() + paired_df.phrase_neg.tolist()
corrupt_prompts = paired_df.phrase_neg.tolist() + paired_df.phrase_pos.tolist()
clean_tokens = model.to_tokens(clean_prompts)
corrupted_tokens = model.to_tokens(corrupt_prompts)
answer_tokens = torch.tensor([(1, 0)] * len(paired_df) + [(0, 1)] * len(paired_df)).unsqueeze(1)
dataset = CleanCorruptedDataset(
    clean_tokens=clean_tokens,
    corrupted_tokens=corrupted_tokens,
    answer_tokens=answer_tokens,
    all_prompts=clean_prompts
)
clean_tokens.shape, corrupted_tokens.shape, answer_tokens.shape, len(clean_prompts)
# %%
save_pickle(dataset, 'treebank', model)
# %%
load_pickle('treebank', model).all_prompts == dataset.all_prompts
# %%