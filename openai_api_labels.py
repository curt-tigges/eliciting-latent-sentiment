#%%
import pandas as pd
import openai
from utils.store import get_csv, to_csv
#%%
pd.set_option('display.max_colwidth', 200)
MODEL = 'gpt2-small'
FILE_NAME = 'normal_samples'
#%%
with open("api_key.txt", "r") as f:
    openai.api_key = f.read()
# %%
csv_df = get_csv(FILE_NAME, MODEL)
csv_df.head()
# %%
prefix = "Your job is to classify the sentiment of a given token (i.e. word or word fragment) into Positive/Neutral/Negative."
sentiment_data = []
assert len(csv_df) < 1_000
for idx, row in csv_df.iterrows():
    token = row['token']
    context = row['text']
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"{prefix} Token: '{token}'. Context: '{context}'. Sentiment: "
        }]
    )
    sentiment_data.append(chat_completion.choices[0].message.content)
    if idx > 1_000:
        break
# %%
out_df = csv_df.iloc[:len(sentiment_data)].copy()
out_df['sentiment'] = sentiment_data
to_csv(out_df, f'labelled_{FILE_NAME}', MODEL)
out_df
# %%
