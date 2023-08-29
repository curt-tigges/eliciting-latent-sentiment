#%%
import pandas as pd
import openai
from utils.store import get_csv, to_csv
#%%
pd.set_option('display.max_colwidth', 200)
MODEL = 'gpt2-small'
#%%
with open("api_key.txt", "r") as f:
    openai.api_key = f.read()
# %%
bin_df = get_csv('bin_samples', MODEL)
bin_df.head()
# %%
prefix = "Your job is to classify the sentiment of a given token (i.e. word or word fragment) into Positive/Somewhat positive/Neutral/Somewhat negative/Negative."
sentiment_data = []
for idx, row in bin_df.iterrows():
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
sentiment_data
# %%
out_df = bin_df.iloc[:len(sentiment_data)].copy()
out_df['sentiment'] = sentiment_data
to_csv(out_df, 'labelled_bin_samples', MODEL)
out_df
# %%
