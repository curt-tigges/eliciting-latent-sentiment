# %%
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
from utils.store import get_csv, to_csv, is_file

# %%
pd.set_option("display.max_colwidth", 200)
MODEL = "Qwen-1_8B"
OVERWRITE = False
# %%
DIRECTIONS = [
    "kmeans_simple_train_ADJ_layer1",
    "mean_diff_simple_train_ADJ_layer1",
    "logistic_regression_simple_train_ADJ_layer1",
    "das_simple_train_ADJ_layer1",
    # "pca_simple_train_ADJ_layer1",
]
SUFFIX = "_bin_samples.csv"
# %%
with open("api_key.txt", "r") as f:
    client = OpenAI(api_key=f.read())


# %%
def classify_tokens(file_name: str, max_rows: int = 1_000):
    csv_df = get_csv(file_name, MODEL)
    assert len(csv_df) > 0, f"File {file_name} is empty or does not exist."
    csv_df.head()
    prefix = "Your job is to classify the sentiment of a given token (i.e. word or word fragment) into Positive/Neutral/Negative."
    sentiment_data = []
    assert len(csv_df) < max_rows
    for idx, row in tqdm(csv_df.iterrows(), total=len(csv_df)):
        token = row["token"]
        context = row["text"]
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"{prefix} Token: '{token}'. Context: '{context}'. Sentiment: ",
                }
            ],
        )
        sentiment_data.append(chat_completion.choices[0].message.content)
        if idx > max_rows:
            break
    out_df = csv_df.iloc[: len(sentiment_data)].copy()
    out_df["sentiment"] = sentiment_data
    to_csv(out_df, f"labelled_{file_name}", MODEL)
    out_df


# %%
bar = tqdm(DIRECTIONS)
for direction in bar:
    file_name = direction + SUFFIX
    bar.set_description(f"Classifying {file_name}")
    labelled_file = f"labelled_{file_name}"
    if is_file(labelled_file, MODEL) and not OVERWRITE:
        print(f"Skipping {labelled_file}")
        continue
    bar.set_description(f"Classifying {file_name} by calling OpenAI API...")
    classify_tokens(file_name)
# %%
