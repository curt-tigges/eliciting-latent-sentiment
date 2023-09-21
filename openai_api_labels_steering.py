#%%
import pandas as pd
import openai
from utils.store import load_pickle, to_csv, save_html, save_pdf, get_csv, is_file
import plotly.graph_objects as go
import plotly.express as px
#%%
pd.set_option('display.max_colwidth', 200)
MODEL = 'gpt2-small'
#%%
with open("api_key.txt", "r") as f:
    openai.api_key = f.read()
#%%
steering_dict = load_pickle('steering_dict', MODEL)
# %%
def generate():
    prefix = "Your job is to classify the sentiment of a snippet of a movie review into Positive/Somewhat positive/Neutral/Somewhat negative/Negative."
    sentiment_data = []
    assert len(steering_dict) < 1_000
    idx = 0
    for coef, outputs in steering_dict.items():
        for sample, review in enumerate(outputs):
            chat_completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"{prefix} Review: '{review}'. Sentiment: "
                }]
            )
            sentiment_data.append((
                coef,
                sample,
                review,
                chat_completion.choices[0].message.content
            ))
            idx += 1
        if idx > 1_000:
            break
    out_df = pd.DataFrame(
        sentiment_data, columns=['coef', 'sample', 'review', 'sentiment']
    )
    to_csv(out_df, 'labelled_steering', MODEL)
    return out_df
#%%
if not is_file('labelled_steering.csv', MODEL):
    gen_df = generate()
    assert is_file('labelled_steering.csv', MODEL)
#%%
out_df = get_csv('labelled_steering', MODEL)
out_df
# %%
def plot_bin_proportions(df: pd.DataFrame, nbins=50):
    df.sentiment = df.sentiment.str.replace('negative', 'Negative').str.replace('positive', 'Positive')
    assert df.sentiment.isin(['Positive', 'Negative', 'Neutral', 'Somewhat Positive', 'Somewhat Negative']).all()

    sentiments = sorted(df['sentiment'].unique())
    df = df.sort_values(by='coef').reset_index(drop=True)
    df['coef_cut'] = pd.cut(df.coef, bins=nbins)
    df.coef_cut = df.coef_cut.apply(lambda x: 0.5 * (x.left + x.right))
    
    fig = go.Figure()
    data = []
    
    for x, bin_df in df.groupby('coef_cut'):
        if bin_df.empty:
            continue
        label_props = bin_df.value_counts('sentiment', normalize=True, sort=False)
        data.append([label_props.get(sentiment, 0) for sentiment in sentiments])
    
    data = pd.DataFrame(data, columns=sentiments)
    cumulative_data = data.cumsum(axis=1)  # Cumulative sum along columns
    
    x_values = df['coef'].unique()
    
    # Adding traces for the rest of the sentiments
    for idx, sentiment in enumerate(sentiments):
        fig.add_trace(go.Scatter(
            x=x_values, y=cumulative_data[sentiment], name=sentiment,
            hovertemplate='<br>'.join([
                'Sentiment: ' + sentiment,
                'coef: %{x}',
                'Cum. Label proportion: %{y:.4f}',
            ]),
            fill='tonexty',
            mode='lines',
        ))
    
    fig.update_layout(
        title="Proportion of Sentiment by Steering Coefficient", # Anthropic Graph 1
        title_x=0.5,
        showlegend=True,
        xaxis_title="coef",
        yaxis_title="Cum. Label proportion",
    )

    return fig
# %%
fig = plot_bin_proportions(out_df)
save_pdf(fig, 'steering_bin_proportions', MODEL)
save_html(fig, 'steering_bin_proportions', MODEL)
save_pdf(fig, 'steering_bin_proportions', MODEL)
fig.show()
# %%
