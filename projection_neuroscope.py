#%%
from functools import partial
import itertools
import gc
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
import einops
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.evals import make_owt_data_loader
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from circuitsvis.activations import text_neuron_activations
from tqdm.notebook import tqdm
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text, to_csv, get_csv
from utils.neuroscope import (
    plot_neuroscope, get_dataloader, get_projections_for_text, plot_top_p, plot_topk, 
    harry_potter_start, harry_potter_fr_start, get_batch_pos_mask, extract_text_window
)
#%%
pd.set_option('display.max_colwidth', 200)
torch.set_grad_enabled(False)
#%%
device = "cuda"
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,
)
model.name = MODEL_NAME
#%%
sentiment_dir = load_array("kmeans_simple_train_ADJ_layer1", model)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(device=device, dtype=torch.float32)
sentiment_dir /= sentiment_dir.norm()
#%%
text = """
Amidst a serene landscape, a group of volunteers worked together to rescue injured animals. Their dedication showcased the true value of compassion and empathy. However, as they tirelessly carried out their mission, they faced an pressing issue: the rapid decline of the local wildlife's natural habitat. Determined to subdue this crisis, they organized campaigns to raise awareness and funds. Their efforts reminded us that even in the face of adversity, unity and determination can make a significant impact.
"""
plot_neuroscope(text, model, centred=True, verbose=False, special_dir=sentiment_dir)
#%%
# ============================================================================ #
# Harry Potter example

#%%
# harry_potter_neuroscope = plot_neuroscope(harry_potter_start, model, centred=True, verbose=False, special_dir=sentiment_dir)
# save_html(harry_potter_neuroscope, "harry_potter_neuroscope", model)
# harry_potter_neuroscope
#%%
# ============================================================================ #

# harry_potter_fr_neuroscope = plot_neuroscope(harry_potter_fr_start, model, centred=True, verbose=False, special_dir=sentiment_dir)
# save_html(harry_potter_fr_neuroscope, "harry_potter_fr_neuroscope", model)
# harry_potter_fr_neuroscope
#%%
# Mandarin example
# mandarin_text = """
# 這是可能發生的最糟糕的事情。 我討厭你這麼說。 你所做的事情太可怕了。

# 然而，你的兄弟卻做了一些了不起的事情。 他非常好，非常令人欽佩，非常善良。 我很愛他。
# """
# plot_neuroscope(mandarin_text, model, centred=True, verbose=False, special_dir=sentiment_dir)
#%%
# ============================================================================ #
# Steering and generating
#%%
def steering_hook(
    input: Float[Tensor, "batch pos d_model"], hook: HookPoint, coef: float, direction: Float[Tensor, "d_model"]
):
    assert 'resid_post' in hook.name
    input += coef * direction
    return input
#%%
def steer_and_test_prompt(
    coef: float,
    direction: Float[Tensor, "d_model"],
    prompt: str,
    answer: str,
    model: HookedTransformer,
    prepend_space_to_answer: bool = True,
):
    model.reset_hooks()
    hook = partial(steering_hook, coef=coef, direction=direction)
    model.add_hook(
        get_act_name("resid_post", 0),
        hook,
        dir="fwd",
    )
    test_prompt(prompt, answer, model, prepend_space_to_answer=prepend_space_to_answer)
    model.reset_hooks()
#%%
def steer_and_generate(
    coef: float,
    direction: Float[Tensor, "d_model"],
    prompt: str,
    model: HookedTransformer,
    **kwargs,
) -> str:
    model.reset_hooks()
    hook = partial(steering_hook, coef=coef, direction=direction)
    model.add_hook(
        get_act_name("resid_post", 0),
        hook,
        dir="fwd",
    )
    input = model.to_tokens(prompt)
    output = model.generate(input, **kwargs)
    model.reset_hooks()
    return model.to_string(output)[0]
#%%
def run_steering_search(
    coefs: Iterable[int], samples: int, sentiment_dir: Float[Tensor, "d_model"], model: HookedTransformer, 
    top_k: int = 10, temperature: float = 1.0, max_new_tokens: int = 20, do_sample: bool = True,
    seed: int = 0,
    prompt: str = "I really enjoyed the movie, in fact I loved it. I thought the movie was just very",
):
    torch.manual_seed(seed)
    out = ""
    for coef, sample in tqdm(itertools.product(coefs, range(samples)), total=len(coefs) * samples):
        if sample == 0:
            out += f"Coef: {coef}\n"
        gen = steer_and_generate(
            coef,
            sentiment_dir,
            prompt,
            model,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        out += gen.replace(prompt, "") + "\n"
    return out.replace("<|endoftext|>", "")
#%%
# steering_text = run_steering_search(
#     coefs=[-20, -10,  0],
#     samples=10,
#     sentiment_dir=sentiment_dir,
#     model=model,
#     top_k=10,
#     temperature=1.0,
#     max_new_tokens=30,
#     do_sample=True,
#     seed=0,
#     prompt="I really enjoyed the movie, in fact I loved it. I thought the movie was just very",
# )
# #%%
# plot_neuroscope(steering_text, centred=True)
#%%
# ============================================================================ #
# Prefixes
#%%
def test_prefixes(fragment: str, prefixes: List[str], model: HookedTransformer):
    single_tokens = []
    for word in prefixes:
        if model.to_str_tokens(word, prepend_bos=False)[0] == fragment:
            single_tokens.append(word)
    single_tokens = list(set(single_tokens))
    text = "\n".join(single_tokens)
    return plot_neuroscope(text, centred=True)
#%%
# test_prefixes(
#     " cr",
#     [' crony', ' crump', ' crinkle', ' craggy', ' cramp', ' crumb', ' crayon', ' cringing', ' cramping'],
#     model
# )
# #%%
# test_prefixes(
#     " clo",
#     [' clopped', ' cloze', ' cloistered', ' clopping', ' cloacal', ' cloister', ' cloaca',],
#     model
# )
#%%
# ============================================================================ #
# Negations
#%%
# negating_positive_text = "Here are my honest thoughts. You're not a good person. I don't like you. I hope that you don't succeed."
# plot_neuroscope(negating_positive_text, centred=True, verbose=False)
#%%
# negating_negative_text = "Here are my honest thoughts. You never fail. You're not bad at all. "
# plot_neuroscope(negating_negative_text, centred=True, verbose=False)
# #%%
# plot_neuroscope(
#     "Here are my honest thoughts. You never fail. You're not bad at all.", 
#     centred=True, 
#     verbose=False,
# )
# #%%
# plot_neuroscope(
#     "Here are my honest thoughts. Don't doubt yourself. You need not fear. You are not wrong. You are very much", 
#     centred=True, 
#     verbose=False,
# )
# #%%
# plot_neuroscope(
#     "Don't be sad. You should not feel ashamed. You are a truly", 
#     centred=True, 
#     verbose=False,
# )
#%%
# test_prompt(
#     "Here are my honest thoughts. You never fail. You're not bad at all. You will always", 
#     "", 
#     model
# )
# #%%
# test_prompt(
#     "Don't be sad. You have nothing to be ashamed of. You are a truly", 
#     "", 
#     model,
#     top_k=20,
# )
# #%%
# test_prompt(
#     "Here are my honest thoughts. You are not a good person. Your behaviour is not okay. You are very", 
#     "", 
#     model,
#     top_k=20
# )
#%%
# negating_weird_text = "Here are my honest thoughts. You are disgustingly beautiful. I hate how much I love you. Stop being so good at everything."
# plot_neuroscope(negating_weird_text, centred=True, verbose=False)
#%%
multi_token_negative_text = """
Alas, it is with a regretful sigh that I endeavor to convey my cogitations regarding the cinematic offering that is "Oppenheimer," a motion picture that sought to render an illuminating portrayal of the eponymous historical figure, yet found itself ensnared within a quagmire of ponderous pacing, desultory character delineations, and an ostentatious predilection for pretentious verbosity, thereby culminating in an egregious amalgamation of celluloid that fails egregiously to coalesce into a coherent and engaging opus.

From its inception, one is greeted with a superfluous indulgence in visual rhapsodies, replete with panoramic vistas and artistic tableaux that appear, ostensibly, to strive for profundity but instead devolve into a grandiloquent spectacle that serves naught but to obfuscate the underlying narrative. The esoteric nature of the cinematographic composition, while intended to convey a sense of erudition, inadvertently estranges the audience, stifling any vestige of emotional resonance that might have been evoked by the thematic elements.

Regrettably, the characters, ostensibly intended to be the vessels through which the audience navigates the tumultuous currents of historical transformation, emerge as little more than hollow archetypes, devoid of psychological nuance or relatable verisimilitude. Their interactions, laden with stilted dialogues and ponderous monologues, meander aimlessly in the midst of a ponderous expanse, rendering their ostensibly profound endeavors an exercise in vapid verbosity rather than poignant engagement.

The directorial predilection for intellectual acrobatics is manifest in the labyrinthine structure of the narrative, wherein chronology becomes a malleable construct, flitting whimsically between past and present without discernible rhyme or reason. While this narrative elasticity might have been wielded as a potent tool of thematic resonance, it instead metastasizes into an obfuscating force that imparts a sense of disjointed incoherence upon the cinematic proceedings, leaving the viewer to grapple with a puzzling tapestry of events that resist cohesive assimilation.

Moreover, the fervent desire to imbue the proceedings with a veneer of intellectual profundity is acutely palpable within the film's verbiage-laden script. Dialogue, often comprising polysyllabic words of labyrinthine complexity, becomes an exercise in linguistic gymnastics that strays perilously close to the precipice of unintentional self-parody. This quixotic dalliance with ostentatious vocabulary serves only to erect an insurmountable barrier between the audience and the narrative, relegating the viewer to a state of befuddled detachment.

In summation, "Oppenheimer," for all its aspirations to ascend the cinematic pantheon as an erudite exploration of historical gravitas, falters egregiously beneath the weight of its own ponderous ambitions. With an overarching penchant for verbal ostentation over emotional resonance, a narrative structure that veers perilously into the realm of disjointed incoherence, and characters bereft of authentic vitality, this cinematic endeavor sadly emerges as an exercise in cinematic misdirection that regrettably fails to ignite the intellectual or emotional faculties of its audience.
"""
# plot_neuroscope(multi_token_negative_text, centred=True, verbose=False, model=model, special_dir=sentiment_dir)
#%%
# ============================================================================ #
# Openwebtext-10k
#%%
dataloader = get_dataloader(model, "stas/openwebtext-10k", batch_size=8)
#%%
def get_activations_from_dataloader(
    data: torch.utils.data.dataloader.DataLoader,
    max_batches: int = None,
) -> Float[Tensor, "row pos"]:
    all_acts = []
    for batch_idx, batch_value in tqdm(enumerate(data), total=len(data)):
        batch_tokens = batch_value['tokens'].to(device)
        batch_acts: Float[Tensor, "batch pos layer"] = get_projections_for_text(batch_tokens, sentiment_dir, model)
        all_acts.append(batch_acts)
        if max_batches is not None and batch_idx >= max_batches:
            break
    # Concatenate the activations into a single tensor
    all_acts: Float[Tensor, "row pos layer"] = torch.cat(all_acts, dim=0)
    return all_acts
#%%
class ClearCache:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        model.cuda()

    def __exit__(self, exc_type, exc_val, exc_tb):
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
#%%
if is_file("sentiment_activations.npy", model):
    sentiment_activations = load_array("sentiment_activations", model)
    sentiment_activations: Float[Tensor, "row pos layer"]  = torch.tensor(
        sentiment_activations, device=device, dtype=torch.float32
    )
else:
    with ClearCache():
        sentiment_activations: Float[Tensor, "row pos layer"]  = get_activations_from_dataloader(dataloader)
    save_array(sentiment_activations, "sentiment_activations", model)
sentiment_activations.shape, sentiment_activations.device
#%%
# ============================================================================ #
# Anthropic Graph 1

#%%
def sample_by_bin(
    data: Float[Tensor, "batch pos"],
    bins: int = 20,
    samples_per_bin: int = 20,
    seed: int = 0,
    window_size: int = 10,
    verbose: bool = False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    flat = data.flatten()
    hist, bin_edges = np.histogram(flat.cpu().numpy(), bins=bins)
    bin_indices: Int[np.ndarray, "batch pos"] = np.digitize(data.cpu().numpy(), bin_edges)
    if verbose:
        print(bin_edges)
    indices = []
    for bin_idx in range(1, bins + 1):
        lb = bin_edges[bin_idx - 1]
        ub = bin_edges[bin_idx]
        bin_batches, bin_positions = np.where(bin_indices == bin_idx)
        bin_samples = np.random.randint(0, len(bin_batches), samples_per_bin)
        indices += [
            (bin_idx, lb, ub, bin_batches[bin_sample], bin_positions[bin_sample])
            for bin_sample in bin_samples
        ]
    df =  pd.DataFrame(indices, columns=["bin", "lb", "ub", "batch", "position"])
    tokens = []
    texts = []
    for _, row in df.iterrows():
        text = extract_text_window(
            int(row.batch), int(row.position), dataloader, model, window_size=window_size
        )
        tokens.append(text[window_size])
        texts.append("".join(text))
    df.reset_index(drop=True, inplace=True)
    df['token'] = tokens
    df['text'] = texts
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
#%%
bin_samples = sample_by_bin(
    sentiment_activations[:, :, 1], verbose=False
)
to_csv(bin_samples, "bin_samples", model)
bin_samples
#%%
labelled_bin_samples = get_csv(
    "labelled_bin_samples", model
)
labelled_bin_samples.sentiment = labelled_bin_samples.sentiment.str.replace('negative', 'Negative').str.replace('positive', 'Positive')
assert labelled_bin_samples.sentiment.isin(['Positive', 'Negative', 'Neutral', 'Somewhat Positive', 'Somewhat Negative']).all()
labelled_bin_samples
#%%
sampled_activations = []
for idx, row in labelled_bin_samples.iterrows():
    sampled_activations.append(sentiment_activations[row.batch, row.position, 1].detach().cpu().numpy())
labelled_bin_samples['activation'] = sampled_activations
labelled_bin_samples
#%%
fig = px.histogram(
    labelled_bin_samples,
    x="activation",
    color="sentiment",
    nbins=200,
    title="Histogram of sentiment activations by label",
    barmode="overlay",
    marginal="rug",
    histnorm="probability density",
    hover_data=["token", "text"]
)
fig.update_layout(
    title_x=0.5,
    showlegend=True,
)
fig.show()
#%%
def cumulative_proportion_plot(
    df: pd.DataFrame,
):
    sentiments = df['sentiment'].unique()
    df = df.sort_values(by='activation').reset_index(drop=True)

    fig = go.Figure()
    df['global_cum_count'] = df.index + 1
    for sentiment in sentiments:
        sentiment_clean = sentiment.lower().replace(' ', '_')
        df[f'{sentiment_clean}_cum_count'] = (df.sentiment == sentiment).cumsum()
        cum_prop = df[f'{sentiment_clean}_cum_count'] / df.global_cum_count
        df[f'{sentiment_clean}_cum_prop'] = cum_prop
        hovertext = [f"Sentiment: {sentiment}<br>Activation: {act:.3f}<br>Cumulative Proportion: {prop:.3f}" 
                     for act, prop in zip(df['activation'], cum_prop)]
        fig.add_trace(go.Scatter(
            x=df['activation'], y=cum_prop, fill='tonexty', name=sentiment, 
            hoverinfo="text", hovertext=hovertext,
            stackgroup='one', hoveron = 'points',
        ))

    fig.update_layout(title="Anthropic Graph 1: Cumulative Proportion of Sentiment by Activation",
                    xaxis_title="Activation",
                    yaxis_title="Cumulative Proportion",
                    title_x=0.5,
                    showlegend=True)

    return fig
#%%
def plot_stacked_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df, x="activation", color="sentiment", barmode="stack",
        hover_data=["token", "text"],
        nbins=100,
        title="Anthropic Graph 2: Stacked Histogram of Activation by Sentiment",
        labels={'activation': 'Activation Value'},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        marginal="rug"
    ) 
    fig.update_layout(
        title_x=0.5,
        showlegend=True,   
    )
    return fig
#%%
cumulative_proportion_plot(labelled_bin_samples)
#%%
# plot_stacked_histogram(labelled_bin_samples)
#%%
# ============================================================================ #
# Anthropic Graph 2
#%%
def get_normal_sample(
    data: Float[Tensor, "batch pos"],
    n_samples: int = 800,
    window_size: int = 10,
    layer: int = 1,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size, seq_len = data.shape
    data_size = batch_size * seq_len
    index: Int[Tensor, "batch_size seq_len"] = einops.rearrange(
        torch.arange(data_size),
        "(batch_size seq_len) -> batch_size seq_len",
        batch_size=batch_size,
        seq_len=seq_len,
    )
    indices: Int[Tensor, "sample"] = torch.randperm(data_size)[:n_samples]
    batches, positions = torch.where(torch.isin(index, indices))
    df = pd.DataFrame({"batch": batches, "position": positions})
    tokens = []
    texts = []
    activations = []
    for _, row in df.iterrows():
        batch = int(row.batch)
        position = int(row.position)
        text = extract_text_window(
            batch, position, dataloader, model, window_size=window_size
        )
        activation = sentiment_activations[batch, position, layer].detach().cpu().numpy()
        tokens.append(model.to_string(dataloader.dataset[batch]['tokens'][position]))
        texts.append("".join(text))
        activations.append(activation)
    df.reset_index(drop=True, inplace=True)
    df['token'] = tokens
    df['text'] = texts
    df['activation'] = activations
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
#%%
normal_samples = get_normal_sample(sentiment_activations[:, :, 1])
to_csv(normal_samples, "normal_samples", model)
normal_samples
    
#%%
labelled_normal_samples = get_csv(
    "labelled_normal_samples", model
)
labelled_normal_samples.sentiment = labelled_normal_samples.sentiment.str.replace('negative', 'Negative').str.replace('positive', 'Positive')
assert labelled_normal_samples.sentiment.isin(['Positive', 'Negative', 'Neutral', 'Somewhat Positive', 'Somewhat Negative']).all()
labelled_normal_samples
#%%
plot_stacked_histogram(labelled_normal_samples)
#%%
# ============================================================================ #
# Anthropic Graph 3
#%%
def plot_ev_histogram(df: pd.DataFrame, nbins: int = 100):
    sentiments = df['sentiment'].unique()
    df = df.sort_values(by='activation').reset_index(drop=True)
    df['activation_cut'] = pd.cut(df.activation, bins=nbins)
    df.activation_cut = df.activation_cut.apply(lambda x: 0.5 * (x.left + x.right))
    fig = go.Figure()
    data = []
    for x, bin_df in df.groupby('activation_cut'):
        prob_x = len(bin_df) / len(df)
        ev = x * prob_x
        label_props = bin_df.value_counts('sentiment', normalize=True, sort=False)
        data.append([ev * label_props.get(sentiment, 0) for sentiment in sentiments])
    data = pd.DataFrame(data, columns=sentiments)
    # Adding bar traces for each sentiment
    x_values = df['activation_cut'].unique()
    for sentiment in sentiments:
        fig.add_trace(go.Bar(
            x=x_values, y=data[sentiment], name=sentiment,
            hovertemplate='<br>'.join([
                'Sentiment: ' + sentiment,
                'Activation: %{x}',
                'EV Contribution: %{y:.4f}',
            ]),
            xaxis='x1', yaxis='y1',
        ))

    fig.update_layout(
        barmode="stack", title="Anthropic Graph 3: Stacked Histogram of EV contribution",
        title_x=0.5,
        showlegend=True,
        xaxis_title="Activation",
        yaxis_title="EV Contribution",
    )

    return fig
#%%
plot_ev_histogram(labelled_normal_samples)
#%%
# ============================================================================ #
# Top k max activating examples
# %%
# plot_topk(sentiment_activations, dataloader, model, k=50, layer=6, window_size=20, centred=True)
# # %%
# plot_topk(sentiment_activations, k=50, layer=12, window_size=20, centred=True)
# %%
# ============================================================================ #
# Top p sampling
#%%
# plot_top_p(sentiment_activations, k=50, layer=1, p=0.01)
#%%
# ============================================================================ #
# Exclusions
#%%
def expand_exclusions(exclusions: Iterable[str]):
    expanded_exclusions = []
    for exclusion in exclusions:
        exclusion = exclusion.strip().lower()
        expanded_exclusions.append(exclusion)
        expanded_exclusions.append(exclusion + " ")
        expanded_exclusions.append(" " + exclusion)
        expanded_exclusions.append(" " + exclusion + " ")
        expanded_exclusions.append(exclusion.capitalize())
        expanded_exclusions.append(" " + exclusion.capitalize())
        expanded_exclusions.append(exclusion.capitalize() + " ")
        expanded_exclusions.append(exclusion.upper())
        expanded_exclusions.append(" " + exclusion.upper())
        expanded_exclusions.append(exclusion.upper() + " ")
    return list(set(expanded_exclusions))
#%%
exclusions = [
    # proper nouns
    # 'Flint', 'Fukushima', 'Obama', 'Assad', 'Gaza',
    # 'CIA', 'BP', 'istan', 'VICE', 'TSA', 'Mitt', 'Romney', 'Afghanistan', 'Kurd', 'Molly',
    # 'DoS', 'Medicaid', 'Kissinger',
    'ISIS', 'GOP',
    # the rest
    'adequate', 'truly', 'mis', 'dys', 'provides', 'offers', 'fully',  'migraine',  
    'really', 'considerable', 'reasonably', 'substantial', 'additional', 'STD', 
    'Narcolepsy', 'Tooth', 'RUDE', 'Diagnostic',  '!', 
    'agoraphobia', 'greenhouse', 
    'stars', 'star',
    ' perfect', ' fantastic',' marvelous',' good',' remarkable',' wonderful',
    ' fabulous',' outstanding',' awesome',' exceptional',' incredible',' extraordinary',
    ' amazing',' lovely',' brilliant',' terrific',' superb',' spectacular',' great',
    ' beautiful'
    ' dreadful',' bad',' miserable',' horrific',' terrible',
    ' disgusting',' disastrous',' horrendous',' offensive',' wretched',
    ' awful',' unpleasant',' horrible',' mediocre',' disappointing',
    'excellent', 'opportunity', 'success', 'generous', 'harmful', 'plaguing', 'derailed', 'unwanted',
    'stigma', 'burdened', 'stereotypes', 'hurts', 'burdens', 'harming', 'winning', 'smooth', 
    'shameful', 'hurting', 'nightmare', 'inflicted', 'disadvantaging', 'stigmatized', 'stigmatizing',
    'stereotyped', 'forced', 'confidence', 'senseless', 'wrong', 'hurt', 'stereotype', 'sexist',
    'unnecesarily', 'horribly',  'impressive', 'fraught', 'brute', 'blight',
    'unnecessary', 'unnecessarily', 'fraught','deleterious', 'scrapped', 'intrusive',
    'unhealthy', 'plague', 'hated', 'burden', 'vilified', 'afflicted', 'polio', 'inaction',
    'condemned', 'crippled', 'unrestrained', 'derail', 'cliché', 
    'toxicity', 'bastard', 'clich', 'politicized', 'overedit', 'curse', 'choked', 'politicize',
    'frowned', 'sorry', 'slurs', 'taboo', 'bullshit', 'painfully', 'premature', 'worsened',
    'pathogens', 'Domestic', 'Violence', 'painful',
    'splendid', 'magnificent', 'beautifully', 'gorgeous', 'nice', 'phenomenal',
    'finest', 'splendid', 'wonderfully',
    'ugly', 'dehuman', 'negatively', 'degrading', 'rotten', 'traumatic', 'crying',
    'criticized', 'dire', 'best', 'exceptionally', 'negative', 'dirty',
    'rotting','enjoy', 'amazingly', 'brilliantly', 'traumatic', 'hinder', 'hindered',
    'depressing', 'diseased', 'depressing', 'bleary', 'carcinogenic', 'demoralizing',
    'traumatizing', 'injustice', 'blemish', 'nausea', 'peeing', 'abhorred', 
    'appreciate', 'perfectly', 'elegant', 'supreme', 'excellence', 'sufficient', 'toxic', 'hazardous',
    'muddy', 'hinder', 'derelict', 'disparaged', 'sour', 'disgraced', 'degenerate',
    'disapproved', 'annoy', 'nicely', 'stellar', 'charming', 'cool', 'handsome', 'exquisite',
    'sufficient', 'cool', 'brilliance', 'flawless', 'delightful', 'impeccable', 'fascinating',
    'decent', 'genius', 'appreciated', 'remarkably', 'greatest', 'humiliating', 
    'embarassing', 'saddening', 'injustice', 'hinders', 'annihilate', 'waste', 'unliked',
    'stunning', 'glorious', 'deft', 'enjoyed', 'ideal', 'stylish', 'sublime', 'admirable',
    'embarass', 'injustices', 'disapproval', 'misery', 'sore', 'prejudice', 'disgrace',
    'messed', 'capable', 'breathtaking', 'suffered', 'poisoned', 'ill', 'unsafe', 
    'morbid', 'irritated', 'irritable', 'contaiminate', 'derogatory',
    'prejudging', 'inconvenienced', 'embarrassingly', 'embarrass', 'embarassed', 'embarrassment',
    'fine', 'better', 'unparalleled', 'astonishing', 'neat', 'embarrassing', 'doom',
    'inconvenient', 'boring', 'conatiminate', 'contaminated', 'contaminating', 'contaminates',
    'penalty', 'tarnish', 'disenfranchised', 'disenfranchising', 'disenfranchisement',
    'super', 'marvel', 'enjoys', 'talented', 'clever', 'enhanced', 'ample',
    'love', 'expert', 'gifted', 'loved', 'enjoying', 'enjoyable', 'enjoyed', 'enjoyable',
    'tremendous', 'confident', 'confidently', 'love', 'harms', 'jeapordize', 'jeapordized',
    'depress', 'penalize', 'penalized', 'penalizes', 'penalizing', 'penalty', 'penalties',
    'tarred', 'nauseating', 'harms', 'lethality', 'loves', 'unique', 'appreciated', 'appreciates',
    'appreciating', 'appreciation', 'appreciative', 'appreciates', 'appreciated', 'appreciating',
    'favorite', 'greatness', 'goodness', 'suitable', 'prowess', 'masterpiece', 'ingenious', 'strong',
    'versatile', 'well', 'effective', 'scare', 'shaming', 'worse', 'bleak', 'hate', 'tainted',
    'destructive', 'doomed', 'celebrated', 'gracious', 'worthy', 'interesting', 'coolest', 
    'intriguing', 'enhance', 'enhances', 'celebrated', 'genuine', 'smoothly', 'greater', 'astounding',
    'classic', 'successful', 'innovative', 'plenty', 'competent', 'noteworthy', 'treasures',
    'adore', 'adores', 'adored', 'adoring', 'adorable', 'adoration', 'adore', 'grim',
    'displeased', 'mismanagement', 'jeopardizes', 'garbage', 'mangle', 'stale',
    'excel', 'wonders', 'faithful', 'extraordinarily', 'inspired', 'vibrant', 'faithful', 'compelling',
    'standout', 'exemplary', 'vibrant', 'toxic', 'contaminate', 'antagonistic', 'terminate',
    'detrimental', 'unpopular', 'fear', 'outdated', 'adept', 'charisma', 'popular', 'popularly',
    'humiliation', 'sick', 'nasty', 'fatal', 'distress', 'unfavorable', 'foul', 
    'bureaucratic', 'dying', 'nasty', 'worst', 'destabilising', 'unforgiving', 'vandalized',
    'polluted', 'poisonous', 'dirt', 'original', 'incredibly', 'invaluable', 'acclaimed',
    'successfully', 'able', 'reliable', 'loving', 'beauty', 'famous', 'solid', 'rich',
    'famous', 'thoughtful', 'enhancement', 'sufficiently', 'robust', 'bestselling', 'renowned',
    'impressed', 'elegence', 'thrilled', 'hostile', 'scar', 'piss', 'danger', 'inflammatory',
    'diseases', 'disillusion', 'depressive', 'bum', 'disgust', 'aggravates', 'pissy',
    'dangerous', 'urinary', 'pissing', 'nihilism', 'nihilistic', 'disillusioned', 'depressive', 
    'dismal', 'trustworthy', 'unjust', 'enthusiastic', 'seamlesslly', 'seamless', 'liked',
    'enthusiasm', 'superior', 'useful', 'master', 'heavenly', 'enthusiastic', 'effortlessly',
    'adequately', 'powerful', 'seamlessly', 'dumb', 'dishonors', 'traitor',
    'bleed', 'invalid', 'horror', 'reprehensible', 'die', 'petty', 'lame', 'fouling', 'foul',
    'racist', 'elegance', 'top', 'waste', 'wasteful', 'wasted', 'wasting', 'wastes', 'wastefulness',
    'trample', 'trampled', 'vexing', 'vitriol', 'stangate', 'stagnant', 'stagnate', 'stagnated',
    'crisis', 'vex', 'corroded', 'sad', 'bitter', 'insults', 'impres', 'cringe', 'humilate', 'humiliates',
    'humiliated', 'humiliating', 'humiliation', 'humiliations', 'humiliates', 'humiliatingly',
    'corrosive', 'corrosion', 'corroded', 'corrodes', 'corroding', 'corrosive', 'corrosively',
    'inhospitable', 'waste', 'wastes', 'wastefulness', 'wasteful', 'wasted', 'wasting',
    'unintended', 'stressful', 'trash', 'unhappy', 'unhappily', 'unhappiness', 'unhappier',
    'unholy', 'peril', 'perilous', 'perils', 'perilously', 'perilousness', 'perilousnesses',
    'faulty', 'damaging', 'damages', 'damaged', 'damagingly', 'damages', 'damaging', 'damaged',
    'trashy', 'punitive', 'punish', 'punished', 'punishes', 'punishing', 'punishment', 'punishments',
    'pessimistic', 'pessimism', 'inspiring', 'impress', 'coward', 'tired', 'empty',
    'trauma', 'torn', 'unease', 'gloomy', 'gloom', 'gloomily', 'gloominess', 'gloomier',
    'hideous', 'embarrassed', 'wastes', 'wasteful', 'misdemeanour', 'nuisance',
    'dilemma',' dilemmas', 'sewage', 'bogie', 'postponed', 'backward', 'paralyze',
    'very', 'special', 'important', 'more', 'nervous', 'awkward', 'problem', 'pain', 'loss',
    'melancholy', 'dismissing', 'complain', 'stomp', 'terrorist', 'racism', 'criminal',
    'colder', 'nuclear', 'divided', 'death', 'chlorine', 'illegal', 'risks',
    'prisons', 'villain', 'incinerate', 'dead', 'lonely', 'mistakes', 'biased', 'illicit',
    'defeat', 'lose', 'unbearable', 'presure', 'desperation', 
    'osteoarthritis', 'Medicating', 'Medications', 'Medication', 'depressed', 'crimes',
    'suck', 'hemorrhage', 'crap', 'dull', 'headaches', 'turbulent', 'intolerant',
    'vulnerable', 'insignificant', 'insignificance', 'blame', 'Lie', 'jail', 'abuse',
    'reputable', 'slave', 'harm', 'died', 'viruses', 'homeless', 'blind', 'mistake',
    'war', 'accident', 'incidents', 'radiation', 'cursed', 'scorn', 'deaths', 'slow',
    'crashing', 'warning', 'hypocritical', 'hypocrisy', 'problems', 'disappointment',
    'blood', 'slut', 'skewer', 'vaguely', 'riots', 'unclear', 'charm', 'disease', 'creepy',
    'burning', 'lack', 'guilty', 'glaring', 'failed', 'indoctrination', 'incoherent',
    'hospital', 'syphilis', 'guilty', 'infection', 'faux', 'burning', 'creepy',
    'disease', 'welts', 'trojans', 'trojan', 'makeshift', 'cant', 'tragic', 'stupid',
    'vulgar', 'horrors', 'ugliness', 'miseries', 'loathing', 'hatred', 'dread', 'brutal',
    'satisfactory', 'okay', 'ok', 'satisfying', 'filthy', 'crash', 'cynical', 'mourning',
    'messy', 'tragedies', 'satisfied', 'cruelty', 'sadness', 'brutality', 'worsening',
    'suicidal', 'despair', 'neatly', 'appropriately', 'handy', 'significant',
    "'kay", 'aogny', 'sadly', 'hates', 'disaster', 'atrocities', 'effectively',
    'worth', 'capability', 'ability', 'optimum', 'agony', 'tragedy', 'desperate',
    'satisfy', 'optimal', 'helpful', 'definitely', 'cruel', 'crashed', 'ignorant',
    'wrongful', 'imprisonment', 'cheap', 'severe', 'contamination', 'worried',
    'anxiety', 'complaining', 'Hurricane', 'threat',

]
exclusions = expand_exclusions(exclusions)
#%%
# plot_topk(
#     sentiment_activations, dataloader, model,
#     k=20, layer=1, window_size=20, centred=True,
#     exclusions=exclusions,
# )
# #%%
# plot_top_p(
#     sentiment_activations, dataloader, model,
#     p=.02,
#     k=20, layer=1, window_size=20, centred=True,
#     exclusions=exclusions,
# )
# %%
save_text('\n'.join(exclusions), 'sentiment_exclusions', model)
#%%
# ============================================================================ #
# Histograms

#%%
def plot_histogram(
    tokens: Union[str, List[str]], 
    all_activations: Float[Tensor, "row pos layer"], 
    name: str = None,
    layer: int = 0, 
    nbins: int = 100,
):
    if name is None:
        assert isinstance(tokens, str)
        name = tokens
    assert isinstance(name, str)
    activations: Float[Tensor, "row pos"] = all_activations[:, :, layer]
    mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(tokens, dataloader, model, all_activations)
    assert mask.shape == activations.shape
    activations_to_plot = activations[mask].flatten()
    fig = go.Histogram(x=activations_to_plot.cpu().numpy(), nbinsx=nbins, name=name)
    return fig
#%%
def plot_histograms(
    tokens: Dict[str, List[str]], all_activations: Float[Tensor, "row pos layer"], layer: int = 0,
    nbins: int = 100,
):
    fig = make_subplots(rows=len(tokens), cols=1, shared_xaxes=True, shared_yaxes=True)
    for idx, (name, token) in enumerate(tokens.items()):
        hist = plot_histogram(token, all_activations, name, layer, nbins)
        fig.add_trace(hist, row=idx+1, col=1)
    fig.update_layout(
        title_text=f"Layer {layer} resid_pre sentiment cosine sims",
        height=200 * (idx + 1)
    )
    return fig
# %%
pos_list = [
    " amazing", " great", " excellent", " good", " wonderful", " fantastic", " awesome", 
    " nice", " superb", " perfect", " incredible", " beautiful"
]
neg_list = [
    " terrible", " bad", " awful", " horrible", " disgusting", " awful", 
    " evil", " scary",
]
neutral_list = [
    " okay", " alright", " decent", " acceptable", " satisfactory",
]
pos_neg_dict = {
    "positive": pos_list,
    "negative": neg_list,
    "neutral": neutral_list,
    # "surprising_proper_nouns": [" Trek", " Yorkshire", " Linux", " Reuters"],
    # "trek": [" Trek"],
    # "yorkshire": [" Yorkshire"],
    # "linux": [" Linux"],
    # "reuters": [" Reuters"],
    # "first_names": [" John", " Mary", " Bob", " Alice"],
    # "places": [" London", " Paris", " Tokyo"],
    # "exclamation_mark": ["!"],
    # "other_punctuation": [".", ",", "?", ":", ";"],
}
# plot_histograms(
#     pos_neg_dict, 
#     all_activations=sentiment_activations, 
#     layer=1, 
#     nbins=100,
# )
# %%
# plot_topk(sentiment_activations, dataloader, model, k=20, layer=1, inclusions=pos_list)
# %%
# plot_topk(sentiment_activations, k=50, layer=1, inclusions=[".", ",", "?", ":", ";"])
#%%
# ============================================================================ #
# Means and variances
#%%
def compute_mean_variance(
    all_activations: Float[Tensor, "row pos layer"], layer: int, model: HookedTransformer,
):
    activations: Float[pd.Series, "batch_and_pos"] = pd.Series(all_activations[:, :, layer].flatten().cpu().numpy())
    tokens: Int[pd.DataFrame, "batch_and_pos"] = dataloader.dataset.data.to_pandas().tokens.explode(ignore_index=True)
    counts = tokens.value_counts()
    means = activations.groupby(tokens).mean()
    std_devs = activations.groupby(tokens).std()
    return counts, means, std_devs
#%%
token_counts, token_means, token_std_devs = compute_mean_variance(sentiment_activations, 1, model)
#%%
def plot_top_mean_variance(
    counts: pd.Series, means: pd.Series, std_devs: pd.Series, model: HookedTransformer, k: int = 10, 
    min_count: int = 10,
):
    means = means[counts >= min_count].dropna().sort_values()
    std_devs = std_devs[counts >= min_count].dropna().sort_values()
    means_top_and_bottom = pd.concat([means.head(k), means.tail(k)]).reset_index()
    means_top_and_bottom['valence'] = ["negative"] * k + ["positive"] * k
    means_top_and_bottom.columns = ['token', 'mean', 'valence']
    means_top_and_bottom.token = [
        f"{i}:{tok}" 
        for i, tok in zip(
            means_top_and_bottom.token, 
            model.to_str_tokens(torch.tensor(means_top_and_bottom.token))
        )
    ]
    fig = px.bar(data_frame=means_top_and_bottom, x='token', y='mean', color='valence')
    fig.update_layout(title_text="Most extreme means", title_x=0.5)
    fig.show()
    std_devs_top_and_bottom = pd.concat([std_devs.head(k), std_devs.tail(k)]).reset_index()
    std_devs_top_and_bottom['variation'] = ["consistent"] * k + ["variable"] * k
    std_devs_top_and_bottom.columns = ['token', 'std_dev', 'variation']
    std_devs_top_and_bottom.token = [
        f"{i}:{tok}" 
        for i, tok in zip(
            std_devs_top_and_bottom.token, 
            model.to_str_tokens(torch.tensor(std_devs_top_and_bottom.token))
        )
    ]
    fig = px.bar(data_frame=std_devs_top_and_bottom, x='token', y='std_dev', color='variation')
    fig.update_layout(title_text="Most extreme standard deviations", title_x=0.5)
    fig.show()

# %%
# plot_top_mean_variance(token_counts, token_means, token_std_devs, model=model, k=10)
# %%
# plot_topk(sentiment_activations, k=10, layer=1, inclusions=[" Yorkshire"], window_size=20)
#%%
def resample_hook(
    input: Float[Tensor, "batch pos d_model"], 
    hook: HookPoint, 
    direction: Float[Tensor, "d_model"],
):
    assert 'resid' in hook.name
    assert direction.shape == (model.cfg.d_model,)
    assert direction.norm().item() == 1.0
    # shuffle input tensor along the batch dimension
    shuffled = input[torch.randperm(input.shape[0])]
    orig_proj: Float[Tensor, "batch pos"] = einops.einsum(
        input, direction, 'batch pos d_model, d_model -> batch pos'
    )
    new_proj: Float[Tensor, "batch pos"] = einops.einsum(
        shuffled, direction, 'batch pos d_model, d_model -> batch pos'
    )
    return (
        input + (new_proj - orig_proj).unsqueeze(-1) * direction
    )
#%%
def get_resample_ablated_loss_diffs(
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
    dataloader: DataLoader,
    k: int = 10,
    window_size: int = 10,
    layer: int = 0,
    seed: int = 0,
    max_batch: int = None
):
    torch.manual_seed(seed)
    model.reset_hooks()
    hook = partial(resample_hook, direction=direction)
    loss_diffs = []

    bar = tqdm(dataloader, total=len(dataloader))
    for batch_idx, batch_value in bar:
        bar.set_description(f"Batch {batch_idx}")
        batch_tokens = batch_value['tokens'].to(device)
        model.reset_hooks()
        orig_loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True)
        model.add_hook(
            get_act_name('resid_post', layer),
            hook,
            dir="fwd",
        )
        new_loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True)
        loss_diff: Float[Tensor, "mb pos"] = new_loss - orig_loss
        loss_diffs.append(loss_diff)
        model.reset_hooks()
        if max_batch is not None and batch_idx + 1 >= max_batch:
            break
    loss_diffs = torch.cat(loss_diffs, dim=0)

    return plot_topk(
        loss_diffs, dataloader, model, k=k, layer=layer, window_size=window_size, centred=True,
    )
# %%
loss_diff_text = get_resample_ablated_loss_diffs(sentiment_dir, model, dataloader, k=50, window_size=10)
plot_neuroscope(''.join(loss_diff_text), centred=True)
#%%
