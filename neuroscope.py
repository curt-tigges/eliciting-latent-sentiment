#%%
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
from circuitsvis.activations import text_neuron_activations
from tqdm.notebook import tqdm
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text
#%%
torch.set_grad_enabled(False)
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
sentiment_dir = load_array("km_2c_line_embed_and_mlp0", model)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(device=device, dtype=torch.float32)
sentiment_dir /= sentiment_dir.norm()
#%%
def names_filter(name: str):
    return name.endswith('resid_post') or name == get_act_name('resid_pre', 0)
#%%
def get_activations_for_input(
    tokens: Int[Tensor, "batch pos"],
) -> Float[Tensor, "batch pos layer"]:
    _, cache = model.run_with_cache(
        tokens, 
        names_filter=names_filter
    )
    acts_by_layer = []
    for layer in range(-1, model.cfg.n_layers):
        if layer >= 0:
            emb: Int[Tensor, "batch pos d_model"] = cache["resid_post", layer]
        else:
            emb: Int[Tensor, "batch pos d_model"] = cache["resid_pre", 0]
        emb /= emb.norm(dim=-1, keepdim=True)
        act: Float[Tensor, "batch pos"] = einops.einsum(
            emb, sentiment_dir,
            "batch pos d_model, d_model -> batch pos"
        ).to(device="cpu")
        acts_by_layer.append(act)
    acts_by_layer: Float[Tensor, "batch pos layer"] = torch.stack(acts_by_layer, dim=2)
    return acts_by_layer
#%%
def plot_neuroscope(
    text: Union[str, List[str]], centred: bool, activations: Float[Tensor, "pos layer 1"] = None,
    verbose=False,
):
    tokens: Int[Tensor, "batch pos"] = model.to_tokens(text)
    if isinstance(text, str):
        str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
    else:
        str_tokens = text
    if verbose:
        print(f"Tokens shape: {tokens.shape}")
    if activations is None:
        if verbose:
            print("Computing activations")
        activations: Float[Tensor, "batch pos layer"] = get_activations_for_input(tokens)
        activations: Float[Tensor, "pos layer 1"] = einops.rearrange(
            activations, "batch pos layer -> pos layer batch"
        )
        if verbose:
            print(f"Activations shape: {activations.shape}")
    if centred:
        if verbose:
            print("Centering activations")
        layer_means = einops.reduce(activations, "pos layer 1 -> 1 layer 1", reduction="mean")
        layer_means = einops.repeat(layer_means, "1 layer 1 -> pos layer 1", pos=activations.shape[0])
        activations -= layer_means
    elif verbose:
        print("Activations already centered")
    assert (
        activations.ndim == 3
    ), f"activations must be of shape [tokens x layers x neurons], found {activations.shape}"
    assert len(str_tokens) == activations.shape[0], (
        f"tokens and activations must have the same length, found tokens={len(str_tokens)} and acts={activations.shape[0]}, "
        f"tokens={str_tokens}, "
        f"activations={activations.shape}"

    )
    return text_neuron_activations(
        tokens=str_tokens, 
        activations=activations,
        first_dimension_name="Layer (resid_pre)",
        second_dimension_name="Model",
        second_dimension_labels=["gpt2-small"],
    )
#%%
# ============================================================================ #
# Harry Potter example

#%%
harry_potter_start = """
    Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.

    Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere.

    The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn’t think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley’s sister, but they hadn’t met for several years; in fact, Mrs. Dursley pretended she didn’t have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn’t want Dudley mixing with a child like that.

    When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.

    None of them noticed a large, tawny owl flutter past the window.

    At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. “Little tyke,” chortled Mr. Dursley as he left the house. He got into his car and backed out of number four’s drive.

    It was on the corner of the street that he noticed the first sign of something peculiar — a cat reading a map. For a second, Mr. Dursley didn’t realize what he had seen — then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn’t a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn’t read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day.

    But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn’t help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr. Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren’t young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt — these people were obviously collecting for something . . . yes, that would be it. The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills.

    Mr. Dursley always sat with his back to the window in his office on the ninth ﬂoor. If he hadn’t, height have found it harder to concentrate on drills that morning. He didn’t see the owls swooping past in broad daylight, though people down in the street did; they pointed and gazed open-mouthed as owl after owl sped overhead. Most of them had never seen an owl even at nighttime. Mr. Dursley, however, had a perfectly normal, owl-free morning. He yelled at ﬁve diﬀerent people. He made several important telephone calls and shouted a bit more. He was in a very good mood until lunchtime, when he thought he’d stretch his legs and walk across the road to buy himself a bun from the bakery.

    He’d forgotten all about the people in cloaks until he passed a group of them next to the baker’s. He eyed them angrily as he passed. He didn’t know why, but they made him uneasy. This bunch were whispering excitedly, too, and he couldn’t see a single collecting tin. It was on his way back past them, clutching a large doughnut in a bag, that he caught a few words of what they were saying.
"""
#%%
harry_potter_neuroscope = plot_neuroscope(harry_potter_start, centred=True, verbose=False)
save_html(harry_potter_neuroscope, "harry_potter_neuroscope", model)
harry_potter_neuroscope
#%%
# ============================================================================ #
# Harry Potter in French
harry_potter_fr_start = """
Mr et Mrs Dursley, qui habitaient au 4, Privet Drive, avaient toujours affirmé avec la plus grande
fierté qu'ils étaient parfaitement normaux, merci pour eux. Jamais quiconque n'aurait imaginé qu'ils
puissent se trouver impliqués dans quoi que ce soit d'étrange ou de mystérieux. Ils n'avaient pas de
temps à perdre avec des sornettes.
Mr Dursley dirigeait la Grunnings, une entreprise qui fabriquait des perceuses. C'était un homme
grand et massif, qui n'avait pratiquement pas de cou, mais possédait en revanche une moustache de
belle taille. Mrs Dursley, quant à elle, était mince et blonde et disposait d'un cou deux fois plus long
que la moyenne, ce qui lui était fort utile pour espionner ses voisins en regardant par-dessus les
clôtures des jardins. Les Dursley avaient un petit garçon prénommé Dudley et c'était à leurs yeux le
plus bel enfant du monde.
Les Dursley avaient tout ce qu'ils voulaient. La seule chose indésirable qu'ils possédaient, c'était un
secret dont ils craignaient plus que tout qu'on le découvre un jour. Si jamais quiconque venait à
entendre parler des Potter, ils étaient convaincus qu'ils ne s'en remettraient pas. Mrs Potter était la
soeur de Mrs Dursley, mais toutes deux ne s'étaient plus revues depuis des années. En fait, Mrs
Dursley faisait comme si elle était fille unique, car sa soeur et son bon à rien de mari étaient aussi
éloignés que possible de tout ce qui faisait un Dursley. Les Dursley tremblaient d'épouvante à la
pensée de ce que diraient les voisins si par malheur les Potter se montraient dans leur rue. Ils savaient
que les Potter, eux aussi, avaient un petit garçon, mais ils ne l'avaient jamais vu. Son existence
constituait une raison supplémentaire de tenir les Potter à distance: il n'était pas question que le petit
Dudley se mette à fréquenter un enfant comme celui-là.
Lorsque Mr et Mrs Dursley s'éveillèrent, au matin du mardi où commence cette histoire, il faisait gris
et triste et rien dans le ciel nuageux ne laissait prévoir que des choses étranges et mystérieuses allaient
bientôt se produire dans tout le pays. Mr Dursley fredonnait un air en nouant sa cravate la plus sinistre
pour aller travailler et Mrs Dursley racontait d'un ton badin les derniers potins du quartier en
s'efforçant d'installer sur sa chaise de bébé le jeune Dudley qui braillait de toute la force de ses
poumons.
"""
harry_potter_fr_neuroscope = plot_neuroscope(harry_potter_fr_start, centred=True, verbose=False)
save_html(harry_potter_fr_neuroscope, "harry_potter_fr_neuroscope", model)
harry_potter_fr_neuroscope
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
test_prefixes(
    " cr",
    [' crony', ' crump', ' crinkle', ' craggy', ' cramp', ' crumb', ' crayon', ' cringing', ' cramping'],
    model
)
#%%
test_prefixes(
    " clo",
    [' clopped', ' cloze', ' cloistered', ' clopping', ' cloacal', ' cloister', ' cloaca',],
    model
)
#%%
# ============================================================================ #
# Negations
#%%
# negating_positive_text = "Here are my honest thoughts. You're not a good person. I don't like you. I hope that you don't succeed."
# plot_neuroscope(negating_positive_text, centred=True, verbose=False)
#%%
negating_negative_text = "Here are my honest thoughts. You never fail. You're not bad at all. "
plot_neuroscope(negating_negative_text, centred=True, verbose=False)
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
plot_neuroscope(multi_token_negative_text, centred=True, verbose=False)
#%%
# ============================================================================ #
# Openwebtext-10k
#%%
def get_dataloader():
    owt_data = load_dataset("stas/openwebtext-10k", split="train")
    dataset = tokenize_and_concatenate(owt_data, model.tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=True
    )
    return data_loader

#%%
dataloader = get_dataloader()
#%%
def get_activations_from_dataloader(
    data: torch.utils.data.dataloader.DataLoader,
    max_batches: int = None,
) -> Float[Tensor, "row pos"]:
    all_acts = []
    for batch_idx, batch_value in tqdm(enumerate(data), total=len(data)):
        batch_tokens = batch_value['tokens'].to(device)
        batch_acts: Float[Tensor, "batch pos layer"] = get_activations_for_input(batch_tokens)
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
    sentiment_activations: Float[Tensor, "row pos layer"]  = torch.tensor(sentiment_activations, device=device, dtype=torch.float32)
else:
    with ClearCache():
        sentiment_activations: Float[Tensor, "row pos layer"]  = get_activations_from_dataloader(dataloader)
    save_array(sentiment_activations, "sentiment_activations", model)
sentiment_activations.shape, sentiment_activations.device
#%%
# ============================================================================ #
# Top k max activating examples

#%%
def get_window(batch: int, pos: int, window_size: int = 10) -> Tuple[int, int]:
    lb = max(0, pos - window_size)
    ub = min(len(dataloader.dataset[batch]['tokens']), pos + window_size)
    return lb, ub
#%%
def extract_text_window(batch: int, pos: int, window_size: int = 10) -> List[str]:
    lb, ub = get_window(batch, pos, window_size)
    tokens = dataloader.dataset[batch]['tokens'][lb:ub]
    str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
    # print(tokens, str_tokens)
    return str_tokens
#%%
def extract_activations_window(
    activations: Float[Tensor, "row pos layer"], 
    batch: int, pos: int, window_size: int = 10,
) -> Float[Tensor, "pos layer"]:
    lb, ub = get_window(batch, pos, window_size)
    return activations[batch, lb:ub, :]
#%%
def get_batch_pos_mask(tokens: Union[str, List[str], Tensor], activations: Float[Tensor, "row pos"] = None):
    mask_values: Int[Tensor, "words"] = torch.unique(model.to_tokens(tokens, prepend_bos=False).flatten())
    masks = []
    for batch_idx, batch_value in enumerate(dataloader):
        batch_tokens: Int[Tensor, "batch_size pos 1"] = batch_value['tokens'].to(device).unsqueeze(-1)
        batch_mask: Bool[Tensor, "batch_size pos"] = torch.any(batch_tokens == mask_values, dim=-1)
        masks.append(batch_mask)
    mask: Bool[Tensor, "row pos"] = torch.cat(masks, dim=0)
    if activations is not None:
        assert mask.shape == activations.shape
    return mask
#%%
def _plot_topk(
    all_activations: Float[Tensor, "row pos layer"], layer: int = 0, k: int = 10, largest: bool = True,
    window_size: int = 10, centred: bool = True, inclusions: Iterable[str] = None, exclusions: Iterable[str] = None,
    verbose: bool = False,
):
    assert not (inclusions is not None and exclusions is not None)
    label = "positive" if largest else "negative"
    layers = all_activations.shape[-1]
    if verbose:
        print(f"Plotting top {k} {label} examples for layer {layer}")
    activations: Float[Tensor, "row pos"] = all_activations[:, :, layer]
    if largest:
        ignore_value = torch.tensor(-np.inf, device=device, dtype=torch.float32)
    else:
        ignore_value = torch.tensor(np.inf, device=device, dtype=torch.float32)
    # create a mask for the inclusions/exclusions
    if exclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(exclusions, all_activations)
        masked_activations = activations.where(~mask, other=ignore_value)
    elif inclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(inclusions, all_activations)
        assert mask.sum() >= k, (
            f"Only {mask.sum()} positions match the inclusions, but {k} are required"
        )
        if verbose:
            print(f"Including {mask.sum()} positions")
        masked_activations = activations.where(mask, other=ignore_value)
    else:
        masked_activations = activations
    top_k_return = torch.topk(masked_activations.flatten(), k=k, largest=largest)
    assert torch.isfinite(top_k_return.values).all()
    topk_pos_indices = top_k_return.indices
    topk_pos_indices = np.array(np.unravel_index(topk_pos_indices.cpu().numpy(), masked_activations.shape)).T.tolist()
    # Get the examples and their activations corresponding to the most positive and negative activations
    topk_pos_examples = [dataloader.dataset[b]['tokens'][s].item() for b, s in topk_pos_indices]
    topk_pos_activations = [masked_activations[b, s].item() for b, s in topk_pos_indices]
    # Print the  most positive and negative examples and their activations
    print(f"Top {k} most {label} examples:")
    zeros = torch.zeros((1, layers), device=device, dtype=torch.float32)
    texts = [model.tokenizer.bos_token]
    text_to_not_repeat = set()
    acts = [zeros]
    text_sep = "\n"
    topk_zip = zip(topk_pos_indices, topk_pos_examples, topk_pos_activations)
    for index, example, activation in topk_zip:
        example_str = model.to_string(example)
        if inclusions is not None:
            assert example_str in inclusions, f"Example '{example_str}' not in inclusions {inclusions}"
        if exclusions is not None:
            assert example_str not in exclusions, f"Example '{example_str}' in exclusions {exclusions}"
        batch, pos = index
        text_window: List[str] = extract_text_window(batch, pos, window_size=window_size)
        activation_window: Float[Tensor, "pos layer"] = extract_activations_window(
            all_activations, batch, pos, window_size=window_size
        )
        assert len(text_window) == activation_window.shape[0], (
            f"Initially text window length {len(text_window)} does not match activation window length {activation_window.shape[0]}"
        )
        text_flat = "".join(text_window)
        if text_flat in text_to_not_repeat:
            continue
        text_to_not_repeat.add(text_flat)
        print(f"Example: {model.to_string(example)}, Activation: {activation:.4f}, Batch: {batch}, Pos: {pos}")
        text_window.append(text_sep)
        activation_window = torch.cat([activation_window, zeros], dim=0)
        assert len(text_window) == activation_window.shape[0]
        texts += text_window
        acts.append(activation_window)
    acts_cat = einops.repeat(torch.cat(acts, dim=0), "pos layer -> pos layer 1")
    assert acts_cat.shape[0] == len(texts)
    html = plot_neuroscope(texts, centred=centred, activations=acts_cat, verbose=False)
    exclusion_suffix = "_w_exclusions" if exclusions is not None else ""
    file_name = f"top_{k}_most_{label}_layer_{layer}_sentiment{exclusion_suffix}.html"
    save_html(html, file_name, model)
    display(html)
#%%
def plot_topk(
    activations: Float[Tensor, "row pos layer"], k: int = 10, layer: int = 0,
    window_size: int = 10, centred: bool = True, 
    inclusions: Iterable[str] = None, exclusions: Iterable[str] = None,
    verbose: bool = False,
):
   _plot_topk(
       activations, layer=layer, k=k, largest=True, 
       window_size=window_size, centred=centred, 
       inclusions=inclusions, exclusions=exclusions, verbose=verbose
    )
   _plot_topk(
       activations, layer=layer, k=k, largest=False, 
       window_size=window_size, centred=centred, 
       inclusions=inclusions, exclusions=exclusions, verbose=verbose
    )
# %%
# plot_topk(sentiment_activations, k=50, layer=0)
# # %%
# plot_topk(sentiment_activations, k=50, layer=6, window_size=20, centred=True)
# # %%
# plot_topk(sentiment_activations, k=50, layer=12, window_size=20, centred=True)
# %%
# ============================================================================ #
# Top p sampling
#%%
#%%
def _plot_top_p(
    all_activations: Float[Tensor, "row pos layer"], layer: int = 0, p: float = 0.1, k: int = 10, largest: bool = True,
    window_size: int = 10, centred: bool = True, inclusions: Iterable[str] = None, exclusions: Iterable[str] = None,
):
    assert not (inclusions is not None and exclusions is not None)
    label = "positive" if largest else "negative"
    activations: Float[Tensor, "batch pos"] = all_activations[:, :, layer]
    if largest:
        ignore_value = torch.tensor(-np.inf, device=device, dtype=torch.float32)
    else:
        ignore_value = torch.tensor(np.inf, device=device, dtype=torch.float32)
    # create a mask for the inclusions/exclusions
    if exclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(exclusions, all_activations)
        masked_activations = activations.where(~mask, other=ignore_value)
    elif inclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(inclusions, all_activations)
        masked_activations = activations.where(mask, other=ignore_value)
    else:
        masked_activations = activations

    activations_flat: Float[Tensor, "(batch pos)"] = masked_activations.flatten()
    sample_size = int(p * len(activations_flat))
    top_p_indices = torch.topk(activations_flat, k=sample_size, largest=largest).indices
    sampled_indices = top_p_indices[torch.randperm(sample_size)[:k]]
    top_p_indices = np.array(np.unravel_index(sampled_indices.cpu().numpy(), masked_activations.shape)).T.tolist()
    top_p_examples = [dataloader.dataset[b]['tokens'][s].item() for b, s in top_p_indices]
    top_p_activations = [masked_activations[b, s].item() for b, s in top_p_indices]
    # Print the  most positive and negative examples and their activations
    print(f"Top {k} most {label} examples:")
    zeros = torch.zeros((1, all_activations.shape[-1]), device=device, dtype=torch.float32)
    texts = [model.tokenizer.bos_token]
    text_to_not_repeat = set()
    acts = [zeros]
    text_sep = "\n"
    topk_zip = zip(top_p_indices, top_p_examples, top_p_activations)
    for index, example, activation in topk_zip:
        batch, pos = index
        text_window: List[str] = extract_text_window(batch, pos, window_size=window_size)
        activation_window: Float[Tensor, "pos layer"] = extract_activations_window(
            all_activations, batch, pos, window_size=window_size
        )
        assert len(text_window) == activation_window.shape[0], (
            f"Initially text window length {len(text_window)} "
            f"does not match activation window length {activation_window.shape[0]}"
        )
        text_flat = "".join(text_window)
        if text_flat in text_to_not_repeat:
            continue
        text_to_not_repeat.add(text_flat)
        print(
            f"Example: {model.to_string(example)}, Activation: {activation:.4f}, Batch: {batch}, Pos: {pos}"
        )
        text_window.append(text_sep)
        activation_window = torch.cat([activation_window, zeros], dim=0)
        assert len(text_window) == activation_window.shape[0]
        texts += text_window
        acts.append(activation_window)
    acts_cat = einops.repeat(torch.cat(acts, dim=0), "pos layer -> pos layer 1")
    assert acts_cat.shape[0] == len(texts)
    html = plot_neuroscope(texts, centred=centred, activations=acts_cat, verbose=False)
    save_html(html, f"top_{p * 100:.0f}pc_most_{label}_layer_{layer}_sentiment.html", model)
    display(html)
#%%
def plot_top_p(
    activations: Float[Tensor, "row pos layer"], k: int = 10, p: float = 0.1, layer: int = 0,
    window_size: int = 10, centred: bool = True,  inclusions: Iterable[str] = None, exclusions: Iterable[str] = None,
):
   _plot_top_p(activations, layer=layer, p=p, k=k, largest=True, window_size=window_size, centred=centred, inclusions=inclusions, exclusions=exclusions)
   _plot_top_p(activations, layer=layer, p=p, k=k, largest=False, window_size=window_size, centred=centred, inclusions=inclusions, exclusions=exclusions)
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
    # worth investigating?
    # ' Trek', ' Yorkshire', 'Puerto', 'Celsius', 'Linux', 'Reuters', 'Romania', 'Gary',
    # more interesting ones
    'adequate', 'truly', 'mis', 'dys', 'provides', 'offers', 'fully', 'Flint', 'migraine',  
    'really', 'considerable', 'reasonably', 'substantial', 'additional', 'STD', 'Fukushima',
    'Narcolepsy', 'Tooth', 'RUDE', 'Diagnostic', 'Kissinger', '!', 'Obama', 'Assad', 'Gaza',
    'CIA', 'BP', 'istan', 'VICE', 'TSA', 'Mitt', 'Romney', 'Afghanistan', 'Kurd', 'Molly',
    'agoraphobia', 'greenhouse', 'DoS', 'Medicaid', 
    # the rest
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
    'disease', 'welts', 'trojans', 'trojan', 'makeshift', 'cant', 


]
exclusions = expand_exclusions(exclusions)
# %%
plot_top_p(sentiment_activations, p=.02, k=50, layer=1, exclusions=exclusions)
# %%
# plot_topk(sentiment_activations, k=50, layer=1, exclusions=exclusions)
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
    mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(tokens, all_activations)
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
        height=100 * (idx + 1)
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
pos_neg_dict = {
    "positive": pos_list,
    "negative": neg_list,
    "surprising_proper_nouns": [" Trek", " Yorkshire", " Linux", " Reuters"],
    "trek": [" Trek"],
    "yorkshire": [" Yorkshire"],
    "linux": [" Linux"],
    "reuters": [" Reuters"],
    "first_names": [" John", " Mary", " Bob", " Alice"],
    "places": [" London", " Paris", " Tokyo"],
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
# plot_topk(sentiment_activations, k=50, layer=1, inclusions=neg_list)
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
plot_top_mean_variance(token_counts, token_means, token_std_devs, model=model, k=10)
# %%
plot_topk(sentiment_activations, k=10, layer=1, inclusions=[" Yorkshire"], window_size=20)
#%%
# "ression" as in "impression" vs. "aggression", "repression"
# 'leasing' as in 'releasing' vs. 'subleasing'
# 'ciplinary' as in 'multidisciplinary' vs. 'disciplinary'
# 'rieved': 'grieved' is more negative than 'aggrieved'
# 'byte': 'overbyte' vs. 'byte'
# 'risons: 'Comparisons' vs. 'Prisons'
# 'Ds': 'TZDs' (diabetes drug) vs. 'STDs'
# 'ested': 'attested' vs. 'molested'
# ' Hare' as in ' Hare Krishna vs. ' Harem'
# ' Toro' as in ' Guillermo del Toro' vs ' Toro Rosso'

#%%
# %%
