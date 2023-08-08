#%%
import gc
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
import einops
from jaxtyping import Float, Int
from typing import List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.evals import make_owt_data_loader
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name
from circuitsvis.activations import text_neuron_activations
from tqdm.notebook import tqdm
from IPython.display import display
from utils.store import load_array, save_html, save_array, is_file
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
#%%
save_html(harry_potter_neuroscope, "harry_potter_neuroscope", model)
#%%
harry_potter_neuroscope
#%%
# ============================================================================ #
# Prefixes

#%%
common_words_cr = [
    ' crony', ' crump', ' crinkle', ' craggy', ' cramp', ' crumb', ' crayon', ' cringing', ' cramping'
]
cr_single_tokens = []
for word in common_words_cr:
    if model.to_str_tokens(word, prepend_bos=False)[0] == " cr":
        # print(word, model.to_str_tokens(word, prepend_bos=False))
        cr_single_tokens.append(word)
cr_single_tokens = list(set(cr_single_tokens))
# cr_single_tokens
#%%
cr_text = "\n".join(cr_single_tokens)
plot_neuroscope(cr_text, centred=True)
#%%
common_words_clo = [
    ' clopped', ' cloze', ' cloistered', ' clopping', ' cloacal', ' cloister', ' cloaca',
]
clo_single_tokens = []
for word in common_words_clo:
    if model.to_str_tokens(word, prepend_bos=False)[0] == " clo":
        # print(word, model.to_str_tokens(word, prepend_bos=False))
        clo_single_tokens.append(word)
clo_single_tokens = list(set(clo_single_tokens))
# clo_single_tokens
#%%
clo_text = "\n".join(clo_single_tokens)
plot_neuroscope(clo_text, centred=True)
#%%
# ============================================================================ #
# Negations
negation_text = "\nText 1: Please don't plague me with your nonsense. I can't bear it when you do that. You're not a good person. I don't like you. I'm not going to do that. I won't do that. I can't do that. \nText 2: It's not bad at all. You didn't get derailed. You didn't rely on stereotypes. There is no stigma in your words. You are a great writer."
plot_neuroscope(negation_text, centred=True, verbose=False)
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
    return model.to_str_tokens(dataloader.dataset[batch]['tokens'][lb:ub], prepend_bos=False)
#%%
def extract_activations_window(
    activations: Float[Tensor, "row pos layer"], 
    batch: int, pos: int, window_size: int = 10,
) -> Float[Tensor, "pos layer"]:
    lb, ub = get_window(batch, pos, window_size)
    return activations[batch, lb:ub, :]

#%%
def _plot_topk(
    all_activations: Float[Tensor, "row pos layer"], layer: int = 0, k: int = 10, largest: bool = True,
    window_size: int = 10, centred: bool = True,
):
    label = "positive" if largest else "negative"
    activations = all_activations[:, :, layer]
    topk_pos_indices = torch.topk(activations.flatten(), k=k, largest=largest).indices
    topk_pos_indices = np.array(np.unravel_index(topk_pos_indices.cpu().numpy(), activations.shape)).T.tolist()
    # Get the examples and their activations corresponding to the most positive and negative activations
    topk_pos_examples = [dataloader.dataset[b]['tokens'][s].item() for b, s in topk_pos_indices]
    topk_pos_activations = [activations[b, s].item() for b, s in topk_pos_indices]
    # Print the  most positive and negative examples and their activations
    print(f"Top {k} most {label} examples:")
    zeros = torch.zeros((1, all_activations.shape[-1]), device=device, dtype=torch.float32)
    texts = [model.tokenizer.bos_token]
    text_to_not_repeat = set()
    acts = [zeros]
    text_sep = "\n"
    topk_zip = zip(topk_pos_indices, topk_pos_examples, topk_pos_activations)
    for index, example, activation in topk_zip:
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
    save_html(html, f"top_{k}_most_{label}_layer_{layer}_sentiment.html", model)
    display(html)
#%%
def plot_topk(
    activations: Float[Tensor, "row pos layer"], k: int = 10, layer: int = 0,
    window_size: int = 10, centred: bool = True,
):
   _plot_topk(activations, layer=layer, k=k, largest=True, window_size=window_size, centred=centred)
   _plot_topk(activations, layer=layer, k=k, largest=False, window_size=window_size, centred=centred)
# %%
plot_topk(sentiment_activations, k=50, layer=0)
# %%
plot_topk(sentiment_activations, k=50, layer=6, window_size=20, centred=True)
# %%
plot_topk(sentiment_activations, k=50, layer=12, window_size=20, centred=True)
# %%
# ============================================================================ #
# Top p sampling
#%%
#%%
def _plot_top_p(
    all_activations: Float[Tensor, "row pos layer"], layer: int = 0, p: float = 0.1, k: int = 10, largest: bool = True,
    window_size: int = 10, centred: bool = True,
):
    label = "positive" if largest else "negative"
    activations: Float[Tensor, "batch pos"] = all_activations[:, :, layer]
    activations_flat: Float[Tensor, "(batch pos)"] = activations.flatten()
    sample_size = int(p * len(activations_flat))
    top_p_indices = torch.topk(activations_flat, k=sample_size, largest=largest).indices
    sampled_indices = top_p_indices[torch.randperm(sample_size)[:k]]
    top_p_indices = np.array(np.unravel_index(sampled_indices.cpu().numpy(), activations.shape)).T.tolist()
    top_p_examples = [dataloader.dataset[b]['tokens'][s].item() for b, s in top_p_indices]
    top_p_activations = [activations[b, s].item() for b, s in top_p_indices]
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
    window_size: int = 10, centred: bool = True,
):
   _plot_top_p(activations, layer=layer, p=p, k=k, largest=True, window_size=window_size, centred=centred)
   _plot_top_p(activations, layer=layer, p=p, k=k, largest=False, window_size=window_size, centred=centred)
#%%
plot_top_p(sentiment_activations, k=50, layer=0, p=0.01)
#%%

