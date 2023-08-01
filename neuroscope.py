#%%
import gc
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
import einops
from jaxtyping import Float, Int
from typing import List, Union
from transformer_lens import HookedTransformer
from transformer_lens.evals import make_owt_data_loader
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name
from circuitsvis.activations import text_neuron_activations
from tqdm.notebook import tqdm
from IPython.display import display
from utils.store import load_array
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
def plot_neuroscope(text: str):
    activations_by_layer = []
    for layer in range(model.cfg.n_layers):
        act_name = get_act_name('resid_post', layer)
        _, cache = model.run_with_cache(
            text, names_filter=lambda name: name == act_name
        )
        embeddings: Int[Tensor, "batch pos d_model"] = cache[act_name]
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        activations = einops.einsum(
            embeddings.to(device), sentiment_dir.to(device),
            "batch pos d_model, d_model -> batch pos"
        )
        activations = einops.rearrange(activations, "batch pos -> pos batch")
        activations_by_layer.append(activations)
    text_activations: Float[Tensor, "pos layer 1"] = torch.stack(activations_by_layer, dim=1)
    return text_neuron_activations(
        tokens=model.to_str_tokens(text), 
        activations=text_activations,
        # first_dimension_name="Layer",
        second_dimension_name="Model",
        # first_dimension_labels=["embed_and_mlp0"],
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
harry_potter_neuroscope = plot_neuroscope(harry_potter_start)
#%%
with open("data/gpt2-small/harry_potter_neuroscope.html", "w") as f:
    f.write(str(harry_potter_neuroscope))
#%%
harry_potter_neuroscope
#%%
# ============================================================================ #
# Prefixes

#%%
common_words_cr = [
    ' crony', ' crump', ' crinkle', ' craggy', ' cramp', ' crumb', ' crayon'
]
cr_single_tokens = []
for word in common_words_cr:
    if model.to_str_tokens(word, prepend_bos=False)[0] == " cr":
        print(word, model.to_str_tokens(word, prepend_bos=False))
        cr_single_tokens.append(word)
cr_single_tokens = list(set(cr_single_tokens))
cr_single_tokens
#%%
cr_text = "\n".join(cr_single_tokens)
plot_neuroscope(cr_text)
#%%
common_words_clo = [
    ' clopped', ' cloze', ' cloistered', ' clopping', ' cloacal', ' cloister'
]
clo_single_tokens = []
for word in common_words_clo:
    if model.to_str_tokens(word, prepend_bos=False)[0] == " clo":
        print(word, model.to_str_tokens(word, prepend_bos=False))
        clo_single_tokens.append(word)
clo_single_tokens = list(set(clo_single_tokens))
clo_single_tokens
#%%
clo_text = "\n".join(clo_single_tokens)
plot_neuroscope(clo_text)
#%%
# ============================================================================ #
# Max activating examples on training data
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
def get_activations_from_data(data: torch.utils.data.dataloader.DataLoader, layer: int = 0):
    all_activations = []
    for batch_idx, batch_value in tqdm(enumerate(data), total=len(data)):
        act_name = get_act_name('resid_post', layer)
        _, cache = model.run_with_cache(
            batch_value['tokens'], names_filter=lambda name: name == act_name
        )
        embeddings: Int[Tensor, "batch pos d_model"] = cache[act_name]
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        activations: Float[Tensor, "batch pos"] = einops.einsum(
            embeddings, sentiment_dir,
            "batch pos d_model, d_model -> batch pos"
        ).to(device="cpu")
        all_activations.append(activations)
        del embeddings
    # Concatenate the activations into a single tensor
    all_activations: Float[Tensor, "full_batch pos"] = torch.cat(all_activations, dim=0)
    return all_activations
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
with ClearCache():
    all_activations = get_activations_from_data(dataloader)
all_activations.shape
#%%
def extract_example(batch: int, pos: int, window_size: int = 10):
    lb = max(0, pos - window_size)
    ub = min(len(dataloader.dataset[batch]['tokens']), pos + window_size)
    return model.to_string(dataloader.dataset[batch]['tokens'][lb:ub])
#%%
def _plot_topk(k: int = 10, largest: bool = True):
    label = "positive" if largest else "negative"
    topk_pos_indices = torch.topk(all_activations.flatten(), k=k, largest=largest).indices
    topk_pos_indices = np.array(np.unravel_index(topk_pos_indices.cpu().numpy(), all_activations.shape)).T.tolist()
    # Get the examples and their activations corresponding to the most positive and negative activations
    topk_pos_examples = [dataloader.dataset[b]['tokens'][s].item() for b, s in topk_pos_indices]
    topk_pos_activations = [all_activations[b, s].item() for b, s in topk_pos_indices]
    # Print the  most positive and negative examples and their activations
    print(f"Top {k} most {label} examples:")
    texts = []
    for index, example, activation in zip(topk_pos_indices, topk_pos_examples, topk_pos_activations):
        batch, pos = index
        print(f"Example: {model.to_string(example)}, Activation: {activation:.4f}, Batch: {batch}, Pos: {pos}")
        texts.append(extract_example(batch, pos))
    texts_cat = '\n'.join(texts)
    display(plot_neuroscope(texts_cat))
#%%
def plot_topk(k: int = 10):
   _plot_topk(k=k, largest=True)
   _plot_topk(k=k, largest=False)
# %%
plot_topk(k=50)
# %%
