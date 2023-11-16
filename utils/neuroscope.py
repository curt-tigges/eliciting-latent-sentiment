from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import einops
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate, get_act_name
from circuitsvis.activations import text_neuron_activations
from IPython.display import display
from tqdm.auto import tqdm
from utils.store import save_html, save_array, load_array, is_file


# Harry Potter in English
harry_potter_start = """
    Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense.

    Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere.

    The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years; in fact, Mrs. Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that.

    When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.

    None of them noticed a large, tawny owl flutter past the window.

    At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. “Little tyke,” chortled Mr. Dursley as he left the house. He got into his car and backed out of number four's drive.

    It was on the corner of the street that he noticed the first sign of something peculiar — a cat reading a map. For a second, Mr. Dursley didn't realize what he had seen — then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn't read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day.

    But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn't help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr. Dursley couldn't bear people who dressed in funny clothes — the getups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren't young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt — these people were obviously collecting for something . . . yes, that would be it. The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills.

    Mr. Dursley always sat with his back to the window in his office on the ninth floor. If he hadn't, height have found it harder to concentrate on drills that morning. He didn't see the owls swooping past in broad daylight, though people down in the street did; they pointed and gazed open-mouthed as owl after owl sped overhead. Most of them had never seen an owl even at nighttime. Mr. Dursley, however, had a perfectly normal, owl-free morning. He yelled at five different people. He made several important telephone calls and shouted a bit more. He was in a very good mood until lunchtime, when he thought he'd stretch his legs and walk across the road to buy himself a bun from the bakery.

    He'd forgotten all about the people in cloaks until he passed a group of them next to the baker's. He eyed them angrily as he passed. He didn't know why, but they made him uneasy. This bunch were whispering excitedly, too, and he couldn't see a single collecting tin. It was on his way back past them, clutching a large doughnut in a bag, that he caught a few words of what they were saying.
"""

# Harry Potter in French
harry_potter_fr_start = """
Mr et Mrs Dursley, qui habitaient au 4, Privet Drive, avaient toujours affirmé avec la plus grande fierté qu'ils étaient parfaitement normaux, merci pour eux. Jamais quiconque n'aurait imaginé qu'ils puissent se trouver impliqués dans quoi que ce soit d'étrange ou de mystérieux. Ils n'avaient pas de temps à perdre avec des sornettes.

Mr Dursley dirigeait la Grunnings, une entreprise qui fabriquait des perceuses. C'était un homme grand et massif, qui n'avait pratiquement pas de cou, mais possédait en revanche une moustache de belle taille. Mrs Dursley, quant à elle, était mince et blonde et disposait d'un cou deux fois plus long que la moyenne, ce qui lui était fort utile pour espionner ses voisins en regardant par-dessus les clôtures des jardins. Les Dursley avaient un petit garçon prénommé Dudley et c'était à leurs yeux le plus bel enfant du monde.

Les Dursley avaient tout ce qu'ils voulaient. La seule chose indésirable qu'ils possédaient, c'était un secret dont ils craignaient plus que tout qu'on le découvre un jour. Si jamais quiconque venait à entendre parler des Potter, ils étaient convaincus qu'ils ne s'en remettraient pas. Mrs Potter était la soeur de Mrs Dursley, mais toutes deux ne s'étaient plus revues depuis des années. En fait, Mrs Dursley faisait comme si elle était fille unique, car sa soeur et son bon à rien de mari étaient aussi éloignés que possible de tout ce qui faisait un Dursley. Les Dursley tremblaient d'épouvante à la pensée de ce que diraient les voisins si par malheur les Potter se montraient dans leur rue. Ils savaient que les Potter, eux aussi, avaient un petit garçon, mais ils ne l'avaient jamais vu. Son existence constituait une raison supplémentaire de tenir les Potter à distance: il n'était pas question que le petit Dudley se mette à fréquenter un enfant comme celui-là.

Lorsque Mr et Mrs Dursley s'éveillèrent, au matin du mardi où commence cette histoire, il faisait gris et triste et rien dans le ciel nuageux ne laissait prévoir que des choses étranges et mystérieuses allaient bientôt se produire dans tout le pays. Mr Dursley fredonnait un air en nouant sa cravate la plus sinistre pour aller travailler et Mrs Dursley racontait d'un ton badin les derniers potins du quartier en s'efforçant d'installer sur sa chaise de bébé le jeune Dudley qui braillait de toute la force de ses poumons.
"""


def get_dataloader(
    model: HookedTransformer,
    name: str = "stas/openwebtext-10k",
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = True,
):
    """Wrapper around HF's load_dataset to quickly get a dataloader"""
    owt_data = load_dataset(name, split=split)
    dataset = tokenize_and_concatenate(owt_data, model.tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return data_loader


def get_activations_from_dataloader(
    data: torch.utils.data.dataloader.DataLoader,
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Float[Tensor, "row pos"]:
    if device is None:
        device = model.cfg.device
    direction = direction.to(device=device)
    acts_list: list = []
    for batch_idx, batch_value in tqdm(enumerate(data), total=len(data)):
        batch_tokens = batch_value["tokens"].to(device)
        batch_acts: Float[Tensor, "batch pos layer"] = get_projections_for_text(
            batch_tokens, direction, model
        )
        acts_list.append(batch_acts)
        if max_batches is not None and batch_idx >= max_batches:
            break
    # Concatenate the activations into a single tensor
    acts_tensor: Float[Tensor, "row pos layer"] = torch.cat(acts_list, dim=0)
    return acts_tensor


# %%
def get_activations_cached(
    data: torch.utils.data.dataloader.DataLoader,
    direction_label: str,
    model: HookedTransformer,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    path = direction_label + "_activations.npy"
    if is_file(path, model):
        if verbose:
            print("Loading activations from file")
        sentiment_activations = load_array(path, model)
        sentiment_activations: Float[Tensor, "row pos layer"] = torch.tensor(
            sentiment_activations, device=device, dtype=torch.float32
        )
    else:
        if verbose:
            print("Computing activations")
        direction = load_array(direction_label + ".npy", model)
        direction = torch.tensor(direction, device=device, dtype=torch.float32)
        direction /= direction.norm()
        sentiment_activations: Float[
            Tensor, "row pos layer"
        ] = get_activations_from_dataloader(data, direction, model)
        save_array(sentiment_activations, path, model)
    return sentiment_activations


def names_filter(name: str):
    """Filter for the names of the activations we want to keep to study the resid stream."""
    return name.endswith("resid_post") or name == get_act_name("resid_pre", 0)


def get_projections_for_text(
    tokens: Int[Tensor, "batch pos"],
    special_dir: Float[Tensor, "d_model"],
    model: HookedTransformer,
    device: str = "cpu",
    prepend_bos: bool = False,
) -> Float[Tensor, "batch pos layer"]:
    """Computes residual stream projections across all layers and positions for a given text."""
    special_dir /= special_dir.norm()
    _, cache = model.run_with_cache(
        tokens,
        names_filter=names_filter,
        prepend_bos=prepend_bos,
    )
    acts_by_layer = []
    for layer in range(-1, model.cfg.n_layers):
        if layer >= 0:
            emb: Int[Tensor, "batch pos d_model"] = cache["resid_post", layer]
        else:
            emb: Int[Tensor, "batch pos d_model"] = cache["resid_pre", 0]
        emb /= emb.norm(dim=-1, keepdim=True)
        act: Float[Tensor, "batch pos"] = einops.einsum(
            emb, special_dir, "batch pos d_model, d_model -> batch pos"
        ).to(device=device)
        acts_by_layer.append(act)
    acts_tensor: Float[Tensor, "batch pos layer"] = torch.stack(acts_by_layer, dim=2)
    return acts_tensor


def plot_neuroscope(
    text: Union[str, List[str]],
    model: HookedTransformer,
    centred: bool,
    activations: Optional[Float[Tensor, "pos layer 1"]] = None,
    special_dir: Optional[Float[Tensor, "d_model"]] = None,
    verbose: bool = False,
    default_layer: Union[Literal["all"], int, List[int]] = 1,
    show_selectors: bool = True,
    prepend_bos: bool = True,
):
    """
    Wrapper around CircuitVis's `text_neuron_activations`.
    Performs centring if `centred` is True.
    Computes activations based on projecting onto a resid stream direction if not provided.
    If you don't need centring or projection, use `text_neuron_activations` directly.
    The keyword "all" for default_layer is used to show all layers at once.
    """
    assert activations is not None or special_dir is not None
    tokens: Int[Tensor, "1 pos"] = model.to_tokens(text, prepend_bos=prepend_bos)
    if verbose:
        print(f"Tokens shape: {tokens.shape}")
    if activations is None:
        assert special_dir is not None
        if verbose:
            print("Computing activations")
        activations: Float[Tensor, "1 pos layer"] = get_projections_for_text(
            tokens, special_dir=special_dir, model=model
        )
        activations: Float[Tensor, "pos layer 1"] = einops.rearrange(
            activations, "1 pos layer -> pos layer 1"
        )
        if verbose:
            print(f"Activations shape: {activations.shape}")
    if centred:
        if verbose:
            print("Centering activations")
        layer_means = einops.reduce(
            activations, "pos layer 1 -> 1 layer 1", reduction="mean"
        )
        layer_means = einops.repeat(
            layer_means, "1 layer 1 -> pos layer 1", pos=activations.shape[0]
        )
        activations -= layer_means
    elif verbose:
        print("Activations already centered")
    assert activations.ndim == 3, (
        f"activations must be of shape [tokens x layers x neurons], "
        f"found {activations.shape}"
    )

    if isinstance(text, str):
        str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
    else:
        str_tokens = text

    if isinstance(default_layer, list) or isinstance(default_layer, str):
        assert prepend_bos
        if default_layer == "all":
            layers = list(range(activations.shape[1]))
        else:
            layers = default_layer
        orig_len = len(str_tokens)
        numbered_tokens = []
        for layer in layers:
            numbered_tokens += [f"L{layer:02d} "] + str_tokens[1:] + ["\n"]
        str_tokens = numbered_tokens
        activations = torch.cat(
            [activations, torch.zeros_like(activations[:1])], dim=0
        )  # Appending 0 activation for the newlines
        activations = activations[:, layers, :]
        activations = einops.rearrange(activations, "pos layer 1 -> (layer pos) 1 1")
        first_dimension_labels = ["multi"]
        default_layer = 0
        assert len(str_tokens) == (orig_len + 1) * len(layers)
        assert len(activations) == (orig_len + 1) * len(layers)
    else:
        first_dimension_labels = [f"{i}_pre" for i in range(activations.shape[1])]

    # Check shapes before calling text_neuron_activations()
    assert len(str_tokens) == activations.shape[0], (
        "tokens and activations must have the same length, found "
        f"tokens={len(str_tokens)} and acts={activations.shape[0]}, "
        f"tokens={str_tokens}, "
        f"activations={activations.shape}"
    )
    if verbose:
        print(
            "Calling text_neuron_activations with "
            f"tokens={len(str_tokens)} and acts={activations.shape}"
            f"first_dimension_labels={first_dimension_labels}, "
            f"first_dimension_default={default_layer},"
            f"show_selectors={show_selectors}"
        )
    return text_neuron_activations(
        tokens=str_tokens,
        activations=activations,
        first_dimension_name="Layer",
        first_dimension_labels=first_dimension_labels,
        second_dimension_name="Model",
        second_dimension_labels=[model.cfg.model_name],
        first_dimension_default=default_layer,
        show_selectors=show_selectors,
    )


def get_window(
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    window_size: int = 10,
) -> Tuple[int, int]:
    """Helper function to get the window around a position in a batch (used in topk plotting))"""
    lb = max(0, pos - window_size)
    ub = min(len(dataloader.dataset[batch]["tokens"]), pos + window_size)
    return lb, ub


def extract_text_window(
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    window_size: int = 10,
) -> List[str]:
    """Helper function to get the text window around a position in a batch (used in topk plotting)"""
    lb, ub = get_window(batch, pos, dataloader=dataloader, window_size=window_size)
    tokens = dataloader.dataset[batch]["tokens"][lb:ub]
    str_tokens = model.to_str_tokens(tokens, prepend_bos=False)
    return str_tokens


def extract_activations_window(
    activations: Float[Tensor, "row pos layer"],
    batch: int,
    pos: int,
    dataloader: torch.utils.data.DataLoader,
    window_size: int = 10,
) -> Float[Tensor, "pos layer"]:
    """Helper function to get the activations window around a position in a batch (used in topk plotting)"""
    lb, ub = get_window(batch, pos, dataloader=dataloader, window_size=window_size)
    return activations[batch, lb:ub, :]


def get_batch_pos_mask(
    tokens: Union[str, List[str], Tensor],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    activations: Optional[Float[Tensor, "row pos"]] = None,
    device: Optional[str] = None,
):
    """Helper function to get a mask for inclusions/exclusions (used in topk plotting)"""
    if device is None and activations is not None:
        device = activations.device
    elif device is None:
        device = model.cfg.device
    mask_values: Int[Tensor, "words"] = torch.unique(
        model.to_tokens(tokens, prepend_bos=False).flatten()
    ).to(device)
    masks = []
    for batch_idx, batch_value in enumerate(dataloader):
        batch_tokens: Int[Tensor, "batch_size pos 1"] = (
            batch_value["tokens"].to(device).unsqueeze(-1)
        )
        batch_mask: Bool[Tensor, "batch_size pos"] = torch.any(
            batch_tokens == mask_values, dim=-1
        )
        masks.append(batch_mask)
    mask: Bool[Tensor, "row pos"] = torch.cat(masks, dim=0)
    if activations is not None:
        assert mask.shape == activations.shape[:2]
        assert mask.device == activations.device
    return mask


def _plot_topk(
    all_activations: Float[Tensor, "row pos layer"],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    layer: int = 0,
    k: int = 10,
    largest: bool = True,
    window_size: int = 10,
    centred: bool = True,
    inclusions: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    verbose: bool = False,
    base_layer: Optional[int] = None,
):
    """
    One-sided topk plotting.
    Main entrypoint should be `plot_topk`.
    """
    assert model.tokenizer is not None
    device = all_activations.device
    assert not (inclusions is not None and exclusions is not None)
    label = "positive" if largest else "negative"
    layers = all_activations.shape[-1]
    if verbose:
        print(f"Plotting top {k} {label} examples for layer {layer}")
    activations: Float[Tensor, "row pos"] = all_activations[:, :, layer]
    if base_layer is not None:
        base_activations: Float[Tensor, "row pos"] = all_activations[:, :, base_layer]
        activations = activations - base_activations
    if largest:
        ignore_value = torch.tensor(-np.inf, device=device, dtype=torch.float32)
    else:
        ignore_value = torch.tensor(np.inf, device=device, dtype=torch.float32)
    # create a mask for the inclusions/exclusions
    if exclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(
            exclusions, dataloader, model, all_activations
        )
        masked_activations = activations.where(~mask, other=ignore_value)
    elif inclusions is not None:
        for incl in inclusions:
            model.to_single_token(incl)
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(
            inclusions, dataloader, model, all_activations
        )
        assert (
            mask.sum() >= k
        ), f"Only {mask.sum()} positions match the inclusions, but {k} are required"
        if verbose:
            print(f"Including {mask.sum()} positions")
        masked_activations = activations.where(mask, other=ignore_value)
    else:
        masked_activations = activations
    top_k_return = torch.topk(masked_activations.flatten(), k=k, largest=largest)
    assert torch.isfinite(top_k_return.values).all()
    topk_pos_indices = top_k_return.indices
    topk_pos_indices = np.array(
        np.unravel_index(topk_pos_indices.cpu().numpy(), masked_activations.shape)
    ).T.tolist()
    # Get the examples and their activations corresponding to the most positive and negative activations
    topk_pos_examples = [
        dataloader.dataset[b]["tokens"][s].item() for b, s in topk_pos_indices
    ]
    topk_pos_activations = [
        masked_activations[b, s].item() for b, s in topk_pos_indices
    ]
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
            assert (
                example_str in inclusions
            ), f"Example '{example_str}' not in inclusions {inclusions}"
        if exclusions is not None:
            assert (
                example_str not in exclusions
            ), f"Example '{example_str}' in exclusions {exclusions}"
        batch, pos = index
        text_window: List[str] = extract_text_window(
            batch, pos, dataloader=dataloader, model=model, window_size=window_size
        )
        activation_window: Float[Tensor, "pos layer"] = extract_activations_window(
            all_activations, batch, pos, window_size=window_size, dataloader=dataloader
        )
        assert len(text_window) == activation_window.shape[0], (
            f"Initially text window length {len(text_window)} does not match "
            f"activation window length {activation_window.shape[0]}"
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
    html = plot_neuroscope(
        texts, model=model, centred=centred, activations=acts_cat, verbose=False
    )
    layer_suffix = f"_layer_{layer}" if layer is not None else ""
    exclusion_suffix = "_w_exclusions" if exclusions is not None else ""
    inclusion_suffix = "_w_inclusions" if inclusions is not None else ""
    base_layer_suffix = f"_base_layer_{base_layer}" if base_layer is not None else ""
    suffices = layer_suffix + exclusion_suffix + inclusion_suffix + base_layer_suffix
    file_name = f"top_{k}_most_{label}_sentiment{suffices}.html"
    save_html(html, file_name, model)
    display(html)


def plot_topk(
    activations: Float[Tensor, "row pos layer"],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    k: int = 10,
    layer: int = 0,
    window_size: int = 10,
    centred: bool = True,
    inclusions: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    verbose: bool = False,
    base_layer: Optional[int] = None,
):
    """
    Main entrypoint for topk plotting. Plots both positive and negative examples.
    Finds topk in a tensor of activations, matches them up against the text from the dataset,
    and plots them neuroscope-style.
    """
    _plot_topk(
        activations,
        dataloader=dataloader,
        model=model,
        layer=layer,
        k=k,
        largest=True,
        window_size=window_size,
        centred=centred,
        inclusions=inclusions,
        exclusions=exclusions,
        verbose=verbose,
        base_layer=base_layer,
    )
    _plot_topk(
        activations,
        dataloader=dataloader,
        model=model,
        layer=layer,
        k=k,
        largest=False,
        window_size=window_size,
        centred=centred,
        inclusions=inclusions,
        exclusions=exclusions,
        verbose=verbose,
        base_layer=base_layer,
    )


def _plot_top_p(
    all_activations: Float[Tensor, "row pos layer"],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    layer: int = 0,
    p: float = 0.1,
    k: int = 10,
    largest: bool = True,
    window_size: int = 10,
    centred: bool = True,
    inclusions: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
):
    """One-sided top-p plotting. Main entrypoint should be `plot_top_p`."""
    device = all_activations.device
    assert not (inclusions is not None and exclusions is not None)
    label = "positive" if largest else "negative"
    activations: Float[Tensor, "batch pos"] = all_activations[:, :, layer]
    if largest:
        ignore_value = torch.tensor(-np.inf, device=device, dtype=torch.float32)
    else:
        ignore_value = torch.tensor(np.inf, device=device, dtype=torch.float32)
    # create a mask for the inclusions/exclusions
    if exclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(
            exclusions, dataloader, model, all_activations
        )
        masked_activations = activations.where(~mask, other=ignore_value)
    elif inclusions is not None:
        mask: Bool[Tensor, "row pos"] = get_batch_pos_mask(
            inclusions, dataloader, model, all_activations
        )
        masked_activations = activations.where(mask, other=ignore_value)
    else:
        masked_activations = activations

    activations_flat: Float[Tensor, "(batch pos)"] = masked_activations.flatten()
    sample_size = int(p * len(activations_flat))
    top_p_indices = torch.topk(activations_flat, k=sample_size, largest=largest).indices
    sampled_indices = top_p_indices[torch.randperm(sample_size)[:k]]
    top_p_indices = np.array(
        np.unravel_index(sampled_indices.cpu().numpy(), masked_activations.shape)
    ).T.tolist()
    top_p_examples = [
        dataloader.dataset[b]["tokens"][s].item() for b, s in top_p_indices
    ]
    top_p_activations = [masked_activations[b, s].item() for b, s in top_p_indices]
    # Print the  most positive and negative examples and their activations
    print(f"Top {k} most {label} examples:")
    zeros = torch.zeros(
        (1, all_activations.shape[-1]), device=device, dtype=torch.float32
    )
    texts = [model.tokenizer.bos_token]
    text_to_not_repeat = set()
    acts = [zeros]
    text_sep = "\n"
    topk_zip = zip(top_p_indices, top_p_examples, top_p_activations)
    for index, example, activation in topk_zip:
        batch, pos = index
        text_window: List[str] = extract_text_window(
            batch, pos, dataloader=dataloader, model=model, window_size=window_size
        )
        activation_window: Float[Tensor, "pos layer"] = extract_activations_window(
            all_activations, batch, pos, window_size=window_size, dataloader=dataloader
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
    html = plot_neuroscope(
        texts, model=model, centred=centred, activations=acts_cat, verbose=False
    )
    save_html(
        html, f"top_{p * 100:.0f}pc_most_{label}_layer_{layer}_sentiment.html", model
    )
    display(html)


def plot_top_p(
    activations: Float[Tensor, "row pos layer"],
    dataloader: torch.utils.data.DataLoader,
    model: HookedTransformer,
    k: int = 10,
    p: float = 0.1,
    layer: int = 0,
    window_size: int = 10,
    centred: bool = True,
    inclusions: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
):
    """
    Main entrypoint for top-p plotting. Plots both positive and negative examples.
    Samples k times from the top p% of activations, matches them up against the text from the dataset,
    and plots them neuroscope-style.
    """
    _plot_top_p(
        activations,
        dataloader=dataloader,
        model=model,
        layer=layer,
        p=p,
        k=k,
        largest=True,
        window_size=window_size,
        centred=centred,
        inclusions=inclusions,
        exclusions=exclusions,
    )
    _plot_top_p(
        activations,
        dataloader=dataloader,
        model=model,
        layer=layer,
        p=p,
        k=k,
        largest=False,
        window_size=window_size,
        centred=centred,
        inclusions=inclusions,
        exclusions=exclusions,
    )
