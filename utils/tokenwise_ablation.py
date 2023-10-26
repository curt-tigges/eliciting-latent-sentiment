import einops
from functools import partial
import torch
import numpy as np
import datasets
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Features, Sequence, Value, Dataset
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text
from utils.circuit_analysis import get_logit_diff
from utils.ablation import ablate_resid_with_precalc_mean, freeze_attn_pattern_hook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- GENERAL UTILS --------------------
def find_positions(
        batch_tensor: Float[Tensor, "batch seq_len"], 
        token_ids: List =[13]
    ) -> List[List[int]]:
    """Finds positions of specified token ids in a tensor of shape (batch, sequence_position)
    
    Args:
        batch_tensor: Tensor of shape (batch, sequence_position)
        token_ids: List of token ids to find positions of

    Returns:
        positions: List of lists of positions of each token id in token_ids in each batch item
    """
    positions = []
    for batch_item in batch_tensor:
        token_positions = {token_id: [] for token_id in token_ids}
        for position, token in enumerate(batch_item):
            if token.item() in token_ids:
                token_positions[token.item()].append(position)
        positions.append([token_positions[token_id] for token_id in token_ids])
    return positions


def convert_to_tensors(
        dataset: Dataset, 
        column_name: str = 'tokens'
    ) -> Dataset:
    """Converts a dataset to a dataset of tensors

    Args:
        dataset (Dataset): Dataset to convert
        column_name (str, optional): Name of column to convert. Defaults to 'tokens'.

    Returns:
        Dataset: Dataset of tensors
    """
    final_batches = []
    
    for batch in dataset:
        trimmed_batch = batch[column_name] #[batch[column_name][0]] + [token for token in batch[column_name] if token != 0]
        final_batches.append(trimmed_batch)
    
    # Convert list of batches to tensors
    final_batches = [torch.tensor(batch, dtype=torch.long) for batch in final_batches]
    # Create a new dataset with specified features
    features = Features({"tokens": Sequence(Value("int64"))})
    final_dataset = Dataset.from_dict({"tokens": final_batches}, features=features)

    final_dataset.set_format(type="torch", columns=["tokens"])
    
    return final_dataset


def names_filter(
        name: str,
        permitted_names: List = ['resid_post', 'resid_pre']
    ) -> bool:
    """Filter for the names of the activations we want to keep to study the resid stream.

    Args:
        name (str): Name of the activation to check
        permitted_names (List, optional): List of names (or suffixes) to keep. Defaults to ['resid_post', 'resid_pre'].

    Returns:
        bool: True if name is in permitted_names, False otherwise
    """
    # check all names with endswith
    for permitted_name in permitted_names:
        if name.endswith(permitted_name):
            return True

    return False


# -------------------- DIRECTION LOADING UTILS --------------------

# TODO: standardize the way we load directions
def load_directions(
        model: HookedTransformer, 
        direction_folder: str = "directions",
        direction_prefix: str = "das_simple_train_ADJ",
    ) -> Float[Tensor, "layer d_model"]:
    """Loads a list of direction vectors of shape (n_layers, d_model)
    """
    directions = []
    for i in range(model.cfg.n_layers):
        dir = np.load(f"{direction_folder}/{direction_prefix}{i}.npy")
        if len(dir.shape) == 2:
            dir = dir[:, 0]
        directions.append(torch.tensor(dir))

    # convert to tensor
    directions = torch.stack(directions).to(device)

    return directions


def get_random_directions(
        model: HookedTransformer, 
    ) -> Float[Tensor, "layer d_model"]:
    """Returns a list of random direction vectors of shape (n_layers, d_model)"""
    directions = []
    num_layers = model.cfg.n_layers
    for _ in range(num_layers):
        dir = torch.randn(model.cfg.d_model).to(device)
        directions.append(dir)

    # convert to tensor
    directions = torch.stack(directions).to(device)

    return directions


def get_zeroed_dir_vector(
        model: HookedTransformer,
    ) -> Float[Tensor, "layer d_model"]:
    """Returns a list of zeroed direction vectors of shape (n_layers, d_model)"""
    zeroed_directions = []
    for _ in range(model.cfg.n_layers):
        dir = torch.zeros(model.cfg.d_model).to(device)
        zeroed_directions.append(dir)

    # convert to tensor
    zeroed_directions = torch.stack(zeroed_directions).to(device)

    return zeroed_directions


# -------------------- METRICS -------------------- 
def compute_last_position_logit_diff(
        logits: Float[Tensor, "batch sequence_position logits"],
        mask: Float[Tensor, "batch sequence_position"],
        answer: Float[Tensor, "batch 2"]
    ) -> Float[Tensor, "batch"]:
    """
    Args:
        logits: Tensor of shape (batch, sequence_position, logits)
        mask: Tensor of shape (batch, sequence_position)
        answer: Tensor of shape (batch, 2)

    Returns:
        logit_diff: Tensor of shape (batch,)
    """
    # Find the last unmasked sequence position for each item in the batch
    last_unmasked_positions = mask.sum(dim=1) - 1  # Subtract 1 to get zero-based index
    #print(last_unmasked_positions)

    # Extract the logits for the last unmasked positions
    last_logits = logits[torch.arange(logits.size(0)), last_unmasked_positions]
    #print(f"last logits: {last_logits.shape}")
    #print(f"last logits shape: {last_logits.shape}")

    # Extract the logits for the correct and incorrect answers
    correct_logits = last_logits[torch.arange(last_logits.size(0)), answer[:, 0]]
    #print(f"correct logits shape: {correct_logits.shape}")
    incorrect_logits = last_logits[torch.arange(last_logits.size(0)), answer[:, 1]]

    # Compute the logit differences
    logit_diff = correct_logits - incorrect_logits
    #print(f"logit diff shape: {logit_diff.shape}")

    return logit_diff

# TODO: Add metrics for loss


# -------------------- ACTIVATION UTILS --------------------
def get_layerwise_token_mean_activations(
        model: HookedTransformer, 
        data_loader: DataLoader, 
        token_id: int = 13
    ) -> Float[Tensor, "layer d_model"]:
    """Get the mean value of a particular token id across a dataset for each layer of a model
    
    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        token_id: Token id to get the mean value of

    Returns:
        token_mean_values: Tensor of shape (num_layers, d_model) containing the mean value of token_id for each layer
    """
    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    activation_sums = torch.stack([torch.zeros(d_model) for _ in range(num_layers)]).to(device)
    token_counts = [0] * num_layers

    print(activation_sums.shape)

    token_mean_values = torch.zeros((num_layers, d_model))
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        
        batch_tokens = batch_value['tokens'].to(device)

        # get positions of all specified token ids in batch
        punct_pos = find_positions(batch_tokens, token_ids=[token_id])

        _, cache = model.run_with_cache(
            batch_tokens, 
            names_filter=names_filter
        )

        
        for i in range(batch_tokens.shape[0]):
            for p in punct_pos[i][0]:
                for layer in range(num_layers):
                    activation_sums[layer] += cache[f"blocks.{layer}.hook_resid_post"][i, p, :]
                    token_counts[layer] += 1

    for layer in range(num_layers):
        token_mean_values[layer] = activation_sums[layer] / token_counts[layer]

    return token_mean_values


# -------------------- ABLATION UTILS --------------------
def zero_attention_pos_hook(
    pattern: Float[Tensor, "batch head seq_Q seq_K"], 
    hook: HookPoint,
    pos_by_batch: List[List[int]], 
    layer: int = 0, 
    head_idx: int = 0,
) -> Float[Tensor, "batch head seq_Q seq_K"]:
    """Zero-ablates an attention pattern tensor at a particular position"""
    assert 'pattern' in hook.name

    batch_size = pattern.shape[0]
    assert len(pos_by_batch) == batch_size

    for i in range(batch_size):
        for p in pos_by_batch[i]:
            pattern[i, head_idx, p, p] = 0
            
    return pattern


def ablate_resid_with_precalc_mean(
    component: Float[Tensor, "batch ..."],
    hook: HookPoint,
    cached_means: Float[Tensor, "layer ..."],
    pos_by_batch: Float[Tensor, "batch ..."],
    layer: int = 0,
) -> Float[Tensor, "batch ..."]:
    """
    Mean-ablates a batch tensor

    Args:
        component: the tensor to compute the mean over the batch dim of
        hook: the hook point

    Returns:
        the mean over the cache component of the tensor
    """
    assert 'resid' in hook.name

    # Identify the positions where pos_by_batch is 1
    batch_indices, sequence_positions = torch.where(pos_by_batch == 1)

    # Replace the corresponding positions in component with cached_means[layer]
    component[batch_indices, sequence_positions] = cached_means[layer]

    return component


def ablate_resid_with_direction(
    component: Float[Tensor, "batch ..."],
    hook: HookPoint,
    direction_vector: Float[Tensor, "layer d_model"],
    labels: torch.Tensor,
    multiplier: float = 1.0,
    pos_by_batch: torch.Tensor = None,
    layer: int = 0,
) -> torch.Tensor:
    """
    Ablates a batch tensor by removing the influence of a direction vector from it.

    Args:
        component: the tensor to compute the mean over the batch dim of
        direction_vector: the direction vector to remove from the component
        multiplier: the multiplier to apply to the direction vector
        pos_by_batch: the positions to ablate
        layer: the layer to ablate

    Returns:
        the ablated component
    """
    assert 'resid' in hook.name

    # Normalize the direction vector to make sure it's a unit vector
    D_normalized = direction_vector[layer] / torch.norm(direction_vector[layer])

    # Calculate the projection of component onto direction_vector
    proj = einops.einsum(component, D_normalized, "b s d, d -> b s").unsqueeze(-1) * D_normalized
    

    # Ablate the direction from component
    component_ablated = component.clone()  # Create a copy to ensure original is not modified
    if pos_by_batch is not None:
        batch_indices, sequence_positions = torch.where(pos_by_batch == 1)
        component_ablated[batch_indices, sequence_positions] = component[batch_indices, sequence_positions] - multiplier * proj[batch_indices, sequence_positions]
        
        # Print the (batch, pos) coordinates of all d_model vectors that were ablated
        # for b, s in zip(batch_indices, sequence_positions):
        #     print(f"(batch, pos) = ({b.item()}, {s.item()})")

        # Check that positions not in (batch_indices, sequence_positions) were not ablated
        check_mask = torch.ones_like(component, dtype=torch.bool)
        check_mask[batch_indices, sequence_positions] = 0
        if not torch.all(component[check_mask] == component_ablated[check_mask]):
            raise ValueError("Positions outside of specified (batch_indices, sequence_positions) were ablated!")

    return component_ablated


# -------------------- EXPERIMENTS --------------------
def compute_mean_ablation_modified_logit_diff(
        model: HookedTransformer, 
        data_loader: DataLoader, 
        layers_to_ablate: List[int],  
        heads_to_freeze: List[Tuple[int, int]],
        cached_means: Float[Tensor, "layer d_model"], 
        frozen_attn_variant: bool = False,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """Computes the change in logit difference (between two answers) when the activations of a particular token are mean-ablated.

    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        heads_to_freeze: List of heads to freeze
        cached_means: List of tensors of shape (layer, d_model) containing the mean value of a given token for each layer

    Returns:
        orig_ld_list: List of tensors of shape (batch,) containing the logit difference for each item in the batch before ablation
        ablated_ld_list: List of tensors of shape (batch,) containing the logit difference for each item in the batch after ablation
        freeze_ablated_ld_list: List of tensors of shape (batch,) containing the logit difference for each item in the batch after ablation with attention frozen
    """
    orig_ld_list = []
    ablated_ld_list = []
    freeze_ablated_ld_list = []
    
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value['tokens'].to(device)
        punct_pos = batch_value['positions'].to(device)

        # get the logit diff for the last token in each sequence
        orig_logits, clean_cache = model.run_with_cache(batch_tokens, return_type="logits", prepend_bos=False)
        orig_ld = compute_last_position_logit_diff(orig_logits, batch_value['attention_mask'], batch_value['answers'])
        orig_ld_list.append(orig_ld)
        
        # repeat with commas ablated
        for layer in layers_to_ablate:
            mean_ablate_comma = partial(ablate_resid_with_precalc_mean, cached_means=cached_means, pos_by_batch=punct_pos, layer=layer)
            model.blocks[layer].hook_resid_post.add_hook(mean_ablate_comma)
       
        ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
        ablated_ld = compute_last_position_logit_diff(ablated_logits, batch_value['attention_mask'], batch_value['answers'])
        ablated_ld_list.append(ablated_ld)
        
        model.reset_hooks()

        if frozen_attn_variant:
        # repeat with attention frozen and commas ablated
            for layer, head in heads_to_freeze:
                freeze_attn = partial(freeze_attn_pattern_hook, cache=clean_cache, layer=layer, head_idx=head)
                model.blocks[layer].attn.hook_pattern.add_hook(freeze_attn)

            for layer in layers_to_ablate:
                mean_ablate_comma = partial(ablate_resid_with_precalc_mean, cached_means=cached_means, pos_by_batch=punct_pos, layer=layer)
                model.blocks[layer].hook_resid_post.add_hook(mean_ablate_comma)
        
            freeze_ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
            freeze_ablated_ld = compute_last_position_logit_diff(freeze_ablated_logits, batch_value['attention_mask'], batch_value['answers'])
            freeze_ablated_ld_list.append(freeze_ablated_ld)
            
            model.reset_hooks()

    return torch.cat(orig_ld_list), torch.cat(ablated_ld_list), torch.cat(freeze_ablated_ld_list)


def compute_directional_ablation_modified_logit_diff(
        model: HookedTransformer, 
        data_loader: DataLoader, 
        layers_to_ablate: List[int], 
        heads_to_freeze: List[Tuple[int, int]],
        direction_vectors: Float[Tensor, "layer d_model"],
        multiplier=1.0,
        frozen_attn_variant: bool = False,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """Computes the change in logit difference (between two answers) when the activations of a particular token are direction-ablated.
    
    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        heads_to_freeze: List of heads to freeze
        direction_vectors: List of tensors of shape (layer, d_model) containing the direction vector for each layer
        multiplier: Multiplier to apply to the direction vector

    Returns:
        orig_ld_list: List of tensors of shape (batch,) containing the logit difference for each item in the batch before ablation
        ablated_ld_list: List of tensors of shape (batch,) containing the logit difference for each item in the batch after ablation
        freeze_ablated_ld_list: List of tensors of shape (batch,) containing the logit difference for each item in the batch after ablation with attention frozen

    """
    
    orig_ld_list = []
    ablated_ld_list = []
    freeze_ablated_ld_list = []
    
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value['tokens'].to(device)
        labels = batch_value['label'].to(device)
        punct_pos = batch_value['positions'].to(device)

        # get the logit diff for the last token in each sequence
        orig_logits, clean_cache = model.run_with_cache(batch_tokens, return_type="logits", prepend_bos=False)
        orig_ld = compute_last_position_logit_diff(orig_logits, batch_value['attention_mask'], batch_value['answers'])
        orig_ld_list.append(orig_ld)
        
        # repeat with token ablated
        for layer in layers_to_ablate:
            dir_ablate_comma = partial(ablate_resid_with_direction, labels=labels, direction_vector=direction_vectors, multiplier=multiplier, pos_by_batch=punct_pos, layer=layer)
            model.blocks[layer].hook_resid_post.add_hook(dir_ablate_comma)
       
        ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
        # check to see if ablated_logits has any nan values
        if torch.isnan(ablated_logits).any():
            print("ablated logits has nan values")
        ablated_ld = compute_last_position_logit_diff(ablated_logits, batch_value['attention_mask'], batch_value['answers'])
        ablated_ld_list.append(ablated_ld)
        
        model.reset_hooks()

        if frozen_attn_variant:
            # repeat with attention frozen and token ablated
            for layer, head in heads_to_freeze:
                freeze_attn = partial(freeze_attn_pattern_hook, cache=clean_cache, layer=layer, head_idx=head)
                model.blocks[layer].attn.hook_pattern.add_hook(freeze_attn)

            for layer in layers_to_ablate:
                dir_ablate_comma = partial(ablate_resid_with_direction, labels=labels, direction_vector=direction_vectors, multiplier=multiplier, pos_by_batch=punct_pos, layer=layer)
                model.blocks[layer].hook_resid_post.add_hook(dir_ablate_comma)
        
            freeze_ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
            freeze_ablated_ld = compute_last_position_logit_diff(freeze_ablated_logits, batch_value['attention_mask'], batch_value['answers'])
            freeze_ablated_ld_list.append(freeze_ablated_ld)
            
            model.reset_hooks()

    return torch.cat(orig_ld_list), torch.cat(ablated_ld_list), torch.cat(freeze_ablated_ld_list)


def compute_directional_ablation_modified_logit_diff_all_pos(
        model: HookedTransformer, 
        data_loader: DataLoader, 
        layers_to_ablate: List[int],
        heads_to_freeze: List[Tuple[int, int]],
        direction_vectors: Float[Tensor, "layer d_model"],
        multiplier: float = 1.0,
        frozen_attn_variant: bool = False,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    
    orig_ld_list = []
    ablated_ld_list = []
    freeze_ablated_ld_list = []
    
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value['tokens'].to(device)
        labels = batch_value['label'].to(device)
        punct_pos = batch_value['attention_mask'].to(device)

        # get the logit diff for the last token in each sequence
        orig_logits, clean_cache = model.run_with_cache(batch_tokens, return_type="logits", prepend_bos=False)
        orig_ld = compute_last_position_logit_diff(orig_logits, batch_value['attention_mask'], batch_value['answers'])
        orig_ld_list.append(orig_ld)
        
        # repeat with commas ablated
        for layer in layers_to_ablate:
            dir_ablate_comma = partial(ablate_resid_with_direction, labels=labels, direction_vector=direction_vectors, multiplier=multiplier, pos_by_batch=punct_pos, layer=layer)
            model.blocks[layer].hook_resid_post.add_hook(dir_ablate_comma)
       
        ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
        
        # check to see if ablated_logits has any nan values
        if torch.isnan(ablated_logits).any():
            print("ablated logits has nan values")
        ablated_ld = compute_last_position_logit_diff(ablated_logits, batch_value['attention_mask'], batch_value['answers'])
        ablated_ld_list.append(ablated_ld)
        
        model.reset_hooks()

        if frozen_attn_variant:
            # repeat with attention frozen and commas ablated
            for layer, head in heads_to_freeze:
                freeze_attn = partial(freeze_attn_pattern_hook, cache=clean_cache, layer=layer, head_idx=head)
                model.blocks[layer].attn.hook_pattern.add_hook(freeze_attn)

            for layer in layers_to_ablate:
                dir_ablate_comma = partial(ablate_resid_with_direction, labels=labels, direction_vector=direction_vectors, multiplier=multiplier, pos_by_batch=punct_pos, layer=layer)
                model.blocks[layer].hook_resid_post.add_hook(dir_ablate_comma)
        
            freeze_ablated_logits = model(batch_tokens, return_type="logits", prepend_bos=False)
            freeze_ablated_ld = compute_last_position_logit_diff(freeze_ablated_logits, batch_value['attention_mask'], batch_value['answers'])
            freeze_ablated_ld_list.append(freeze_ablated_ld)
            
            model.reset_hooks()

    return torch.cat(orig_ld_list), torch.cat(ablated_ld_list), torch.cat(freeze_ablated_ld_list)


def compute_zeroed_attn_modified_loss(model: HookedTransformer, data_loader: DataLoader, heads_to_ablate) -> float:
    total_loss = 0
    loss_list = []
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value['tokens'].to(device)

        # get positions of all 11 and 13 token ids in batch
        punct_pos = find_positions(batch_tokens, token_ids=[13])

        # get the loss for each token in the batch
        initial_loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True)
        
        # add hooks for the activations of the 11 and 13 tokens
        for layer, head in heads_to_ablate:
            ablate_punct = partial(zero_attention_pos_hook, pos_by_batch=punct_pos, layer=layer, head_idx=head)
            model.blocks[layer].attn.hook_pattern.add_hook(ablate_punct)

        # get the loss for each token when run with hooks
        hooked_loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True)

        # compute the percent difference between the two losses
        loss_diff = (hooked_loss - initial_loss) / initial_loss

        loss_list.append(loss_diff)

    model.reset_hooks()
    return loss_list, batch_tokens


def compute_mean_ablation_modified_loss(
        model: HookedTransformer, 
        data_loader: DataLoader, 
        layers_to_ablate: List[int], 
        cached_means: Float[Tensor, "layer d_model"],
        debug: bool = False,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """Computes the change in loss when the activations of a particular token are mean-ablated.
    
    Args:
        model: HookedTransformer model
        data_loader: DataLoader for the dataset
        layers_to_ablate: List of layers to ablate
        cached_means: List of tensors of shape (layer, d_model) containing the mean value of a given token for each layer

    Returns:
        loss_diff_list: List of tensors of shape (batch,) containing the loss difference for each item in the batch
        orig_loss_list: List of tensors of shape (batch,) containing the original loss for each item in the batch
    """
    loss_diff_list = []
    orig_loss_list = []
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value['tokens'].to(device)
        punct_pos = batch_value['positions'].to(device)

        # get the loss for each token in the batch
        initial_loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True)
        # concatenate column of 0s
        initial_loss = torch.cat([torch.zeros((initial_loss.shape[0], 1)).to(device), initial_loss], dim=1)

        if debug:
            print(f"initial loss shape: {initial_loss.shape}")
            print(initial_loss[0])
        orig_loss_list.append(initial_loss)
        
        # add hooks for the activations of the relevant tokens
        for layer in layers_to_ablate:
            mean_ablate_token = partial(ablate_resid_with_precalc_mean, cached_means=cached_means, pos_by_batch=punct_pos, layer=layer)
            model.blocks[layer].hook_resid_post.add_hook(mean_ablate_token)

        # get the loss for each token when run with hooks
        hooked_loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=True)
        # concatenate column of 0s
        hooked_loss = torch.cat([torch.zeros((hooked_loss.shape[0], 1)).to(device), hooked_loss], dim=1)

        if debug:
            print(f"hooked loss shape: {hooked_loss.shape}")
            print(hooked_loss[0])

        # compute the difference between the two losses
        loss_diff = hooked_loss - initial_loss

        # set all positions right after punct_pos to zero
        for p in punct_pos:
            if p+1 < loss_diff.shape[1]:
                if debug:
                    print(f"zeroing {p}")
                loss_diff[0, p+1] = 0

        # set all masked positions to zero
        loss_diff[batch_value['attention_mask'] == 0] = 0

        loss_diff_list.append(loss_diff)

        model.reset_hooks()
    
    
    return torch.cat(loss_diff_list), torch.cat(orig_loss_list)
