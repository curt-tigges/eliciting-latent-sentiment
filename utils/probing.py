from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import copy

import os
import torch
import numpy as np
import pickle

from transformer_lens import HookedTransformer, ActivationCache


def get_layer_act(model, prompts_tokens, layer):

    layer_act_list = []

    for i in range(0, prompts_tokens.shape[0], 16):
        logits, cache = model.run_with_cache(prompts_tokens[i:i+16])
        layer_resid = cache[f'blocks.{layer}.hook_resid_post']
        layer_act_list.append(layer_resid.squeeze().cpu().numpy())

    layer_act = np.concatenate(layer_act_list, axis=0)
    return layer_act


def cache_and_save_component_activations(model, data, component, pos):
    batch_list = []
    
    for i in range(0, data.shape[0], 16):
        logits, cache = model.run_with_cache(data[i:i+16])
        layer_list = []
        for l in range(model.cfg.n_layers):
            layer_act = cache[f'blocks.{l}.{component}'][:, pos, :]
            layer_list.append(layer_act.unsqueeze(1).cpu().numpy())
        layer_list = np.concatenate(layer_list, axis=1)
        batch_list.append(layer_list)
    batch_list = np.concatenate(batch_list, axis=0)
    # batch, layer, pos, dim
    print(f"Activation shape: {batch_list.shape}")
    # save as pkl file
    with open(f"data/cached_activations/2_8b_mood_inference/{component}_pos_{pos}_activations.pkl", "wb") as f:
        pickle.dump(batch_list, f)

    return batch_list


def train_probe_at_layer_pos(model, act_folder, labels, layer, pos=0, component='hook_resid_post', with_scaler=True, max_iter=100):

    # Get activations
    with open(os.path.join(act_folder, f"{component}_pos_{pos}_activations.pkl"), "rb") as f:
        act_list = pickle.load(f)[:, layer, :]

    X_train, X_test, y_train, y_test = train_test_split(act_list, labels, test_size=0.2, random_state=42)
    #print(f"Data size: {X_train.shape} Label size: {y_train.shape}")

    if with_scaler:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    else:
        pipe = make_pipeline(LogisticRegression(max_iter=max_iter))
    pipe.fit(X_train, y_train)  # apply scaling on training data
    score = pipe.score(X_test, y_test)  # apply scaling on testing data, without leaking training data.
    print(f"Layer {layer} position {pos} test score: {score}")

    return pipe, score


def get_probe_direction(pipeline):

    # Get the standard deviation and mean from the StandardScaler
    scaler = pipeline.named_steps['standardscaler']
    std = scaler.scale_
    mean = scaler.mean_

    # Get the coefficients and intercept from the LogisticRegression
    logistic_regression_model = pipeline.named_steps['logisticregression']
    scaled_coef = logistic_regression_model.coef_
    scaled_intercept = logistic_regression_model.intercept_

    # Unscale the coefficients
    probe_coef = scaled_coef / std

    # Unscale the intercept
    probe_intercept = scaled_intercept - (scaled_coef * mean / std).sum(axis=1)

    return probe_coef, probe_intercept


def cache_and_save_probe_coefficients(
    model,
    act_folder,
    save_folder,
    labels,
    component,
    seq_len,
    with_scaler=True,
    max_iter=100,
):
    """
    Cache and save probe coefficients for all layers and positions.
    """

    probe_coefficients = torch.zeros((model.cfg.n_layers, seq_len, model.cfg.d_model))
    probe_intercepts = torch.zeros((model.cfg.n_layers, seq_len))
    probe_scores = torch.zeros((model.cfg.n_layers, seq_len))

    for layer in range(model.cfg.n_layers):
        for pos in range(seq_len):
            probe, score = train_probe_at_layer_pos(
                model=model, 
                act_folder=act_folder, 
                labels=labels, 
                layer=layer, 
                pos=pos, 
                component=component,
                with_scaler=with_scaler,
                max_iter=max_iter,
            )

            probe_coef, probe_intercept = get_probe_direction(probe)
            
            probe_coefficients[layer, pos] = torch.tensor(probe_coef)
            probe_intercepts[layer, pos] = torch.tensor(probe_intercept)
            probe_scores[layer, pos] = torch.tensor(score)

    # save as pickle file
    probe_coefficients_path = os.path.join(save_folder, f"probe_coefficients_{component}.pkl")
    probe_intercepts_path = os.path.join(save_folder, f"probe_intercepts_{component}.pkl")
    probe_scores_path = os.path.join(save_folder, f"probe_scores_{component}.pkl")
    
    with open(probe_coefficients_path, "wb") as f:
        pickle.dump(probe_coefficients, f)
    with open(probe_intercepts_path, "wb") as f:
        pickle.dump(probe_intercepts, f)
    with open(probe_scores_path, "wb") as f:
        pickle.dump(probe_scores, f)

    return probe_coefficients, probe_intercepts, probe_scores


def apply_probe_to_cache(
    model: HookedTransformer,
    cache: ActivationCache,
    component: str,
    probe_coefs: torch.Tensor,
    alpha: float = 1.0,
    device: str = "cuda",
    verbose: bool = False,
) -> ActivationCache:
    """
    Train a probe on a given part of the model, apply the direction of the probe to the cache, and return the new cache.
    """
    new_cache = copy.deepcopy(cache)
    n_layers = model.cfg.n_layers
    seq_len = cache["blocks.0.hook_mlp_out"].shape[1]
    d_model = model.cfg.d_model

    assert probe_coefs.shape == (n_layers, seq_len, d_model)

    for layer in range(n_layers):
        for pos in range(seq_len):

            if verbose:
                print(f"Applying probe for layer {layer}, position {pos}...")

            probe_coef = probe_coefs[layer, pos, :]
            probe_coef = probe_coef.to(device)

            # Scale the steering vector by the norm of the activations
            steering_effect = alpha * (probe_coef / torch.norm(probe_coef)) * torch.norm(new_cache[f"blocks.{layer}.{component}"][:, pos, :])
            

            # Add the steering effect to the activations
            new_cache[f"blocks.{layer}.{component}"][:, pos, :] += steering_effect

            # remove steering_effect from device
            steering_effect = steering_effect.cpu()
            probe_coef = probe_coef.cpu()

            # Check that the activations have changed

            assert not torch.allclose(cache[f"blocks.{layer}.{component}"][:, pos, :], new_cache[f"blocks.{layer}.{component}"][:, pos, :])


    return new_cache


# get cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))