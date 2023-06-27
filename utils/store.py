import numpy as np
import os
from transformer_lens import HookedTransformer
from typing import Union
import torch


def clean_label(label: str) -> str:
    label = label.replace('.npy', '')
    label = label.replace('data/', '')
    assert "/" not in label, "Label must not contain slashes"
    return label


def save_array(
        array: Union[np.ndarray, torch.Tensor], label: str, model: HookedTransformer
    ):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    label = clean_label(label)
    assert len(model.name) > 0, "Model must have a name"
    model_path = os.path.join('data', model.name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'wb') as f:
        np.save(f, array)
    return path


def load_array(label: str, model: HookedTransformer) -> np.ndarray:
    label = clean_label(label)
    assert len(model.name) > 0, "Model must have a name"
    model_path = os.path.join('data', model.name)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'rb') as f:
        array = np.load(f)
    return array