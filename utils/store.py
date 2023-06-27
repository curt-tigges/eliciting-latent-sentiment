import numpy as np
import os
from transformer_lens import HookedTransformer
from typing import Union
import torch


def store_array(array: Union[np.ndarray, torch.Tensor], label: str, model: HookedTransformer):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    label = label.replace('.npy', '')
    label = label.replace('data/', '')
    assert len(model.name) > 0, "Model must have a name"
    assert "/" not in label, "Label must not contain slashes"
    model_path = os.path.join('data', model.name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'wb') as f:
        np.save(f, array)
    return path