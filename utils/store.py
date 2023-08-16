import numpy as np
import pandas as pd
import glob
import os
from transformer_lens import HookedTransformer
from typing import Iterable, Union
import torch
import plotly.graph_objects as go
from circuitsvis.utils.render import RenderedHTML
from pandas.io.formats.style import Styler


def clean_label(label: str) -> str:
    label = label.replace('.npy', '')
    label = label.replace('.html', '')
    label = label.replace('data/', '')
    assert "/" not in label, "Label must not contain slashes"
    return label


def get_model_name(model: Union[HookedTransformer, str]) -> str:
    if isinstance(model, HookedTransformer):
        assert len(model.name) > 0, "Model must have a name"
        model = model.name
    model = model.replace('EleutherAI/', '')
    return model


def update_csv(
    data: pd.DataFrame,
    label: str, 
    model: Union[HookedTransformer, str], 
    key_cols: Iterable[str] = None,
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.csv')
    curr = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    curr = pd.concat([curr, data], axis=0)
    if key_cols is not None:
        curr = curr.drop_duplicates(subset=key_cols)
    curr.to_csv(path, index=False)
    return path


def get_csv(
    label: str,
    model: Union[HookedTransformer, str],
    key_cols: Iterable[str] = None,
) -> pd.DataFrame:
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    path = os.path.join(model_path, label + '.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if key_cols is not None:
        df = df.drop_duplicates(subset=key_cols)
    return df


def eval_csv(
    query: str,
    label: str,
    model: Union[HookedTransformer, str],
    key_cols: Iterable[str] = None,
):
    df = get_csv(label, model)
    if df.empty:
        return False
    if key_cols is not None:
        df = df.drop_duplicates(subset=key_cols)
    return df.eval(query).any()


def save_array(
    array: Union[np.ndarray, torch.Tensor], 
    label: str, 
    model: Union[HookedTransformer, str]
):
    model: str = get_model_name(model)
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'wb') as f:
        np.save(f, array)
    return path


def load_array(label: str, model: Union[HookedTransformer, str]) -> np.ndarray:
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'rb') as f:
        array = np.load(f)
    return array


def save_html(
    html_data: Union[go.Figure, RenderedHTML, Styler],
    label: str, 
    model: Union[HookedTransformer, str]
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.html')
    if isinstance(html_data, go.Figure):
        html_data.write_html(path)
    elif isinstance(html_data, RenderedHTML):
        with open(path, 'w') as f:
            f.write(str(html_data))
    elif isinstance(html_data, Styler):
        with open(path, 'w') as f:
            f.write(html_data.to_html())
    return path


def get_labels(glob_str: str, model: Union[HookedTransformer, str]) -> list:
    model: str = get_model_name(model)
    model_path = os.path.join('data', model)
    labels = [os.path.split(p)[-1] for p in glob.iglob(os.path.join(model_path, glob_str))]
    return labels


def is_file(name: str, model: Union[HookedTransformer, str]) -> list:
    model: str = get_model_name(model)
    model_path = os.path.join('data', model)
    file_path = os.path.join(model_path, name)
    return os.path.exists(file_path)


def save_text(
    text: str, 
    label: str, 
    model: Union[HookedTransformer, str]
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.txt')
    with open(path, 'w') as f:
        f.write(text)
    return path
