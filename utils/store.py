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
import re


def add_styling(html):
    # Extract the table ID from the HTML using regex
    table_id_match = re.search(r'<table id="([^"]+)">', html)
    if not table_id_match:
        return "Invalid HTML: Table ID not found"
    
    table_id = table_id_match.group(1)

    # Define the general styles using the extracted table ID
    styles = f"""
    /* General styles */
    #{table_id} {{
        border-collapse: collapse;
        width: 300px; /* Specify the width you want */
        height: 200px; /* Specify the height you want */
        overflow: auto;
        position: relative;
    }}

    #{table_id} th, #{table_id} td {{
        padding: 8px 12px;
        border: 1px solid #d4d4d4;
        text-align: center;
        min-width: 50px;
        box-sizing: border-box;
        position: relative;
    }}

    /* Freeze first column */   
    #{table_id} .level0 {{
        background-color: #ddd;
        position: -webkit-sticky;
        position: sticky;
        left: 0;
        z-index: 1;
    }}

    /* Freeze first row */
    #{table_id} thead {{
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 2;
    }}

    #{table_id} thead {{
        background-color: #ddd;
    }}
    """
    
    # Insert the general styles into the existing style section of the HTML
    style_start_index = html.find("<style type=\"text/css\">")
    if style_start_index == -1:
        return "Invalid HTML: Style section not found"
    
    style_start_index += len("<style type=\"text/css\">")
    html_with_styles = html[:style_start_index] + styles + html[style_start_index:]
    
    return html_with_styles


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


def get_csv_path(
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.csv')
    return path


def update_csv(
    data: pd.DataFrame,
    label: str, 
    model: Union[HookedTransformer, str], 
    key_cols: Iterable[str] = None,
):
    path = get_csv_path(label, model)
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
    path = get_csv_path(label, model)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if key_cols is not None:
        df = df.drop_duplicates(subset=key_cols)
    return df


def to_csv(
    data: Union[pd.DataFrame, pd.Series],
    label: str,
    model: Union[HookedTransformer, str],
):
    path = get_csv_path(label, model)
    data.to_csv(path, index=False)
    return path


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
    model: Union[HookedTransformer, str],
    root: str = 'data',
):
    if not os.path.exists(root):
        os.mkdir(root)
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
        html = html_data.to_html()
        html = add_styling(html)
        with open(path, 'w') as f:
            f.write(html)
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
