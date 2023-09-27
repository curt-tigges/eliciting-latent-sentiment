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
import pickle
from datasets import dataset_dict
import imgkit


DIRECTION_PATTERN = (
    r'^(kmeans|pca|das|das2d|das3d|logistic_regression|mean_diff)_'
    r'(simple_train|treebank_train)_'
    r'(ADJ|ALL)_'
    r'layer(\d*)'
    r'\.npy$'
)


def flatten_multiindex(in_df):
    df = in_df.copy()
    # Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    
    # Flatten index if it's a multiindex
    if isinstance(df.index, pd.MultiIndex):
        df.index = ['_'.join(map(str, idx)).strip() for idx in df.index.values]
    
    return df


def extract_layer_from_string(s: str) -> int:
    # Find numbers that directly follow the text "layer"
    match = re.search(r'(?<=layer)\d+', s)
    if match:
        number = match.group()
        return int(number)
    else:
        return None

def zero_pad_layer_string(s: str) -> str:
    # Find numbers that directly follow the text "layer"
    number = extract_layer_from_string(s)
    if number is not None:
        # Replace the original number with the zero-padded version
        s = s.replace(f'layer{number}', f'layer{number:02d}')
    return s


def add_styling(
    html: str,
    width: int = 300,
    height: int = 200,
    padding_x: int = 8,
    padding_y: int = 12,
    font_size: int = 24,
    border: int = 1,
    min_width: int = 50,
):
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
        width: {width}px; /* Specify the width you want */
        height: {height}px; /* Specify the height you want */
        overflow: auto;
        position: relative;
        font-size: {font_size}px;
    }}

    #{table_id} th, #{table_id} td {{
        padding: {padding_x}px {padding_y}px;
        border: {border}px solid #d4d4d4;
        text-align: center;
        min-width: {min_width}px;
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

    #{table_id} caption {{
        display: none;
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
    label = label.replace('.csv', '')
    label = label.replace('.txt', '')
    label = label.replace('.pkl', '')
    label = label.replace('.pdf', '')
    assert "/" not in label, "Label must not contain slashes"
    return label


def get_model_name(model: Union[HookedTransformer, str]) -> str:
    if isinstance(model, HookedTransformer):
        assert len(model.cfg.model_name) > 0, "Model must have a name"
        model = model.cfg.model_name
    model = model.replace('EleutherAI/', '')
    if model == 'gpt2':
        model = 'gpt2-small'
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
    index: bool = False,
):
    path = get_csv_path(label, model)
    curr = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    curr = pd.concat([curr, data], axis=0)
    if key_cols is not None:
        curr = curr.drop_duplicates(subset=key_cols)
    curr.to_csv(path, index=index)
    return path


def get_csv(
    label: str,
    model: Union[HookedTransformer, str],
    key_cols: Iterable[str] = None,
    index_col: int = None,
    header: Iterable[int] = 0,
) -> pd.DataFrame:
    path = get_csv_path(label, model)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=index_col, header=header)
    if key_cols is not None:
        df = df.drop_duplicates(subset=key_cols)
    return df


def to_csv(
    data: Union[pd.DataFrame, pd.Series],
    label: str,
    model: Union[HookedTransformer, str],
    index: bool = False,
):
    path = get_csv_path(label, model)
    data.to_csv(path, index=index)
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
    model: Union[HookedTransformer, str],
    local: bool = True,
    static: bool = False,
    **styling_kwargs,
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
        html_str = html_data.local_src if local else html_data.cdn_src
        with open(path, 'w') as f:
            f.write(html_str)
    elif isinstance(html_data, Styler):
        html = html_data.to_html()
        html = add_styling(html, **styling_kwargs)
        with open(path, 'w') as f:
            f.write(html)
    else:
        raise ValueError(f"Invalid type: {type(html_data)}")
    if static:
        static_path = os.path.join(model_path, label + '.png')
        imgkit.from_file(path, static_path)
    return path


def get_labels(glob_str: str, model: Union[HookedTransformer, str]) -> list:
    model: str = get_model_name(model)
    model_path = os.path.join('data', model)
    labels = [os.path.split(p)[-1] for p in glob.iglob(os.path.join(model_path, glob_str))]
    return labels


def is_file(name: str, model: Union[HookedTransformer, str]) -> bool:
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


def save_pickle(
    obj: object,
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def load_pickle(
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    path = os.path.join(model_path, label + '.pkl')
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_dataset_dict(
    dataset_dict: dataset_dict,
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.pkl')
    dataset_dict.save_to_disk(path)
    return path


def save_image(
    figure: go.Figure,
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.png')
    figure.write_image(path)
    return path


def save_pdf(
    figure: go.Figure,
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.pdf')
    figure.write_image(path, format='pdf')
    figure.write_image(path, format='pdf')
    return path
