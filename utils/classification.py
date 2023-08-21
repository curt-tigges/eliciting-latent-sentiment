from enum import Enum
from functools import partial
from typing import Iterable, List, Tuple
from jaxtyping import Int, Float
from torch import Tensor
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import warnings
from utils.residual_stream import ResidualStreamDataset
from utils.store import save_array, update_csv


class ClassificationMethod(Enum):
    KMEANS = "kmeans"
    LOGISTIC_REGRESSION = "logistic_regression"
    PCA = "pca"
    SVD = "svd"
    MEAN_DIFF = "mean_diff"


def safe_cosine_sim(
    line1: Float[np.ndarray, "d_model"], line2: Float[np.ndarray, "d_model"],
    tol: float = 1e-6, min_value: float = -0.9,
):
    if np.linalg.norm(line1) < tol or np.linalg.norm(line2) < tol:
        cosine_sim = 0
    else:
        cosine_sim = np.dot(line1, line2) / (
            np.linalg.norm(line1) * np.linalg.norm(line2)
        )
    assert cosine_sim >= min_value, (
        f"cosine sim very negative ({cosine_sim:.2f}), looks like a flipped sign"
    )
    return cosine_sim


def split_by_label(
    adjectives: List[str], labels: Int[np.ndarray, "batch"]
) -> Tuple[List[str], List[str]]:
    first_cluster = [
        adj[1:]
        for i, adj in enumerate(adjectives) 
        if labels[i] == 0
    ]
    second_cluster = [
        adj[1:]
        for i, adj in enumerate(adjectives) 
        if labels[i] == 1
    ]
    return first_cluster, second_cluster


def get_accuracy(
    predicted_positive: Iterable[str],
    predicted_negative: Iterable[str],
    actual_positive: Iterable[str],
    actual_negative: Iterable[str],
) -> Tuple[str, int, int, float]:
    correct = (
        len(set(predicted_positive) & set(actual_positive)) +
        len(set(predicted_negative) & set(actual_negative))
    )
    total = len(actual_positive) + len(actual_negative)
    accuracy = correct / total
    return correct, total, accuracy


def _fit(
    train_data: ResidualStreamDataset, train_pos: str, train_layer: int,
    test_data: ResidualStreamDataset, test_pos: str, test_layer: int,
    n_init: int = 10,
    n_clusters: int = 2,
    random_state: int = 0,
    n_components: int = None,
    method: ClassificationMethod = ClassificationMethod.KMEANS,
):
    train_embeddings: Float[Tensor, "batch d_model"] = train_data.embed(train_pos, train_layer)
    test_embeddings: Float[Tensor, "batch d_model"] = test_data.embed(test_pos, test_layer)
    train_positive_str_labels, train_negative_str_labels = train_data.get_positive_negative_labels()
    test_positive_str_labels, test_negative_str_labels = test_data.get_positive_negative_labels()
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    if method == ClassificationMethod.KMEANS or method == ClassificationMethod.MEAN_DIFF:
        kmeans.fit(train_embeddings)
        test_km_labels = kmeans.predict(test_embeddings)
    elif method == ClassificationMethod.PCA:
        pca = PCA(n_components=n_components)
        train_pcs = pca.fit_transform(train_embeddings.numpy())
        test_pcs = pca.transform(test_embeddings.numpy())
        kmeans.fit(train_pcs)
        test_km_labels = kmeans.predict(test_pcs)
    elif method == ClassificationMethod.SVD:
        u_train: Float[np.ndarray, "batch s_vector"]
        s_train: Float[np.ndarray, "s_vector"]
        v_train: Float[np.ndarray, "s_vector d_model"]
        u_train, s_train, v_train = np.linalg.svd(train_embeddings.numpy())
        train_pcs = train_embeddings @ v_train[:, :n_components]
        _, _, v_test = np.linalg.svd(test_embeddings.numpy())
        test_pcs = test_embeddings @ v_test[:, :n_components]
        kmeans.fit(train_pcs)
        test_km_labels = kmeans.predict(test_pcs)
    train_km_labels: Int[np.ndarray, "batch"] = kmeans.labels_
    km_centroids: Float[np.ndarray, "cluster d_model"] = kmeans.cluster_centers_

    km_first_cluster, km_second_cluster = split_by_label(
        train_data.str_labels, train_km_labels
    )
    one_pos = len(set(km_first_cluster) & set(train_positive_str_labels))
    one_neg = len(set(km_first_cluster) & set(train_negative_str_labels))
    two_pos = len(set(km_second_cluster) & set(train_positive_str_labels))
    two_neg = len(set(km_second_cluster) & set(train_negative_str_labels))
    pos_first = one_pos + two_neg > one_neg + two_pos
    if pos_first:
        train_positive_cluster = km_first_cluster
        train_negative_cluster = km_second_cluster
        km_positive_centroid = km_centroids[0, :]
        km_negative_centroid = km_centroids[1, :]
        test_positive_cluster, test_negative_cluster = split_by_label(
            test_data.str_labels, test_km_labels
        )
    else:
        train_positive_cluster = km_second_cluster
        train_negative_cluster = km_first_cluster
        km_positive_centroid = km_centroids[1, :]
        km_negative_centroid = km_centroids[0, :]
        test_negative_cluster, test_positive_cluster = split_by_label(
            test_data.str_labels, test_km_labels
        )
        train_km_labels = 1 - train_km_labels
        test_km_labels = 1 - test_km_labels
    if method == ClassificationMethod.KMEANS:
        line: Float[np.ndarray, "d_model"] = (
            km_positive_centroid - km_negative_centroid
        )
    elif method == ClassificationMethod.PCA:
        line: Float[np.ndarray, "d_model"]  = (
            pca.components_[0, :] / np.linalg.norm(pca.components_[0, :])
        )
    elif method == ClassificationMethod.SVD:
        line: Float[np.ndarray, "d_model"]  = (
            v_train[:, 0] / np.linalg.norm(v_train[:, 0])
        ) * np.sign(s_train[0])
    elif method == ClassificationMethod.MEAN_DIFF:
        train_pos_embeddings = train_embeddings[train_data.binary_labels == 1, :]
        train_neg_embeddings = train_embeddings[train_data.binary_labels == 0, :]
        line: Float[np.ndarray, "d_model"]  = (
            train_pos_embeddings.mean(axis=0) - train_neg_embeddings.mean(axis=0)
        )
    # get accuracy
    _, _, insample_accuracy = get_accuracy(
        train_positive_cluster,
        train_negative_cluster,
        train_positive_str_labels,
        train_negative_str_labels,
    )
    correct, total, accuracy = get_accuracy(
        test_positive_cluster,
        test_negative_cluster,
        test_positive_str_labels,
        test_negative_str_labels,
    )
    # insample accuracy check
    is_insample = (
        train_data.prompt_type == test_data.prompt_type and
        train_pos == test_pos and
        train_layer == test_layer
    )
    if is_insample:
        assert accuracy >= 0.5, (
            f"Accuracy should be at least 50%, got {accuracy:.1%}, "
            f"direct calc:{insample_accuracy:.1%}, "
            f"train:{train_data.prompt_type.value}, layer:{train_layer}, \n"
            f"positive cluster: {sorted(test_positive_cluster)}\n"
            f"negative cluster: {sorted(test_negative_cluster)}\n"
            f"positive adjectives: {sorted(test_positive_str_labels)}\n"
            f"negative adjectives: {sorted(test_negative_str_labels)}\n"
        )
    if method == ClassificationMethod.PCA:
        # 
        plot_data = [[
            train_data.prompt_type.value, train_layer, train_pos,
            test_data.prompt_type.value, test_layer, test_pos,
            train_pcs, train_data.str_labels, train_data.binary_labels, 
            test_pcs, test_data.str_labels, test_data.binary_labels,
            km_centroids
        ]]
        plot_columns = [
            'train_set', 'train_layer', 'train_pos',
            'test_set', 'test_layer',  'test_pos',
            'train_pcs', 'train_str_labels', 'train_true_labels',
            'test_pcs', 'test_str_labels', 'test_true_labels',
            'pca_centroids',
        ]
        plot_df = pd.DataFrame(plot_data, columns=plot_columns)
        update_csv(
            plot_df, "pca_plot", train_data.model, 
            key_cols=(
                'train_set', 'train_pos', 'train_layer', 'test_set', 'test_pos', 'test_layer'
            )
        )
    
    return line, correct, total, accuracy


def _fit_logistic_regression(
    train_data: ResidualStreamDataset, train_pos: str, train_layer: int,
    test_data: ResidualStreamDataset, test_pos: str, test_layer: int,
    random_state: int = 0,
    solver: str = 'liblinear',
    max_iter: int = 1000,
    tol: float = 1e-4,  
):
    train_embeddings = train_data.embed(train_pos, train_layer)
    test_embeddings = test_data.embed(test_pos, test_layer)
    lr = LogisticRegression(
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
    )
    lr.fit(train_embeddings, train_data.binary_labels)
    total = len(test_data.binary_labels)
    accuracy = lr.score(test_embeddings, test_data.binary_labels)
    correct = int(accuracy * total)
    line = lr.coef_[0, :]
    return line, correct, total, accuracy



def train_classifying_direction(
    train_data: ResidualStreamDataset, train_pos: str, train_layer: int,
    test_data: ResidualStreamDataset, test_pos: str, test_layer: int,
    method: ClassificationMethod,
    **kwargs,
):
    if method == ClassificationMethod.PCA:
        assert 'n_components' in kwargs, "Must specify n_components for PCA"
    model = train_data.model
    if method == ClassificationMethod.LOGISTIC_REGRESSION:
        fitting_method = _fit_logistic_regression
    else:
        fitting_method = partial(_fit, method=method)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # Turn the warning into an error
        try:
            train_line, correct, total, accuracy = fitting_method(
                train_data, train_pos, train_layer,
                test_data, test_pos, test_layer,
                **kwargs,
            )
            test_line, _, _, _ = fitting_method(
                test_data, test_pos, test_layer,
                test_data, test_pos, test_layer,
                **kwargs,
            )
        except ConvergenceWarning:
            print(
                f"Convergence warning for {method.value}; "
                f"train type:{train_data.prompt_type.value}, pos: {train_pos}, layer:{train_layer}, "
                f"test type:{test_data.prompt_type.value}, pos: {test_pos}, layer:{test_layer}, "
                f"kwargs: {kwargs}\n"
                f"train str_labels:{train_data.str_labels}\n"
                f"test str_labels:{test_data.str_labels}\n"
            )
            return
    # write line to file
    save_array(train_line, f"{method.value}_{train_data.prompt_type.value}_{train_pos}_layer{train_layer}", model)

    cosine_sim = safe_cosine_sim(train_line, test_line)
    columns = [
        'train_set', 'train_layer', 'train_pos',
        'test_set', 'test_layer',  'test_pos',
        'method',
        'correct', 'total', 'accuracy', 'similarity',
    ]
    data = [[
        train_data.prompt_type.value, train_layer, train_pos,
        test_data.prompt_type.value, test_layer, test_pos,
        method.value,
        correct, total, accuracy, cosine_sim,
    ]]
    stats_df = pd.DataFrame(data, columns=columns)
    
    update_csv(
        stats_df, "direction_fitting_stats", model, 
        key_cols=(
            'method', 'train_set', 'train_pos', 'train_layer', 'test_set', 'test_pos', 'test_layer'
        )
    )