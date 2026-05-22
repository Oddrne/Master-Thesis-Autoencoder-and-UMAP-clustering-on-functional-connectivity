"""
Cleaned utilities for evaluating DCEC clustering results.

This file keeps the useful functionality from the original Evaluate_models.py while
removing duplicated imports/functions and reducing hard-coded paths.

Main functionality
------------------
1. Compute internal clustering metrics from DCEC label and latent-space files.
2. Plot internal clustering metrics across cluster numbers.
3. Plot t-SNE visualisations of latent-space cluster assignments.
4. Plot CCA/MLR behavioural evaluation scores from JSON files.
5. Compare selected variables and removed subjects across clusters/runs.
6. Compare age groups across behavioural metrics.
7. Summarise selected-variable frequencies from JSON files.
"""

from __future__ import annotations

import colorsys
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)


PathLike = Union[str, Path]
JSONDict = Dict[str, Any]


# =============================================================================
# Generic helper functions
# =============================================================================


def load_json_results(json_path: PathLike) -> JSONDict:
    """Load a JSON file."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Alias used in later helper functions.
load_json = load_json_results


def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
    """
    Safely retrieve a nested dictionary value using a colon-separated key path.

    Example
    -------
    get_nested_value(cluster_data, "cca_removed_subjects_results:cv_mean_cc")
    """
    current: Any = data
    for key in key_path.split(":"):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_cluster_number(name: str) -> Optional[int]:
    """
    Extract cluster number from a filename or JSON key.

    Supports examples such as:
    - Cluster_2_results
    - cluster_2
    - Cluster-2
    - k_2
    - 2_clusters
    """
    patterns = [
        r"[Cc]luster[_\- ]?(\d+)",
        r"[_\-]k[_\-]?(\d+)",
        r"(\d+)[_\- ]?[Cc]lusters?",
    ]

    for pattern in patterns:
        match = re.search(pattern, str(name))
        if match:
            return int(match.group(1))

    return None


def cluster_sort_key(cluster_name: str) -> int:
    """Sort JSON cluster keys such as 'Cluster_2_results' numerically."""
    cluster_number = extract_cluster_number(cluster_name)
    if cluster_number is None:
        raise ValueError(f"Could not extract cluster number from: {cluster_name}")
    return cluster_number


def parse_model_filename(filename: str) -> Dict[str, str]:
    """
    Parse metadata from DCEC model output filenames.

    Expected pattern inside filename:
    DCEC_400x400_68_O_subjects_run1_...
    """
    name = Path(filename).stem
    match = re.search(
        r"DCEC_(?P<parcellation>\d+x\d+)_(?P<n_subjects>\d+)_(?P<age_group>[OY])_subjects_(?P<run>run\d+)",
        name,
        flags=re.IGNORECASE,
    )

    if match is None:
        return {
            "parcellation": "unknown",
            "n_subjects": "unknown",
            "age_group": "unknown",
            "run": "unknown",
            "age_label": "unknown",
            "movie": "unknown",
        }

    metadata = match.groupdict()
    age_group = metadata["age_group"].upper()
    run = metadata["run"].lower()

    metadata["age_label"] = "Young" if age_group == "Y" else "Old"
    metadata["movie"] = "neutral" if run == "run1" else "negative" if run == "run2" else run
    return metadata


def parse_behaviour_json_filename(path: PathLike) -> Optional[Dict[str, Any]]:
    """
    Parse behavioural JSON filenames.

    Expected filename:
    {parcellation}_{age group}_{movie number}_cluster_behavioural_results.json

    Example:
    1000x1000_Young_run1_cluster_behavioural_results.json

    Returns None if the filename does not match the expected pattern.
    """
    path = Path(path)
    pattern = re.compile(
        r"(?P<parcellation>\d+x\d+)_(?P<age_group>Young|Old)_run(?P<run>\d+)_cluster_behavioural_results\.json$",
        flags=re.IGNORECASE,
    )

    match = pattern.match(path.name)
    if match is None:
        return None

    parcellation = match.group("parcellation")
    age_group = match.group("age_group").lower()
    run = int(match.group("run"))
    movie = "neutral" if run == 1 else "negative" if run == 2 else f"run{run}"

    return {
        "path": path,
        "parcellation": parcellation,
        "age_group": age_group,
        "movie": movie,
        "run": run,
    }


def adjust_lightness(color: str, amount: float = 1.2) -> Tuple[float, float, float]:
    """
    Lighten or darken a matplotlib colour.

    amount > 1 gives a lighter colour.
    amount < 1 gives a darker colour.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * amount))
    return colorsys.hls_to_rgb(h, l, s)


def format_list(values: Sequence[Any], max_len: int = 5) -> str:
    """Format a list for figure text/annotation without making huge text blocks."""
    if not values:
        return "None"
    if len(values) <= max_len:
        return ", ".join(map(str, values))
    return ", ".join(map(str, values[:max_len])) + ", ..."


def _safe_list(value: Any) -> List[Any]:
    """Convert list-like JSON values to a list; return [] for missing/null values."""
    return value if isinstance(value, list) else []


def _safe_set(value: Any) -> set:
    """Convert list-like JSON values to a set; return empty set for missing/null values."""
    return set(value) if isinstance(value, list) else set()


def _normalize_keys(keys: Union[str, Sequence[str]]) -> List[str]:
    """Allow a single key string or a list/tuple of key strings."""
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, (list, tuple)):
        return list(keys)
    raise TypeError("keys must be a string or a list/tuple of strings.")


def _combine_keys_as_set(data_dict: Dict[str, Any], keys: Union[str, Sequence[str]]) -> set:
    """Combine multiple list-valued keys from a JSON cluster result into one set."""
    combined = set()
    for key in _normalize_keys(keys):
        combined |= _safe_set(data_dict.get(key))
    return combined


# =============================================================================
# Internal clustering metrics
# =============================================================================


def evaluate_single_clustering(
    functional_connectivity_matrix: np.ndarray,
    labels_path: Union[PathLike, Sequence[int], np.ndarray],
    print_results: bool = True,
    invalid_value: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

    Parameters
    ----------
    functional_connectivity_matrix:
        Data matrix used for evaluation. In this thesis this is usually the
        latent representation from the middle layer.
    labels_path:
        Either a path to labels or an array/list of labels.
    print_results:
        If True, print the scores.
    invalid_value:
        Returned for all metrics if fewer than two clusters exist.

    Returns
    -------
    (silhouette, davies_bouldin, calinski_harabasz)
    """
    labels = np.loadtxt(labels_path) if isinstance(labels_path, (str, Path)) else np.asarray(labels_path)

    if len(np.unique(labels)) < 2:
        silhouette_avg = davies_bouldin_avg = calinski_harabasz_avg = invalid_value
        if print_results:
            print("Only one cluster present; clustering metrics not computed.")
    else:
        silhouette_avg = float(silhouette_score(functional_connectivity_matrix, labels))
        davies_bouldin_avg = float(davies_bouldin_score(functional_connectivity_matrix, labels))
        calinski_harabasz_avg = float(calinski_harabasz_score(functional_connectivity_matrix, labels))

    if print_results:
        print("\nScores for the given labels:")
        print(f"Silhouette coefficient: {silhouette_avg}")
        print(f"Davies-Bouldin score: {davies_bouldin_avg}")
        print(f"Calinski-Harabasz score: {calinski_harabasz_avg}")

    return silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg


def compute_clustering_scores(
    embedding: np.ndarray,
    labels: Sequence[int],
    invalid_value: float = np.nan,
) -> Dict[str, Any]:
    """Compute internal clustering metrics and return a dictionary."""
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return {
            "silhouette": invalid_value,
            "davies_bouldin": invalid_value,
            "calinski_harabasz": invalid_value,
            "valid": False,
        }

    return {
        "silhouette": float(silhouette_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
        "valid": True,
    }


def collect_clustering_scores_from_folder(
    models_dir: PathLike = "Clusters",
    model_prefix: Optional[str] = None,
    labels_tag: str = "labels_predicted_labels_",
    middle_layer_tag: str = "middle_layer_predicted_labels_",
    cluster_range: Iterable[int] = range(2, 11),
    invalid_value: float = np.nan,
    print_results: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Collect internal clustering scores from a folder of DCEC output files.

    Expected paired files:
    - labels_predicted_labels_...
    - middle_layer_predicted_labels_...

    Returns
    -------
    dict keyed by cluster number.
    """
    models_dir = Path(models_dir)
    cluster_range = set(cluster_range)
    results: Dict[int, Dict[str, Any]] = {}

    for labels_file in sorted(models_dir.glob("*.txt")):
        filename = labels_file.name

        if labels_tag not in filename:
            continue
        if model_prefix is not None and not filename.startswith(model_prefix):
            continue

        cluster_number = extract_cluster_number(filename)
        if cluster_number is None:
            print(f"Skipping {filename}: could not extract cluster number.")
            continue
        if cluster_number not in cluster_range:
            continue

        middle_layer_file = Path(str(labels_file).replace(labels_tag, middle_layer_tag))
        if not middle_layer_file.exists():
            print(f"Skipping {filename}: missing middle-layer file.")
            continue

        try:
            labels = np.loadtxt(labels_file)
            embedding = np.loadtxt(middle_layer_file)
            scores = compute_clustering_scores(embedding, labels, invalid_value=invalid_value)

            results[cluster_number] = {
                **scores,
                "file": filename,
                "labels_file": str(labels_file),
                "middle_layer_file": str(middle_layer_file),
                "metadata": parse_model_filename(filename),
            }

            if print_results:
                print(f"\nScores for {filename}:")
                print(f"Silhouette: {scores['silhouette']}")
                print(f"Davies-Bouldin: {scores['davies_bouldin']}")
                print(f"Calinski-Harabasz: {scores['calinski_harabasz']}")

        except Exception as error:
            print(f"Skipping {filename}: {error}")

    return dict(sorted(results.items()))


def Calculate_clustering_scores_from_folder(
    models_dir: PathLike = "Clusters",
    model_prefix: Optional[str] = None,
    labels_tag: str = "labels_predicted_labels_",
    middle_layer_tag: str = "middle_layer_predicted_labels_",
    print_results: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Backwards-compatible wrapper for the old function name.

    Returns a dictionary keyed by filename, as the old code did.
    New code should preferably use collect_clustering_scores_from_folder().
    """
    by_cluster = collect_clustering_scores_from_folder(
        models_dir=models_dir,
        model_prefix=model_prefix,
        labels_tag=labels_tag,
        middle_layer_tag=middle_layer_tag,
        print_results=print_results,
        invalid_value=0.0,
    )

    return {
        value["file"]: {
            "silhouette": value["silhouette"],
            "davies_bouldin": value["davies_bouldin"],
            "calinski_harabasz": value["calinski_harabasz"],
        }
        for value in by_cluster.values()
    }


def _scores_to_dataframe(results: Dict[Any, Dict[str, Any]]) -> pd.DataFrame:
    """Convert either filename-keyed or cluster-keyed score dictionaries to a DataFrame."""
    rows = []

    for key, value in results.items():
        if isinstance(key, int):
            cluster = key
            filename = value.get("file", f"Cluster_{cluster}")
        else:
            filename = str(key)
            cluster = extract_cluster_number(filename)

        rows.append(
            {
                "cluster": cluster,
                "file": filename,
                "silhouette": value.get("silhouette"),
                "davies_bouldin": value.get("davies_bouldin"),
                "calinski_harabasz": value.get("calinski_harabasz"),
                "valid": value.get("valid", True),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "cluster" in df.columns:
        df = df.sort_values("cluster", na_position="last")
    return df


def plot_scores(
    results: Dict[Any, Dict[str, Any]],
    save_path: Optional[PathLike] = None,
    sort_scores: bool = True,
    annotate: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> Dict[str, Tuple[Any, Any]]:
    """
    Backwards-compatible plotting function.

    Plots three separate bar plots:
    - Silhouette
    - Davies-Bouldin
    - Calinski-Harabasz
    """
    df = _scores_to_dataframe(results)
    if df.empty:
        print("No results to plot.")
        return {}

    if sort_scores:
        df = df.sort_values("cluster", na_position="last")

    x_labels = [f"Cluster {int(c)}" if pd.notna(c) else str(f) for c, f in zip(df["cluster"], df["file"])]
    metrics = [
        ("silhouette", "Silhouette score", "[-1, 1] Higher is better."),
        ("davies_bouldin", "Davies-Bouldin score", "[0, ∞) Lower is better."),
        ("calinski_harabasz", "Calinski-Harabasz score", "[0, ∞) Higher is better."),
    ]

    figures = {}
    for metric, ylabel, footnote in metrics:
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(x_labels, df[metric].astype(float))
        ax.set_xlabel("Cluster")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} across clusters")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        if annotate:
            for bar, score in zip(bars, df[metric].astype(float)):
                if not np.isnan(score):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )

        fig.text(0.5, -0.02, footnote, ha="center", fontsize=12)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            fig.savefig(save_path.with_name(f"{save_path.stem}_{metric}.png"), dpi=300, bbox_inches="tight")

        plt.show()
        figures[metric] = (fig, ax)

    return figures


def plot_clustering_scores_sorted(
    models_dir: PathLike = "Clusters",
    model_prefix: Optional[str] = None,
    labels_tag: str = "labels_predicted_labels_",
    middle_layer_tag: str = "middle_layer_predicted_labels_",
    print_results: bool = False,
    save_path: Optional[PathLike] = None,
    normalize: bool = False,
    cluster_range: Iterable[int] = range(2, 11),
    ytop: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 7),
    annotate_points: bool = True,
    ch_scale: float = 1000.0,
    ylim: float = 1.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute and plot Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

    The scores are sorted by cluster number. The Calinski-Harabasz score is
    divided by ch_scale by default to make it easier to compare visually with
    the other two metrics.
    """
    results = collect_clustering_scores_from_folder(
        models_dir=models_dir,
        model_prefix=model_prefix,
        labels_tag=labels_tag,
        middle_layer_tag=middle_layer_tag,
        cluster_range=cluster_range,
        invalid_value=np.nan,
        print_results=print_results,
    )

    if not results:
        print("No valid models found for plotting.")
        return results

    clusters = np.array(sorted(results.keys()))
    silhouette_vals = np.array([results[k]["silhouette"] for k in clusters], dtype=float)
    davies_vals = np.array([results[k]["davies_bouldin"] for k in clusters], dtype=float)
    calinski_vals = np.array([results[k]["calinski_harabasz"] for k in clusters], dtype=float) / ch_scale

    first_metadata = next(iter(results.values())).get("metadata", {})
    title = (
        f"Clustering scores for the {first_metadata.get('movie', 'unknown')} movie | "
        f"Parcellation: {first_metadata.get('parcellation', 'unknown')} | "
        f"Subjects: {first_metadata.get('age_group', 'unknown')}"
    )

    def minmax_scale(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return values
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        if np.isclose(min_val, max_val):
            return np.ones_like(values) * 0.5
        return (values - min_val) / (max_val - min_val)

    if normalize:
        y_sil = minmax_scale(silhouette_vals)
        y_db = minmax_scale(davies_vals)
        y_ch = minmax_scale(calinski_vals)
        ylabel = "Normalized score"
    else:
        y_sil, y_db, y_ch = silhouette_vals, davies_vals, calinski_vals
        ylabel = "Score"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(clusters, y_sil, marker="o", label="Silhouette")
    ax.plot(clusters, y_db, marker="s", label="Davies-Bouldin")
    ax.plot(clusters, y_ch, marker="^", label=f"Calinski-Harabasz / {int(ch_scale)}")

    if annotate_points:
        for x, y in zip(clusters, y_sil):
            if np.isfinite(y):
                ax.annotate(f"Cluster {x}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=12)

    ax.set_ylim(top=ylim)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(cluster_range))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if ytop is not None:
        ax.set_ylim(bottom=-0.1, top=ytop)

    fig.text(
        0.5,
        -0.02,
        "Silhouette [-1, 1] ↑ | Davies-Bouldin [0, ∞) ↓ | Calinski-Harabasz [0, ∞) ↑ (scaled)",
        ha="center",
        fontsize=12,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
    return results


def calculate_and_plot_scores_from_folder(**kwargs) -> Dict[int, Dict[str, Any]]:
    """Convenience wrapper around plot_clustering_scores_sorted()."""
    return plot_clustering_scores_sorted(**kwargs)


# =============================================================================
# t-SNE visualisation of cluster assignments
# =============================================================================


def plot_tsne(
    embedding: np.ndarray,
    labels: Sequence[int],
    title: str = "t-SNE visualisation of clusters",
    save_path: Optional[PathLike] = None,
    figsize: Tuple[int, int] = (6, 4),
    figtext: str = None,
    perplexity: Optional[float] = None,
    random_state: int = 42,
    annotate: bool = False,
) -> Dict[str, Any]:
    """Create a t-SNE plot from a latent representation and cluster labels."""
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)

    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)

    n_samples = embedding.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE requires at least three samples.")

    if perplexity is None:
        perplexity = min(30, max(2, (n_samples - 1) // 3))
    perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embedding)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=45, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    if figtext is not None:
        fig.text(
            0.5,
            -0.02,
            figtext,
            ha="center",
            va="bottom",
            fontsize=12
        )

    if annotate:
        for idx, (x, y) in enumerate(coords):
            ax.annotate(str(idx), (x, y), fontsize=12, alpha=0.7)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"coordinates": coords, "labels": labels, "fig": fig, "ax": ax}


def plot_clustering(
    labels_path: Optional[PathLike] = None,
    middle_layer_path: Optional[PathLike] = None,
    labels: Optional[Sequence[int]] = None,
    embedding: Optional[np.ndarray] = None,
    title: str = "t-SNE visualisation of clusters",
    figtext: str = None,
    save_path: Optional[PathLike] = None,
    **tsne_kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot a t-SNE visualisation of one clustering.

    Either provide labels and embedding directly, or provide paths to .txt files.
    """
    if labels is None:
        if labels_path is None:
            raise ValueError("Provide either labels or labels_path.")
        labels = np.loadtxt(labels_path)

    if embedding is None:
        if middle_layer_path is None:
            raise ValueError("Provide either embedding or middle_layer_path.")
        embedding = np.loadtxt(middle_layer_path)

    return plot_tsne(embedding=embedding, labels=labels, title=title, save_path=save_path, figtext=figtext, **tsne_kwargs)


# =============================================================================
# CCA/MLR plotting from behavioural JSON files
# =============================================================================


def _cluster_values_from_json(
    results: JSONDict,
    metric_path: str,
    cluster_numbers: Sequence[int],
    dead_value: float = 0.0,
) -> List[float]:
    values = []
    for cluster_num in cluster_numbers:
        cluster_data = results.get(f"Cluster_{cluster_num}_results", {})
        value = get_nested_value(cluster_data, metric_path)
        values.append(dead_value if value is None else float(value))
    return values


def plot_cca_mlr_across_clusters(
    json_path: PathLike,
    cca_metric: str = "cca_removed_subjects_results:cv_mean_cc",
    mlr_metric: str = "mlr_removed_subjects_results:mean_accuracy",
    title: str = "CCA and MLR across clusters",
    xlabel: str = "Cluster",
    ylabel: str = "Score",
    save_path: Optional[PathLike] = None,
    annotate: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    ylim: float = 1.0,
    dead_value: float = 0.0,
) -> Dict[str, Any]:
    """Plot CCA and MLR scores across cluster numbers for one behavioural JSON file."""
    results = load_json_results(json_path)
    cluster_numbers = sorted(cluster_sort_key(key) for key in results if key.startswith("Cluster_"))

    cca_values = _cluster_values_from_json(results, cca_metric, cluster_numbers, dead_value=dead_value)
    mlr_values = _cluster_values_from_json(results, mlr_metric, cluster_numbers, dead_value=dead_value)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cluster_numbers, cca_values, marker="o", label="CCA")
    ax.plot(cluster_numbers, mlr_values, marker="o", label="MLR")

    if annotate:
        for x, y in zip(cluster_numbers, cca_values):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=12)
        for x, y in zip(cluster_numbers, mlr_values):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=12)

    ax.set_xticks(cluster_numbers)
    ax.set_xticklabels([f"Cluster {n}" for n in cluster_numbers], rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0, top=ylim)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.text(
        0.5,
        -0.02,
        "CCA [0, 1] and MLR [0, 1]. Both cross validated.",
        ha="center",
        fontsize=12,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"cluster_numbers": cluster_numbers, "cca_values": cca_values, "mlr_values": mlr_values, "fig": fig, "ax": ax}


def plot_single_cluster_cca_mlr_variants(
    json_path: PathLike,
    cluster_number: int,
    cca_value_key: str = "cv_mean_cc",
    mlr_value_key: str = "mean_accuracy",
    dead_value: float = 0.0,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 7),
    ylim: float = 1.0,
) -> Dict[str, Any]:
    """
    Plot CCA/MLR scores for one cluster across:
    - all variables
    - selected variables
    - removed subjects
    """
    results = load_json_results(json_path)
    cluster_key = f"Cluster_{cluster_number}_results"
    if cluster_key not in results:
        raise ValueError(f"{cluster_key} not found in {json_path}.")

    cluster_data = results[cluster_key]
    categories = ["All variables", "Selected variables", "Filtered subjects"]

    cca_paths = [
        f"cca_all_variables_results:{cca_value_key}",
        f"cca_selected_variables_results:{cca_value_key}",
        f"cca_removed_subjects_results:{cca_value_key}",
    ]
    mlr_paths = [
        f"mlr_all_variables_results:{mlr_value_key}",
        f"mlr_selected_variables_results:{mlr_value_key}",
        f"mlr_removed_subjects_results:{mlr_value_key}",
    ]

    cca_values = [dead_value if get_nested_value(cluster_data, p) is None else get_nested_value(cluster_data, p) for p in cca_paths]
    mlr_values = [dead_value if get_nested_value(cluster_data, p) is None else get_nested_value(cluster_data, p) for p in mlr_paths]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, cca_values, marker="o", label="CCA")
    ax.plot(x, mlr_values, marker="o", label="MLR")

    if annotate:
        for xi, yi in zip(x, cca_values):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 7), textcoords="offset points", ha="center")
        for xi, yi in zip(x, mlr_values):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, -14), textcoords="offset points", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_title(title or f"Cluster {cluster_number}: CCA and MLR across result types")
    ax.set_ylim(0, ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_text = (
        f"CCA selected variables: {format_list(cluster_data.get('cca_selected_variables', []), 3)}\n"
        f"MLR selected variables: {format_list(cluster_data.get('mlr_selected_variables', []), 3)}\n"
        f"CCA removed subjects: {format_list(cluster_data.get('cca_removed_subjects', []), 3)}\n"
        f"MLR removed subjects: {format_list(cluster_data.get('mlr_removed_subjects', []), 3)}"
    )
    fig.text(0.02, -0.03, fig_text, ha="left", va="bottom", fontsize=12)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"categories": categories, "cca_values": cca_values, "mlr_values": mlr_values, "figure_text": fig_text, "fig": fig, "ax": ax}



def plot_single_cluster_score_variants_multiple_runs(
    json_paths: Sequence[PathLike],
    cluster_number: int,
    score_type: str = "cca",
    condition_labels: Optional[Sequence[str]] = None,
    value_key: Optional[str] = None,
    dead_value: float = np.nan,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    annotate: bool = True,
    figsize: Tuple[int, int] = (7, 4),
    y_lim: Optional[Tuple[float, float]] = (0, 1),
    marker_size: float = 60,
    show: bool = True,
) -> pd.DataFrame:
    """
    Plot CCA or MLR values for one chosen cluster number across multiple JSON files.

    For each JSON path, the function plots the selected score type for:
        - all variables
        - selected variables
        - filtered subjects

    Parameters
    ----------
    json_paths:
        List of behavioural result JSON files. Each file represents one run condition.

    cluster_number:
        Cluster number to plot, for example 2, 3, ..., 10.

    score_type:
        Which score type to plot:
            - "cca"
            - "mlr"

    condition_labels:
        Labels used in the legend. If None, filename stems are used.

    value_key:
        Which value to extract from the result dictionaries.
        If None:
            - CCA uses "cv_mean_cc"
            - MLR uses "mean_accuracy"

    dead_value:
        Value used when a score is missing.
        Default np.nan shows missing values as gaps.

    Returns
    -------
    plot_df:
        DataFrame containing all extracted values used in the plot.
    """

    score_type = score_type.lower().strip()

    if score_type not in {"cca", "mlr"}:
        raise ValueError("score_type must be either 'cca' or 'mlr'.")

    if value_key is None:
        value_key = "cv_mean_cc" if score_type == "cca" else "mean_accuracy"

    json_paths = [Path(p) for p in json_paths]

    if condition_labels is None:
        condition_labels = [p.stem for p in json_paths]
    else:
        condition_labels = list(condition_labels)

    if len(condition_labels) != len(json_paths):
        raise ValueError("condition_labels must have the same length as json_paths.")

    def _get_results_container(loaded_json):
        """
        Supports both:
            {"Cluster_2_results": {...}}
        and:
            {"results": {"Cluster_2_results": {...}}}
        """
        if isinstance(loaded_json, dict) and "results" in loaded_json:
            return loaded_json["results"]
        return loaded_json

    def _safe_float(value):
        if value is None:
            return dead_value

        try:
            value = float(value)
        except (TypeError, ValueError):
            return dead_value

        if not np.isfinite(value):
            return dead_value

        return value

    cluster_key = f"Cluster_{cluster_number}_results"

    result_specs = [
        {
            "category": "All variables",
            "result_path": f"{score_type}_all_variables_results:{value_key}",
            "variables_key": None,
            "subjects_key": None,
        },
        {
            "category": "Selected variables",
            "result_path": f"{score_type}_selected_variables_results:{value_key}",
            "variables_key": f"{score_type}_selected_variables",
            "subjects_key": None,
        },
        {
            "category": "Filtered subjects",
            "result_path": f"{score_type}_removed_subjects_results:{value_key}",
            "variables_key": f"{score_type}_selected_variables",
            "subjects_key": f"{score_type}_removed_subjects",
        },
    ]

    rows = []

    for json_path, condition_label in zip(json_paths, condition_labels):
        loaded_json = load_json_results(json_path)
        results = _get_results_container(loaded_json)

        if cluster_key not in results:
            raise ValueError(f"{cluster_key} not found in {json_path}.")

        cluster_data = results[cluster_key]

        for spec in result_specs:
            value = get_nested_value(cluster_data, spec["result_path"])

            variables = (
                cluster_data.get(spec["variables_key"], [])
                if spec["variables_key"] is not None
                else []
            )

            removed_subjects = (
                cluster_data.get(spec["subjects_key"], [])
                if spec["subjects_key"] is not None
                else []
            )

            rows.append(
                {
                    "condition": condition_label,
                    "json_path": str(json_path),
                    "cluster_number": cluster_number,
                    "score_type": score_type,
                    "category": spec["category"],
                    "result_path": spec["result_path"],
                    "value_key": value_key,
                    "score_value": _safe_float(value),
                    "variables": variables,
                    "removed_subjects": removed_subjects,
                }
            )

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        raise ValueError("No values were extracted.")

    categories = [spec["category"] for spec in result_specs]

    category_to_x = {
        category: idx
        for idx, category in enumerate(categories)
    }

    condition_order = list(condition_labels)

    if len(condition_order) == 1:
        condition_offsets = {condition_order[0]: 0.0}
    else:
        offsets = np.linspace(-0.18, 0.18, len(condition_order))
        condition_offsets = dict(zip(condition_order, offsets))

    fig, ax = plt.subplots(figsize=figsize)

    for condition in condition_order:
        condition_df = plot_df[
            plot_df["condition"] == condition
        ].copy()

        if condition_df.empty:
            continue

        condition_df["x_base"] = condition_df["category"].map(category_to_x)
        condition_df = condition_df.sort_values("x_base")

        x = (
            condition_df["x_base"].to_numpy(dtype=float)
            + condition_offsets[condition]
        )

        y = condition_df["score_value"].to_numpy(dtype=float)

        # Same colour logic as plot_cca_mlr_side_by_side_same_selected_variables:
        # let Matplotlib choose the line colour, then reuse it for scatter.
        line = ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.2,
            alpha=0.75,
            label=condition,
        )

        line_color = line[0].get_color()

        ax.scatter(
            x,
            y,
            s=marker_size,
            color=line_color,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

        if annotate:
            for x_i, y_i in zip(x, y):
                if np.isfinite(y_i):
                    ax.annotate(
                        f"{y_i:.3f}",
                        xy=(x_i, y_i),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    # ax.set_xlabel("Evaluation variant")

    if score_type == "cca":
        ylabel = "CCA value"
        default_title = (
            f"Cluster {cluster_number}: CCA values across runs "
            f"({value_key})"
        )
    else:
        ylabel = "MLR accuracy"
        default_title = (
            f"Cluster {cluster_number}: MLR values across runs "
            f"({value_key})"
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title, fontsize=10)
    ax.grid(True, alpha=0.25)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        fontsize=8,
        title="Condition",
    )

    fig.text(
        0.5,
        -0.03,
        (
            f"{score_type.upper()} value key: {value_key}. "
            "All variables, selected variables, and filtered subjects are compared "
            "for the same cluster number."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.04, 0.86, 0.92])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return plot_df



# =============================================================================
# Comparing common variables and removed subjects
# =============================================================================


def _extract_common_and_union(dict_of_sets: Dict[str, set], ignore_empty: bool = True) -> Dict[str, Any]:
    items = {k: v for k, v in dict_of_sets.items() if (len(v) > 0 or not ignore_empty)}
    if not items:
        return {"common": set(), "union": set(), "used_entries": []}

    sets = list(items.values())
    return {
        "common": set.intersection(*sets) if sets else set(),
        "union": set.union(*sets) if sets else set(),
        "used_entries": list(items.keys()),
    }


def _count_occurrences(dict_of_sets: Dict[str, set]) -> Counter:
    counter = Counter()
    for values in dict_of_sets.values():
        counter.update(values)
    return counter


def compare_common_across_clusters(
    json_path: PathLike,
    variable_keys: Union[str, Sequence[str]] = ("cca_selected_variables", "mlr_selected_variables"),
    removed_subjects_keys: Union[str, Sequence[str]] = ("cca_removed_subjects", "mlr_removed_subjects"),
    clusters_to_include: Optional[Sequence[int]] = None,
    ignore_empty: bool = True,
) -> Dict[str, Any]:
    """Compare selected variables and removed subjects across clusters within one JSON file."""
    results = load_json_results(json_path)
    cluster_variable_sets: Dict[str, set] = {}
    cluster_removed_sets: Dict[str, set] = {}

    for cluster_name, cluster_data in sorted(results.items(), key=lambda item: cluster_sort_key(item[0])):
        if not cluster_name.startswith("Cluster_"):
            continue
        cluster_num = cluster_sort_key(cluster_name)
        if clusters_to_include is not None and cluster_num not in clusters_to_include:
            continue

        short_name = f"Cluster_{cluster_num}"
        cluster_variable_sets[short_name] = _combine_keys_as_set(cluster_data, variable_keys)
        cluster_removed_sets[short_name] = _combine_keys_as_set(cluster_data, removed_subjects_keys)

    variable_summary = _extract_common_and_union(cluster_variable_sets, ignore_empty=ignore_empty)
    removed_summary = _extract_common_and_union(cluster_removed_sets, ignore_empty=ignore_empty)
    variable_counts = _count_occurrences(cluster_variable_sets)
    removed_counts = _count_occurrences(cluster_removed_sets)

    return {
        "json_file": str(json_path),
        "variable_keys": _normalize_keys(variable_keys),
        "removed_subjects_keys": _normalize_keys(removed_subjects_keys),
        "per_cluster_variables": {k: sorted(v) for k, v in cluster_variable_sets.items()},
        "per_cluster_removed_subjects": {k: sorted(v) for k, v in cluster_removed_sets.items()},
        "common_variables": sorted(variable_summary["common"]),
        "union_variables": sorted(variable_summary["union"]),
        "variable_counts": dict(variable_counts),
        "variable_counts_sorted": sorted(variable_counts.items(), key=lambda x: (-x[1], str(x[0]))),
        "clusters_used_for_variables": variable_summary["used_entries"],
        "common_removed_subjects": sorted(removed_summary["common"]),
        "union_removed_subjects": sorted(removed_summary["union"]),
        "removed_subjects_counts": dict(removed_counts),
        "removed_subjects_counts_sorted": sorted(removed_counts.items(), key=lambda x: (-x[1], str(x[0]))),
        "clusters_used_for_removed_subjects": removed_summary["used_entries"],
    }


def compare_common_across_runs_same_cluster(
    json_paths: Sequence[PathLike],
    cluster_number: int,
    variable_keys: Union[str, Sequence[str]] = ("cca_selected_variables", "mlr_selected_variables"),
    removed_subjects_keys: Union[str, Sequence[str]] = ("cca_removed_subjects", "mlr_removed_subjects"),
    run_names: Optional[Sequence[str]] = None,
    ignore_empty: bool = True,
) -> Dict[str, Any]:
    """Compare selected variables and removed subjects across runs for the same cluster number."""
    json_paths = [Path(p) for p in json_paths]
    run_names = list(run_names) if run_names is not None else [p.stem for p in json_paths]

    if len(run_names) != len(json_paths):
        raise ValueError("run_names must have the same length as json_paths.")

    cluster_key = f"Cluster_{cluster_number}_results"
    run_variable_sets: Dict[str, set] = {}
    run_removed_sets: Dict[str, set] = {}

    for json_path, run_name in zip(json_paths, run_names):
        results = load_json_results(json_path)
        if cluster_key not in results:
            print(f"Skipping {json_path}: {cluster_key} not found.")
            continue

        cluster_data = results[cluster_key]
        run_variable_sets[run_name] = _combine_keys_as_set(cluster_data, variable_keys)
        run_removed_sets[run_name] = _combine_keys_as_set(cluster_data, removed_subjects_keys)

    variable_summary = _extract_common_and_union(run_variable_sets, ignore_empty=ignore_empty)
    removed_summary = _extract_common_and_union(run_removed_sets, ignore_empty=ignore_empty)
    variable_counts = _count_occurrences(run_variable_sets)
    removed_counts = _count_occurrences(run_removed_sets)

    return {
        "cluster_number": cluster_number,
        "variable_keys": _normalize_keys(variable_keys),
        "removed_subjects_keys": _normalize_keys(removed_subjects_keys),
        "per_run_variables": {k: sorted(v) for k, v in run_variable_sets.items()},
        "per_run_removed_subjects": {k: sorted(v) for k, v in run_removed_sets.items()},
        "common_variables": sorted(variable_summary["common"]),
        "union_variables": sorted(variable_summary["union"]),
        "variable_counts": dict(variable_counts),
        "variable_counts_sorted": sorted(variable_counts.items(), key=lambda x: (-x[1], str(x[0]))),
        "runs_used_for_variables": variable_summary["used_entries"],
        "common_removed_subjects": sorted(removed_summary["common"]),
        "union_removed_subjects": sorted(removed_summary["union"]),
        "removed_subjects_counts": dict(removed_counts),
        "removed_subjects_counts_sorted": sorted(removed_counts.items(), key=lambda x: (-x[1], str(x[0]))),
        "runs_used_for_removed_subjects": removed_summary["used_entries"],
    }


def print_comparison_summary(comparison_result: Dict[str, Any], save_path: Optional[PathLike] = None, top_n: int = 5) -> None:
    """Pretty-print the result from compare_common_across_* functions."""
    if "json_file" in comparison_result:
        print(f"\nComparison across clusters in: {comparison_result['json_file']}")
        save_results = {"json_file": comparison_result["json_file"]}
    else:
        print(f"\nComparison across runs for Cluster {comparison_result['cluster_number']}")
        save_results = {"cluster_number": comparison_result["cluster_number"]}

    print(f"Variable keys: {comparison_result['variable_keys']}")
    print(f"Removed-subject keys: {comparison_result['removed_subjects_keys']}")

    print("\nCommon variables:")
    print(comparison_result["common_variables"])

    print(f"\nTop {top_n} variable counts:")
    for var, count in comparison_result["variable_counts_sorted"][:top_n]:
        print(f"{var}: {count}")

    print("\nCommon removed subjects:")
    print(comparison_result["common_removed_subjects"])

    print(f"\nTop {top_n} removed subject counts:")
    for subject, count in comparison_result["removed_subjects_counts_sorted"][:top_n]:
        print(f"{subject}: {count}")

    save_results.update(
        {
            "common_variables": comparison_result["common_variables"],
            "variable_counts_sorted": comparison_result["variable_counts_sorted"][:top_n],
            "common_removed_subjects": comparison_result["common_removed_subjects"],
            "removed_subjects_counts_sorted": comparison_result["removed_subjects_counts_sorted"][:top_n],
        }
    )

    if save_path is not None:
        with Path(save_path).open("w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=4)
        print(f"\nComparison result saved to: {save_path}")


# =============================================================================
# Age-group comparison plots
# =============================================================================


def _collect_json_metric_series(
    json_paths: Sequence[PathLike],
    metric_to_use: str,
    dead_value: float,
    include_all_clusters: bool,
) -> Tuple[List[int], Dict[str, Dict[str, Any]]]:
    """Internal helper for age-comparison plotting."""
    parsed_runs = []
    all_clusters = set()

    for path in json_paths:
        metadata = parse_behaviour_json_filename(path)
        if metadata is None:
            print(f"Skipping non-matching JSON filename: {Path(path).name}")
            continue
        results = load_json_results(path)
        clusters = {cluster_sort_key(k) for k in results if k.startswith("Cluster_")}
        all_clusters |= clusters
        parsed_runs.append((Path(path), metadata, results, clusters))

    if not parsed_runs:
        return [], {}

    if include_all_clusters:
        cluster_numbers = sorted(all_clusters)
    else:
        common_clusters = set.intersection(*(clusters for _, _, _, clusters in parsed_runs))
        cluster_numbers = sorted(common_clusters)

    series = {}
    for path, metadata, results, _ in parsed_runs:
        label = f"{metadata['age_group'].title()} {metadata['parcellation']} run{metadata['run']}"
        values = _cluster_values_from_json(results, metric_to_use, cluster_numbers, dead_value=dead_value)
        series[label] = {"values": values, "metadata": metadata, "path": str(path)}

    return cluster_numbers, series


def plot_compare_ages_across_clusters(
    json_paths: Optional[Sequence[PathLike]] = None,
    results_dir: PathLike = "Results",
    metric_to_use: str = "cca_removed_subjects_results:cv_mean_cc",
    dead_value: float = 0.0,
    include_all_clusters: bool = True,
    annotate: bool = False,
    title: str = "Comparison of scores between age groups across clusters",
    xlabel: str = "Cluster",
    ylabel: str = "Score",
    save_path: Optional[PathLike] = None,
    figsize: Tuple[int, int] = (11, 6),
    ylim: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Plot one behavioural metric across clusters for all matching age-group JSON files.

    If json_paths is None, all files matching
    '*_cluster_behavioural_results.json' in results_dir are used.
    """
    if json_paths is None:
        json_paths = sorted(Path(results_dir).glob("*_cluster_behavioural_results.json"))

    cluster_numbers, series = _collect_json_metric_series(
        json_paths=json_paths,
        metric_to_use=metric_to_use,
        dead_value=dead_value,
        include_all_clusters=include_all_clusters,
    )

    if not series:
        print("No valid JSON files found.")
        return {"cluster_numbers": [], "series": {}}

    metric_name = "CCA" if metric_to_use.startswith("cca") else "MLR" if metric_to_use.startswith("mlr") else "Metric"
    age_colors = {"young": "green", "old": "red"}
    run_markers = {1: "o", 2: "s", 3: "^"}
    parcellation_lightness = {"400x400": 0.85, "1000x1000": 1.25}

    fig, ax = plt.subplots(figsize=figsize)

    for label, item in series.items():
        metadata = item["metadata"]
        age = metadata["age_group"]
        run = metadata["run"]
        parcellation = metadata["parcellation"]
        base_color = age_colors.get(age, "black")
        color = adjust_lightness(base_color, parcellation_lightness.get(parcellation, 1.0))
        marker = run_markers.get(run, "o")
        linestyle = "-" if run == 1 else "--" if run == 2 else ":"

        ax.plot(
            cluster_numbers,
            item["values"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=f"{label} - {metric_name}",
            alpha=0.9,
        )

        if annotate:
            for x, y in zip(cluster_numbers, item["values"]):
                ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=12)

    ax.set_xticks(cluster_numbers)
    ax.set_xticklabels([f"Cluster {n}" for n in cluster_numbers], rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"cluster_numbers": cluster_numbers, "series": series, "fig": fig, "ax": ax}


def parse_filename(path):
    """Parse age group, parcellation, and run number from a JSON filename."""
    pattern = re.compile(r"(?P<parcellation>\d+x\d+)_(?P<age_group>young|old)_(?P<run>run\d+)", re.IGNORECASE)   
    match = pattern.search(Path(path).name)
    # 400x400_Old_run{run}_cluster_behavioural_results.json
    
    if match.group("run") == "run1":
        movie = "neutral"
        run = 1
    elif match.group("run") == "run2":
        movie = "negative"
        run = 2
    if match:
        return {
            "age_group": match.group("age_group").lower(),
            "parcellation": match.group("parcellation"),
            "run": run,
            "movie": movie,
        }
    else:
        print(f"Warning: Filename does not match expected pattern: {Path(path).name}")
        return None


def plot_cca_mlr_multiple_runs(
    json_paths,
    labels=None,
    cca_metric="cca_removed_subjects_results:cv_mean_cc",
    mlr_metric="mlr_removed_subjects_results:mean_accuracy",
    dead_value=0.0,
    title="CCA and MLR comparison across runs",
    save_path=None,
    figsize=(11, 6),
    annotate=False,
):
    json_paths = [Path(p) for p in json_paths]

    if labels is None:
        labels = [p.stem.replace("_cluster_behavioural_results", "") for p in json_paths]

    if len(labels) != len(json_paths):
        raise ValueError("labels must have the same length as json_paths.")

    age_colors = {
        "Young": "green",
        "Old": "red",
    }

    parcellation_styles = {
        "400x400": "-",
        "1000x1000": "--",
    }

    score_markers = {
        "CCA": "o",
        "MLR": "s",
    }

    all_results = [load_json_results(p) for p in json_paths]

    cluster_numbers = sorted({
        int(cluster_name.split("_")[1])
        for results in all_results
        for cluster_name in results.keys()
        if cluster_name.startswith("Cluster_")
    })

    fig, ax = plt.subplots(figsize=figsize)

    for path, results, label in zip(json_paths, all_results, labels):
        metadata = parse_filename(path)

        if metadata is None:
            age_group = "Unknown"
            parcellation = "Unknown"
        else:
            age_group = metadata["age_group"].capitalize()
            parcellation = metadata["parcellation"]

        color = age_colors.get(age_group, "black")
        linestyle = parcellation_styles.get(parcellation, ":")

        cca_values = []
        mlr_values = []

        for cluster_num in cluster_numbers:
            cluster_key = f"Cluster_{cluster_num}_results"
            cluster_data = results.get(cluster_key, {})

            cca_val = get_nested_value(cluster_data, cca_metric)
            mlr_val = get_nested_value(cluster_data, mlr_metric)

            cca_values.append(dead_value if cca_val is None else cca_val)
            mlr_values.append(dead_value if mlr_val is None else mlr_val)

        ax.plot(
            cluster_numbers,
            cca_values,
            color=color,
            linestyle=linestyle,
            marker=score_markers["CCA"],
            label=f"{label} - CCA",
        )

        ax.plot(
            cluster_numbers,
            mlr_values,
            color=color,
            linestyle=linestyle,
            marker=score_markers["MLR"],
            label=f"{label} - MLR",
            alpha=0.75,
        )

        if annotate:
            for x, y in zip(cluster_numbers, cca_values):
                ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )

            for x, y in zip(cluster_numbers, mlr_values):
                ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Score")
    ax.set_xticks(cluster_numbers)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.text(
        0.5,
        -0.02,
        "Colour = age group | Line style = parcellation | Marker = score type",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "cluster_numbers": cluster_numbers,
    }
    

def plot_silhouette_age_comparison(
    run_results: Sequence[Dict[str, Any]],
    save_path: Optional[PathLike] = None,
    figsize: Tuple[int, int] = (6, 4),
    cluster_range: Iterable[int] = range(2, 11),
    title: str = "Silhouette score comparison between age groups",
    annotate: bool = False,
) -> Dict[str, Any]:
    """
    Compare Silhouette scores between age groups.

    Each run is one line. Runs from the same age group use the same colour.

    run_results example
    -------------------
    [
        {"label": "Old 400 run1", "age_group": "Old", "results": old_400_run1},
        {"label": "Young 400 run1", "age_group": "Young", "results": young_400_run1},
    ]
    """
    age_colors = {"young": "green", "y": "green", "old": "red", "o": "red"}
    cluster_numbers = list(cluster_range)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, run in enumerate(run_results):
        label = run["label"]
        age_group = str(run["age_group"]).lower()
        results = run["results"]
        values = [results.get(k, {}).get("silhouette", np.nan) for k in cluster_numbers]
        color = age_colors.get(age_group, "black")
        linestyle = "-" if idx % 2 == 0 else "--"
        marker = "o" if idx % 3 == 0 else "s" if idx % 3 == 1 else "^"

        ax.plot(cluster_numbers, values, marker=marker, linestyle=linestyle, color=color, label=label, alpha=0.85)

        if annotate:
            for x, y in zip(cluster_numbers, values):
                if np.isfinite(y):
                    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=12)

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(cluster_numbers)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.text(0.5, -0.02, "Silhouette score ∈ [-1, 1]. Higher values indicate better-defined clusters.", ha="center", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"fig": fig, "ax": ax, "cluster_numbers": cluster_numbers}




def plot_silhouette_age_comparison_all_runs(
    models_dir: PathLike = "Clusters",
    model_prefix: Optional[str] = None,
    labels_tag: str = "labels_predicted_labels_",
    middle_layer_tag: str = "middle_layer_predicted_labels_",
    cluster_range: Iterable[int] = range(2, 11),
    save_path: Optional[PathLike] = None,
    figsize: Tuple[int, int] = (6, 4),
    title: str = "Silhouette score comparison across runs",
    annotate: bool = False,
    include_invalid: bool = True,
    invalid_value: float = np.nan,
    highlight_by: str = "age",
) -> Dict[str, Any]:
    """
    Automatically collect all completed DCEC runs in a folder and compare
    Silhouette scores across cluster numbers.

    A "run" is inferred from filenames matching the pattern used by the DCEC
    outputs, for example:
        DCEC_400x400_68_O_subjects_run1_...labels_predicted_labels_...
        DCEC_400x400_71_Y_subjects_run1_...labels_predicted_labels_...

    The function plots one line per discovered run.

    Parameters
    ----------
    models_dir:
        Folder containing label files and matching middle-layer files.
    model_prefix:
        Optional prefix filter. Use this if you only want to include a subset of
        runs, e.g. "DCEC_400x400".
    labels_tag:
        String identifying label files.
    middle_layer_tag:
        String identifying middle-layer / latent-representation files.
    cluster_range:
        Cluster numbers to include, default K=2..10.
    save_path:
        If provided, save the figure to this path.
    figsize:
        Figure size.
    title:
        Plot title.
    annotate:
        If True, annotate each point with its Silhouette value.
    include_invalid:
        If True, invalid/collapsed clusterings are plotted as invalid_value.
        If False, invalid/collapsed clusterings are shown as gaps.
    invalid_value:
        Value used for invalid/collapsed clusterings. Default is np.nan.
    highlight_by:
        Controls what the colour encodes:
            - "age": colour = age group, marker = parcellation, line style = movie
            - "parcellation": colour = parcellation, marker = age group, line style = movie

    Returns
    -------
    dict
        Dictionary containing the discovered runs, the figure, and the axes.
    """
    highlight_by = highlight_by.lower().strip()
    if highlight_by not in {"age", "parcellation"}:
        raise ValueError("highlight_by must be either 'age' or 'parcellation'.")

    models_dir = Path(models_dir)
    cluster_numbers = list(cluster_range)
    cluster_set = set(cluster_numbers)

    # ------------------------------------------------------------------
    # Collect scores grouped by run metadata
    # ------------------------------------------------------------------
    runs: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}

    for labels_file in sorted(models_dir.glob("*.txt")):
        filename = labels_file.name

        if labels_tag not in filename:
            continue
        if model_prefix is not None and not filename.startswith(model_prefix):
            continue

        metadata = parse_model_filename(filename)
        if metadata.get("age_group") == "unknown":
            print(f"Skipping {filename}: could not parse run metadata.")
            continue

        cluster_number = extract_cluster_number(filename)
        if cluster_number is None:
            print(f"Skipping {filename}: could not extract cluster number.")
            continue
        if cluster_number not in cluster_set:
            continue

        middle_layer_file = Path(str(labels_file).replace(labels_tag, middle_layer_tag))
        if not middle_layer_file.exists():
            print(f"Skipping {filename}: missing middle-layer file.")
            continue

        try:
            labels = np.loadtxt(labels_file)
            embedding = np.loadtxt(middle_layer_file)
            scores = compute_clustering_scores(embedding, labels, invalid_value=invalid_value)
        except Exception as error:
            print(f"Skipping {filename}: {error}")
            continue

        run_key = (
            metadata.get("age_label", "unknown"),
            metadata.get("parcellation", "unknown"),
            metadata.get("movie", "unknown"),
            metadata.get("run", "unknown"),
            metadata.get("n_subjects", "unknown"),
        )

        if run_key not in runs:
            age_label, parcellation, movie, run, n_subjects = run_key
            runs[run_key] = {
                "label": f"{age_label} {parcellation} {movie} (N={n_subjects})",
                "age_group": age_label,
                "parcellation": parcellation,
                "movie": movie,
                "run": run,
                "n_subjects": n_subjects,
                "results": {},
            }

        runs[run_key]["results"][cluster_number] = {
            **scores,
            "file": filename,
            "labels_file": str(labels_file),
            "middle_layer_file": str(middle_layer_file),
            "metadata": metadata,
        }

    if not runs:
        print("No matching runs found.")
        return {"runs": [], "fig": None, "ax": None}

    # Sort order: age, parcellation, run/movie.
    run_items = sorted(
        runs.values(),
        key=lambda r: (
            str(r["age_group"]),
            str(r["parcellation"]),
            str(r["run"]),
            str(r["n_subjects"]),
        ),
    )

    # ------------------------------------------------------------------
    # Plot styles
    # ------------------------------------------------------------------
    age_colors = {
        "Young": "green",
        "young": "green",
        "Y": "green",
        "Old": "red",
        "old": "red",
        "O": "red",
    }
    parcellation_colors = {
        "400x400": "tab:blue",
        "1000x1000": "tab:orange",
    }

    # Same line style for the same movie condition.
    movie_linestyles = {
        "neutral": "-",
        "negative": "--",
    }

    # Secondary encodings used to keep lines distinguishable.
    parcellation_markers = {
        "400x400": "o",
        "1000x1000": "s",
    }
    age_markers = {
        "Young": "o",
        "young": "o",
        "Y": "o",
        "Old": "s",
        "old": "s",
        "O": "s",
    }

    if highlight_by == "age":
        style_note = "Colour = age group, marker = parcellation, line style = movie condition."
    else:
        style_note = "Colour = parcellation, marker = age group, line style = movie condition."

    fig, ax = plt.subplots(figsize=figsize)

    for run in run_items:
        values = []
        for k in cluster_numbers:
            if k not in run["results"]:
                values.append(np.nan)
                continue

            value = run["results"][k].get("silhouette", np.nan)
            if not include_invalid and not run["results"][k].get("valid", True):
                value = np.nan
            values.append(value)

        if highlight_by == "age":
            color = age_colors.get(run["age_group"], "black")
            marker = parcellation_markers.get(run["parcellation"], "^")
        else:
            color = parcellation_colors.get(run["parcellation"], "black")
            marker = age_markers.get(run["age_group"], "^")

        linestyle = movie_linestyles.get(run["movie"], "-.")

        ax.plot(
            cluster_numbers,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=run["label"],
            alpha=0.85,
        )

        if annotate:
            for x, y in zip(cluster_numbers, values):
                if np.isfinite(y):
                    ax.annotate(
                        f"{y:.2f}",
                        (x, y),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=12,
                    )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(cluster_numbers)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.text(
        0.5,
        -0.02,
        "Silhouette score ∈ [-1, 1] ↑ | " + style_note,
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return {
        "runs": run_items,
        "cluster_numbers": cluster_numbers,
        "highlight_by": highlight_by,
        "fig": fig,
        "ax": ax,
    }


def save_internal_clustering_metrics_json(
    models_dir: PathLike = "Clusters",
    output_json_path: PathLike = "internal_clustering_scores.json",
    labels_tag: str = "labels_predicted_labels_",
    middle_layer_tag: str = "middle_layer_predicted_labels_",
    cluster_range: Iterable[int] = range(2, 11),
    invalid_value: float = np.nan,
    overwrite: bool = True,
    print_results: bool = False,
) -> Dict[str, Any]:
    """
    Calculate and save all internal clustering metrics for every run condition.

    This function uses the existing collect_clustering_scores_from_folder()
    function. It does not recompute scores manually and it does not plot.

    The output JSON contains one entry per run condition, where each run contains
    scores for K = 2, ..., 10 by default.

    Metrics saved:
        - silhouette
        - davies_bouldin
        - calinski_harabasz
    """

    models_dir = Path(models_dir)
    output_json_path = Path(output_json_path)

    if output_json_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_json_path}")

    def safe_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return value

    def safe_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def discover_model_prefixes() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Find one model prefix per run condition.

        Example prefix:
            DCEC_400x400_68_O_subjects_run1

        This prefix is then passed to collect_clustering_scores_from_folder().
        """
        prefixes = {}
        skipped_files = []

        for labels_file in sorted(models_dir.glob("*.txt")):
            filename = labels_file.name

            if labels_tag not in filename:
                continue

            metadata = parse_model_filename(filename)

            if metadata.get("age_group") == "unknown":
                skipped_files.append(filename)
                continue

            model_prefix = (
                f"DCEC_{metadata['parcellation']}_"
                f"{metadata['n_subjects']}_"
                f"{metadata['age_group']}_subjects_"
                f"{metadata['run']}"
            )

            prefixes[model_prefix] = metadata

        return prefixes, skipped_files

    model_prefixes, skipped_parse_files = discover_model_prefixes()

    if not model_prefixes:
        raise ValueError(
            "No model prefixes found. Check that the files follow the expected "
            "DCEC_400x400_68_O_subjects_run1_... naming pattern."
        )

    runs = []
    skipped_runs = []

    for model_prefix, prefix_metadata in sorted(model_prefixes.items()):
        scores_by_cluster = collect_clustering_scores_from_folder(
            models_dir=models_dir,
            model_prefix=model_prefix,
            labels_tag=labels_tag,
            middle_layer_tag=middle_layer_tag,
            cluster_range=cluster_range,
            invalid_value=invalid_value,
            print_results=print_results,
        )

        if not scores_by_cluster:
            skipped_runs.append(
                {
                    "model_prefix": model_prefix,
                    "reason": "No valid clustering scores found for this prefix.",
                }
            )
            continue

        # Prefer metadata from the actual collected result if available.
        first_result = next(iter(scores_by_cluster.values()))
        metadata = first_result.get("metadata", prefix_metadata)

        age_label = metadata.get("age_label", "unknown")
        age_group = metadata.get("age_group", "unknown")
        parcellation = metadata.get("parcellation", "unknown")
        movie = metadata.get("movie", "unknown")
        run = metadata.get("run", "unknown")
        n_subjects = metadata.get("n_subjects", "unknown")

        condition_id = f"{parcellation}_{age_label}_{movie}"

        scores = {}

        for cluster_number, result in sorted(scores_by_cluster.items()):
            scores[str(cluster_number)] = {
                "cluster": safe_int(cluster_number),
                "silhouette": safe_float(result.get("silhouette")),
                "davies_bouldin": safe_float(result.get("davies_bouldin")),
                "calinski_harabasz": safe_float(result.get("calinski_harabasz")),
                "valid": bool(result.get("valid", True)),
                "file": result.get("file"),
                "labels_file": result.get("labels_file"),
                "middle_layer_file": result.get("middle_layer_file"),
            }

        runs.append(
            {
                "condition_id": condition_id,
                "label": f"{age_label} {parcellation} {movie} (N={n_subjects})",
                "model_prefix": model_prefix,
                "age_label": age_label,
                "age_group": age_group,
                "parcellation": parcellation,
                "movie": movie,
                "run": run,
                "n_subjects": safe_int(n_subjects),
                "scores": scores,
            }
        )

    runs = sorted(
        runs,
        key=lambda r: (
            str(r["age_label"]),
            str(r["parcellation"]),
            str(r["run"]),
            str(r["movie"]),
        ),
    )

    cluster_numbers = sorted(
        {
            int(k)
            for run in runs
            for k in run["scores"].keys()
        }
    )

    output = {
        "metadata": {
            "source_function": "collect_clustering_scores_from_folder",
            "models_dir": str(models_dir),
            "labels_tag": labels_tag,
            "middle_layer_tag": middle_layer_tag,
            "cluster_range": list(cluster_range),
            "metrics": [
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
            ],
            "n_run_conditions": len(runs),
            "n_cluster_solutions": sum(len(run["scores"]) for run in runs),
            "cluster_numbers": cluster_numbers,
        },
        "runs": runs,
        "skipped": {
            "unparsed_label_files": skipped_parse_files,
            "runs_without_scores": skipped_runs,
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Saved internal clustering metrics to: {output_json_path}")
    print(f"Run conditions saved: {len(runs)}")
    print(f"Cluster solutions saved: {output['metadata']['n_cluster_solutions']}")

    return output


def plot_internal_clustering_metric_from_json(
    json_path: PathLike,
    metric: str = "silhouette",
    save_path: Optional[PathLike] = None,
    figsize: Tuple[int, int] = (7, 4),
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    annotate: bool = False,
    offset_points: bool = True,
    calinski_scale: Optional[float] = 1000.0,
) -> Dict[str, Any]:
    """
    Plot one internal clustering metric for every run condition stored in the
    JSON file created by save_internal_clustering_metrics_json().

    Parameters
    ----------
    json_path:
        Path to the saved internal clustering metrics JSON.

    metric:
        Which metric to plot. Accepted values:
            - "silhouette"
            - "silhouette_score"
            - "davies_bouldin"
            - "davies_bouldin_score"
            - "calinski_harabasz"
            - "calinski_harabasz_score"

    calinski_scale:
        Optional scaling for Calinski-Harabasz. For example, use 1000.0 to plot
        Calinski-Harabasz / 1000. If None, the raw value is plotted.

    Style
    -----
    Similar to the CCA/MLR comparison style:
        - colour = age group
        - line style = parcellation
        - marker = movie condition
    """

    json_path = Path(json_path)
    data = load_json_results(json_path)

    metric_aliases = {
        "silhouette": "silhouette",
        "silhouette_score": "silhouette",
        "davies_bouldin": "davies_bouldin",
        "davies_bouldin_score": "davies_bouldin",
        "davies-bouldin": "davies_bouldin",
        "calinski_harabasz": "calinski_harabasz",
        "calinski_harabasz_score": "calinski_harabasz",
        "calinski-harabasz": "calinski_harabasz",
    }

    metric_key = metric_aliases.get(metric.lower())

    if metric_key is None:
        raise ValueError(
            "Unknown metric. Use one of: "
            "'silhouette', 'davies_bouldin', or 'calinski_harabasz'."
        )

    metric_labels = {
        "silhouette": {
            "ylabel": "Silhouette score",
            "title": "Silhouette score across run conditions",
            "footnote": "Silhouette score ∈ [-1, 1]. Higher is better.",
        },
        "davies_bouldin": {
            "ylabel": "Davies-Bouldin score",
            "title": "Davies-Bouldin score across run conditions",
            "footnote": "Davies-Bouldin score ∈ [0, ∞). Lower is better.",
        },
        "calinski_harabasz": {
            "ylabel": "Calinski-Harabasz score",
            "title": "Calinski-Harabasz score across run conditions",
            "footnote": "Calinski-Harabasz score ∈ [0, ∞). Higher is better.",
        },
    }

    runs = data.get("runs", [])
    # Sort the runs by parcellation(400x400 before 1000x1000), then by age group (Old before Young), then by run.
    def run_sort_key(r):
        parcellation_order = {
            "400x400": 0,
            "1000x1000": 1,
        }

        age_order = {
            "Old": 0,
            "old": 0,
            "O": 0,
            "Young": 1,
            "young": 1,
            "Y": 1,
        }

        parcellation = str(r.get("parcellation", "unknown"))
        age = str(r.get("age_label", r.get("age_group", "unknown")))
        run = str(r.get("run", "unknown"))

        # Extract run number, so run10 does not sort before run2
        run_match = re.search(r"\d+", run)
        run_number = int(run_match.group()) if run_match else 999

        return (
            parcellation_order.get(parcellation, 999),
            age_order.get(age, 999),
            run_number,
        )
    runs = sorted(runs, key=run_sort_key)
    
    
    if not runs:
        raise ValueError(f"No runs found in {json_path}")

    cluster_numbers = sorted(
        {
            int(cluster)
            for run in runs
            for cluster in run.get("scores", {}).keys()
        }
    )

    if not cluster_numbers:
        raise ValueError(f"No cluster scores found in {json_path}")

    age_colors = {
        "Young": "green",
        "young": "green",
        "Y": "green",
        "Old": "red",
        "old": "red",
        "O": "red",
    }

    age_linestyles = {
        "Young": "-",
        "young": "-",
        "Y": "-",
        "Old": "--",
        "old": "--",
        "O": "--"
    }

    movie_markers = {
        "neutral": "o",
        "negative": "s",
        "run1": "o",
        "run2": "s",
    }
    
    if offset_points and len(runs) > 1:
        offsets = np.linspace(-0.22, 0.22, len(runs))
    else:
        offsets = np.zeros(len(runs))

    fig, ax = plt.subplots(figsize=figsize)

    for idx, run in enumerate(runs):
        scores = run.get("scores", {})

        y_values = []

        for cluster_number in cluster_numbers:
            cluster_scores = scores.get(str(cluster_number), {})
            value = cluster_scores.get(metric_key)

            if value is None:
                y_values.append(np.nan)
            else:
                value = float(value)

                if metric_key == "calinski_harabasz" and calinski_scale is not None:
                    value = value / calinski_scale

                y_values.append(value)

        x_values = np.asarray(cluster_numbers, dtype=float) + offsets[idx]

        age_label = run.get("age_label", run.get("age_group", "unknown"))
        parcellation = run.get("parcellation", "unknown")
        movie = run.get("movie", "unknown")

        #color = age_colors.get(age_label, "black")

        linestyle = age_linestyles.get(age_label, ":")
        marker = movie_markers.get(movie, "^")

        label = run.get(
            "label",
            f"{age_label} {parcellation} {movie}",
        )

        ax.plot(
            x_values,
            y_values,
            # color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.4,
            markersize=5,
            alpha=0.85,
            label=label,
        )

        if annotate:
            for x, y in zip(x_values, y_values):
                if np.isfinite(y):
                    ax.annotate(
                        f"{y:.3f}",
                        (x, y),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )

    ylabel = metric_labels[metric_key]["ylabel"]
    plot_title = title or metric_labels[metric_key]["title"]
    footnote = metric_labels[metric_key]["footnote"]

    if metric_key == "calinski_harabasz" and calinski_scale is not None:
        ylabel = f"{ylabel} / {calinski_scale:g}"
        footnote += f" Values are divided by {calinski_scale:g} in this plot."

    ax.set_xlabel("Number of clusters $K$")
    ax.set_ylabel(ylabel)
    ax.set_title(plot_title)
    ax.set_xticks(cluster_numbers)
    ax.set_xticklabels([str(k) for k in cluster_numbers])
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(
        fontsize=7,
        title="Run condition",
        title_fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.text(
        0.5,
        -0.03,
        footnote + "Line style = age group | Marker = movie.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "metric": metric_key,
        "cluster_numbers": cluster_numbers,
        "runs": runs,
    }


def plot_silhouette_parcellation_comparison_all_runs(
    models_dir: PathLike = "Clusters",
    model_prefix: Optional[str] = None,
    labels_tag: str = "labels_predicted_labels_",
    middle_layer_tag: str = "middle_layer_predicted_labels_",
    cluster_range: Iterable[int] = range(2, 11),
    save_path: Optional[PathLike] = None,
    figsize: Tuple[int, int] = (12, 7),
    title: str = "Silhouette score comparison between parcellations across runs",
    annotate: bool = False,
    include_invalid: bool = True,
    invalid_value: float = np.nan,
) -> Dict[str, Any]:
    """
    Convenience wrapper for plot_silhouette_age_comparison_all_runs() where
    colour highlights parcellation instead of age group.

    Plot encoding:
        - Colour = parcellation
        - Marker = age group
        - Line style = movie condition
    """
    return plot_silhouette_age_comparison_all_runs(
        models_dir=models_dir,
        model_prefix=model_prefix,
        labels_tag=labels_tag,
        middle_layer_tag=middle_layer_tag,
        cluster_range=cluster_range,
        save_path=save_path,
        figsize=figsize,
        title=title,
        annotate=annotate,
        include_invalid=include_invalid,
        invalid_value=invalid_value,
        highlight_by="parcellation",
    )
    

# =============================================================================
# Frequency summaries for selected variables from behavioural JSON files
# =============================================================================


def is_valid_cluster(cluster_result: Dict[str, Any]) -> bool:
    """A cluster is valid for variable counting if CCA or MLR selected variables exist."""
    return isinstance(cluster_result.get("cca_selected_variables"), list) or isinstance(
        cluster_result.get("mlr_selected_variables"), list
    )


def selected_variable_union(cluster_result: Dict[str, Any]) -> set:
    """Union CCA and MLR selected variables. Each variable counts once per cluster."""
    return _combine_keys_as_set(cluster_result, ("cca_selected_variables", "mlr_selected_variables"))

def removed_subjects_union(cluster_result: Dict[str, Any]) -> set:
    """Union CCA and MLR removed subjects. Each subject counts once per cluster."""
    return _combine_keys_as_set(cluster_result, ("cca_removed_subjects", "mlr_removed_subjects"))


def collect_cluster_rows(json_paths: Sequence[PathLike]) -> List[Dict[str, Any]]:
    """
    Convert behavioural JSON files into row dictionaries.

    Files whose names do not match the expected pattern are skipped.
    Each row corresponds to one valid cluster solution.
    """
    rows: List[Dict[str, Any]] = []

    for path in json_paths:
        path = Path(path)
        metadata = parse_behaviour_json_filename(path)
        if metadata is None:
            print(f"Skipping non-matching JSON filename: {path.name}")
            continue

        data = load_json_results(path)
        for cluster_key, cluster_result in data.items():
            if not cluster_key.startswith("Cluster_"):
                continue
            if not is_valid_cluster(cluster_result):
                continue

            rows.append(
                {
                    **metadata,
                    "cluster": cluster_sort_key(cluster_key),
                    "variables": selected_variable_union(cluster_result),
                    "removed_subjects": removed_subjects_union(cluster_result),
                }
            )

    return rows


def most_common_with_ties(counter: Counter) -> Tuple[List[Any], int]:
    """Return most frequent item(s), allowing ties."""
    if not counter:
        return [], 0
    max_count = max(counter.values())
    return sorted(item for item, count in counter.items() if count == max_count), max_count


def second_most_common_with_ties(counter: Counter) -> Tuple[List[Any], int]:
    """Return second most frequent item(s), allowing ties."""
    if not counter:
        return [], 0
    counts = sorted(set(counter.values()), reverse=True)
    if len(counts) < 2:
        return [], 0
    second_count = counts[1]
    return sorted(item for item, count in counter.items() if count == second_count), second_count


def third_most_common_with_ties(counter: Counter) -> Tuple[List[Any], int]:
    """Return third most frequent item(s), allowing ties."""
    if not counter:
        return [], 0
    counts = sorted(set(counter.values()), reverse=True)
    if len(counts) < 3:
        return [], 0
    third_count = counts[2]
    return sorted(item for item, count in counter.items() if count == third_count), third_count


def summarize_grouped(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    include_second: bool = True,
    include_third: bool = False,
) -> pd.DataFrame:
    """
    Summarise most frequent selected variables grouped by metadata.

    Examples
    --------
    summarize_grouped(rows, ["age_group"])
    summarize_grouped(rows, ["movie"])
    summarize_grouped(rows, ["cluster"])
    summarize_grouped(rows, ["parcellation", "age_group", "movie"])
    """
    grouped_counters: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
    grouped_valid_counts: Counter = Counter()

    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped_valid_counts[group] += 1
        for variable in row["variables"]:
            grouped_counters[group][variable] += 1

    output_rows = []
    for group, counter in sorted(grouped_counters.items()):
        top_vars, top_count = most_common_with_ties(counter)
        result = {
            **{key: value for key, value in zip(group_keys, group)},
            "valid_cluster_solutions": grouped_valid_counts[group],
            "most_frequent_variable": "; ".join(map(str, top_vars)),
            "occurrences": top_count,
        }

        if include_second:
            second_vars, second_count = second_most_common_with_ties(counter)
            result["second_most_frequent_variable"] = "; ".join(map(str, second_vars))
            result["second_occurrences"] = second_count

        if include_third:
            third_vars, third_count = third_most_common_with_ties(Counter({var: cnt for var, cnt in counter.items() if cnt < top_count and cnt < second_count}))
            result["third_most_frequent_variable"] = "; ".join(map(str, third_vars))
            result["third_occurrences"] = third_count
        output_rows.append(result)

    return pd.DataFrame(output_rows)


def summarize_grouped_subjects(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    include_second: bool = True,
    include_third: bool = False,
) -> pd.DataFrame:
    """
    Summarise most frequent selected variables grouped by metadata.

    Examples
    --------
    summarize_grouped(rows, ["age_group"])
    summarize_grouped(rows, ["movie"])
    summarize_grouped(rows, ["cluster"])
    summarize_grouped(rows, ["parcellation", "age_group", "movie"])
    """
    grouped_counters: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
    grouped_valid_counts: Counter = Counter()
    clusters_with_removed: Counter = Counter()

    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped_valid_counts[group] += 1
        if not row["removed_subjects"]:
            continue
        else:
            clusters_with_removed[group] += 1
            for subject in row["removed_subjects"]:
                grouped_counters[group][subject] += 1

    output_rows = []
    for group, counter in sorted(grouped_counters.items()):
        top_vars, top_count = most_common_with_ties(counter)
        result = {
            **{key: value for key, value in zip(group_keys, group)},
            "valid_cluster_solutions": grouped_valid_counts[group],
            "clusters_with_removed": clusters_with_removed[group],
            "most_frequent_subject": "; ".join(map(str, top_vars)),
            "occurrences": top_count,
        }

        if include_second:
            second_vars, second_count = second_most_common_with_ties(counter)
            result["second_most_frequent_subject"] = "; ".join(map(str, second_vars))
            result["second_occurrences"] = second_count

        if include_third:
            third_vars, third_count = third_most_common_with_ties(Counter({var: cnt for var, cnt in counter.items() if cnt < top_count and cnt < second_count}))
            result["third_most_frequent_subject"] = "; ".join(map(str, third_vars))
            result["third_occurrences"] = third_count
        output_rows.append(result)

    return pd.DataFrame(output_rows)


# =============================================================================
# Miscellaneous utility
# =============================================================================


def compute_my_accuracy(true_labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    """Compute clustering accuracy after optimal label matching."""
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-confusion_mat)
    return float(confusion_mat[row_ind, col_ind].sum() / np.sum(confusion_mat))
