from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
import os
from scipy.io import loadmat
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def evaluate_clustering(functional_connectivity_matrix, labels_path):
    for file in os.listdir(labels_path):
        filename = os.fsdecode(file)
        
        if filename.endswith(".txt"):
            labels_file = os.path.join(labels_path, file)
        else:
            print("No labels file found.")
            break
        
    
        labels = np.loadtxt(os.fsdecode(labels_file))
        if len(np.unique(labels)) > 1:  # Ensure more than one cluster exists
            print(f"\n Scores for {filename}:")
            
            # Print the silhouette coefficient for the current labels
            silhouette_avg = silhouette_score(functional_connectivity_matrix, labels)
            print(f"Silhouette coefficient: {silhouette_avg}")
            
            # Print the Davies-Bouldin scores
            davies_bouldin_avg = davies_bouldin_score(functional_connectivity_matrix, labels)
            print(f"Davies-Bouldin score: {davies_bouldin_avg}")
            
            # Print the Calinski-Harabasz scores
            calinski_harabasz_avg = calinski_harabasz_score(functional_connectivity_matrix, labels)
            print(f"Calinski-Harabasz score: {calinski_harabasz_avg}")
        else:
            print(f"Only one cluster in {filename}, silhouette coefficient not computed.")

def evaluate_single_clustering(functional_connectivity_matrix, labels_path, print_results=True):
    """
    Calculates clustering evaluation metrics (silhouette coefficient, Davies-Bouldin score, and Calinski-Harabasz score) for a given set of labels and a functional connectivity matrix. The function reads the labels from a specified path, computes the metrics, and prints the results. 

    Args:
        functional_connectivity_matrix (_type_): The functional connectivity matrix representing the data points to be evaluated.
        labels_path (_type_): The path to the file containing the predicted labels for the clustering results. This can be a string representing the file path or an array of labels directly.

    Returns:
        tuple: A tuple containing the computed silhouette coefficient, Davies-Bouldin score, and Calinski-Harabasz score.
    """
    if type(labels_path) is str:
        labels = np.loadtxt(labels_path)
    else:
        labels = labels_path
    
    if len(np.unique(labels)) > 1:  # Ensure more than one cluster exists
        # Calculate the silhouette coefficient, Davies-Bouldin score, and Calinski-Harabasz score
        silhouette_avg = silhouette_score(functional_connectivity_matrix, labels)
        davies_bouldin_avg = davies_bouldin_score(functional_connectivity_matrix, labels)
        calinski_harabasz_avg = calinski_harabasz_score(functional_connectivity_matrix, labels)
    else: # Only one cluster exists, metrics cannot be computed
        print("Only one cluster in the provided labels, silhouette coefficient not computed.")  
        silhouette_avg = 0
        davies_bouldin_avg = 0
        calinski_harabasz_avg = 0
    
    # Print the results if requested
    if print_results:
        print(f"\n Scores for the given labels:")
        print(f"Silhouette coefficient: {silhouette_avg}")
        print(f"Davies-Bouldin score: {davies_bouldin_avg}")
        print(f"Calinski-Harabasz score: {calinski_harabasz_avg}")
        
    return silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg
        
        
def compute_my_accuracy(true_labels, predicted_labels):
    """ 
    Compute the accuracy of clustering results by finding the best matching between true and predicted labels.
    Parameters:
    - true_labels: The ground truth labels for the data points.
    - predicted_labels: The labels assigned by the clustering algorithm.
    Returns:
    - acc: The computed accuracy of the clustering results.
    """
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-confusion_mat)  # Maximize
    
    acc = confusion_mat[row_ind, col_ind].sum() / np.sum(confusion_mat)
    return acc
        
# Example
# PTSD = loadmat("C:\\Users\\oddar\\Downloads\\PTSD_connectivity.mat")
# PTSD is a dataset containing 87 samples (subjects) with 340 features (as vectorized functional connectivity matrices)
# The expected number of clusters are 3

# Define the functional connectivity matrix (example)
# Functional_connectivity_matrix = PTSD["connectivities"]  # Example matrix

# Define where to find the labels 
# labels_path = os.fsencode("Clusters")
# evaluate_clustering(Functional_connectivity_matrix, labels_path)


def Calculate_clustering_scores_from_folder(
    models_dir = "Clusters",
    model_prefix = None,
    labels_tag = "labels_predicted_labels_",
    middle_layer_tag = "middle_layer_predicted_labels_",
    print_results = False
    ):
    """
        This function iterates through a specified directory to find clustering result files, extracts the predicted labels and corresponding middle-layer embeddings, and computes clustering evaluation metrics (silhouette coefficient, Davies-Bouldin score, and Calinski-Harabasz score) for each model. The results are stored in a dictionary for further analysis or visualization.
    
    Args:
        models_dir (str, optional): The directory containing the clustering result files. Defaults to "Clusters".
        model_prefix (_type_, optional): The prefix of the model files to process. Defaults to None.
        labels_tag (str, optional): The tag used to identify the labels file. Defaults to "labels_predicted_labels_".
        middle_layer_tag (str, optional): The tag used to identify the middle-layer file. Defaults to "middle_layer_predicted_labels_".

    Returns:
        dict: A dictionary containing the computed silhouette coefficient, Davies-Bouldin score, and Calinski-Harabasz score for each valid model found in the specified directory.
    """
    results = {}
    
    for file in os.listdir(models_dir):
        if file.startswith(model_prefix) and file.endswith(".txt") and labels_tag in file:
            labels_file = os.path.join(models_dir, file)
            middle_layer_file = labels_file.replace(labels_tag, middle_layer_tag)
            
            if not os.path.exists(middle_layer_file):
                print(f"Skipping {file}: missing middle-layer file.")
                continue

            try:
                labels = np.loadtxt(labels_file)
                z = np.loadtxt(middle_layer_file)
                                
                silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg = evaluate_single_clustering(z, labels, print_results=print_results)
                results[file] = {
                    "silhouette": silhouette_avg,
                    "davies_bouldin": davies_bouldin_avg,
                    "calinski_harabasz": calinski_harabasz_avg
                }
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    if len(results) == 0:
        print("No valid models found for silhouette plotting.")
        return results
    
    return results

def plot_scores(
    results,
    save_path=None,
    sort_scores=True,
    annotate=True
    ):
    
    if len(results) == 0:
        print("No valid models found for silhouette plotting.")
        return results
    
    model_names = list(results.keys())
    # Extract the cluster number for better readability [eg. Cluster_3]
    cluster_names = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    print(f"Model names: {model_names}")
    print(f"Size of model names: {len(model_names)}")
    for name in model_names:
        cluster_names.append("Cluster_" + name.split("_cluster_")[1].split("_")[0])
        silhouette_scores.append(results[name]["silhouette"])
        davies_bouldin_scores.append(results[name]["davies_bouldin"])
        calinski_harabasz_scores.append(results[name]["calinski_harabasz"])

    if sort_scores:        
        cluster_numbers = [int(cluster_names[i].split("Cluster_")[1]) for i in range(len(cluster_names))]
        sorted_indices = np.argsort(cluster_numbers)   # Sort by cluster number
        print(f"Sorted indices: {sorted_indices}")
        # Reorder all lists according to the sorted cluster numbers
        model_names = [model_names[i] for i in sorted_indices]
        cluster_names = [cluster_names[i] for i in sorted_indices]
        silhouette_scores = [silhouette_scores[i] for i in sorted_indices]
        davies_bouldin_scores = [davies_bouldin_scores[i] for i in sorted_indices]
        calinski_harabasz_scores = [calinski_harabasz_scores[i] for i in sorted_indices]

    # Plot one score at a time
    # Silhouette Score
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, silhouette_scores, color='skyblue', label='Silhouette Score')
    ax.set_xlabel('Model')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Scores for Different Models')
    ax.set_xticklabels(cluster_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')
    if annotate:
        for bar, score in zip(bars, silhouette_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom"
            )
    plt.tight_layout()
    fig.text(0.5, 0.01, '[-1, 1] Higher silhouette scores indicate better-defined clusters.', ha='center', fontsize=10, color='gray')
    if save_path is not None:
        # Join the save path with the filename without // inbetween
        plt.savefig(save_path + "_silhouette_scores.png")
    plt.show()
    
    # Davies-Bouldin Score
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, davies_bouldin_scores, color='salmon', label='Davies-Bouldin Score')
    ax.set_xlabel('Model')
    ax.set_ylabel('Davies-Bouldin Score')
    ax.set_title('Davies-Bouldin Scores for Different Models')
    ax.set_xticklabels(cluster_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')
    if annotate:
        for bar, score in zip(bars, davies_bouldin_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom"
            )
    plt.tight_layout()
    fig.text(0.5, 0.01, '[0, inf] Lower Davies-Bouldin scores indicate better-defined clusters.', ha='center', fontsize=10, color='gray')
    if save_path is not None:
        plt.savefig(save_path + "_davies_bouldin_scores.png")
    plt.show()
    
    # Calinski-Harabasz Score
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, calinski_harabasz_scores, color='lightgreen', label='Calinski-Harabasz Score')
    ax.set_xlabel('Model')
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Scores for Different Models')
    ax.set_xticklabels(cluster_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')
    if annotate:
        for bar, score in zip(bars, calinski_harabasz_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom"
            )
    plt.tight_layout()
    fig.text(0.5, 0.01, '[0, inf] Higher Calinski-Harabasz scores indicate better-defined clusters.', ha='center', fontsize=10, color='gray')
    if save_path is not None:           
        plt.savefig(save_path + "_calinski_harabasz_scores.png")
    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt


import os
import re
import numpy as np
import matplotlib.pyplot as plt


def _extract_cluster_number(filename):
    """
    Extract cluster number from filename.
    Tries patterns like:
    - Cluster_2
    - cluster_2
    - Cluster2
    - k_2
    - _2clusters
    """
    patterns = [
        r'[Cc]luster[_\- ]?(\d+)',
        r'[_\-]k[_\-]?(\d+)',
        r'(\d+)[_\- ]?[Cc]lusters?'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))

    return None


def _extract_run_metadata(filename):
    """
    Extract run metadata from filename after 'DCEC_'.

    Assumes the filename contains something like:
    DCEC_400x400_O_subjects_run1_...

    Returns:
        parcellation, subject_group, movie_run
    """
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    match = re.search(r'DCEC_(.+)', basename)
    if not match:
        return "Unknown parcellation", "Unknown subjects", "Unknown run"

    tail = match.group(1)
    parts = tail.split('_')

    if len(parts) < 3:
        return "Unknown parcellation", "Unknown subjects", "Unknown run"

    parcellation = parts[0]

    run_idx = None
    for i, part in enumerate(parts):
        if re.fullmatch(r'run\d+', part, flags=re.IGNORECASE):
            run_idx = i
            break

    if run_idx is None:
        subject_group = "_".join(parts[1:-1]) if len(parts) > 2 else "Unknown subjects"
        movie_run = parts[-1]
    else:
        subject_group = "_".join(parts[1:run_idx]) if run_idx > 1 else "Unknown subjects"
        movie_run = parts[run_idx]

    return parcellation, subject_group, movie_run


def plot_clustering_scores_sorted(
    models_dir="Clusters",
    model_prefix=None,
    labels_tag="labels_predicted_labels_",
    middle_layer_tag="middle_layer_predicted_labels_",
    print_results=False,
    save_path=None,
    normalize=False,
    cluster_range=range(2, 11),
    ytop=None,
    figsize=(12, 7),
    annotate_points=True
):
    """
    Compute and plot Silhouette, Davies-Bouldin, and Calinski-Harabasz scores
    for every clustering in a run, sorted by cluster number.

    The plot:
    - sorts results from cluster 2 to 10
    - labels every data point with 'Cluster {x}'
    - can save the figure
    - uses title info parsed from filename after 'DCEC_'

    Args:
        models_dir (str): Folder with clustering result files.
        model_prefix (str or None): Prefix used to filter files.
        labels_tag (str): Identifier for label files.
        middle_layer_tag (str): Identifier for embedding files.
        print_results (bool): Whether to print metric results.
        save_path (str or None): Path to save figure. If None, figure is not saved.
        normalize (bool): If True, min-max normalize each metric before plotting.
        cluster_range (iterable): Allowed cluster numbers, default 2..10.
        figsize (tuple): Figure size.
        annotate_points (bool): Whether to label points with 'Cluster {x}'.

    Returns:
        dict: Results dictionary keyed by cluster number.
    """
    raw_results = {}

    for file in os.listdir(models_dir):
        if not file.endswith(".txt"):
            continue
        if labels_tag not in file:
            continue
        if model_prefix is not None and not file.startswith(model_prefix):
            continue

        labels_file = os.path.join(models_dir, file)
        middle_layer_file = labels_file.replace(labels_tag, middle_layer_tag)

        if not os.path.exists(middle_layer_file):
            print(f"Skipping {file}: missing middle-layer file.")
            continue

        cluster_num = _extract_cluster_number(file)
        if cluster_num is None:
            print(f"Skipping {file}: could not extract cluster number.")
            continue
        if cluster_num not in cluster_range:
            continue

        try:
            labels = np.loadtxt(labels_file)
            z = np.loadtxt(middle_layer_file)

            silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg = evaluate_single_clustering(
                z, labels, print_results=print_results
            )

            raw_results[cluster_num] = {
                "file": file,
                "silhouette": silhouette_avg,
                "davies_bouldin": davies_bouldin_avg,
                "calinski_harabasz": calinski_harabasz_avg
            }

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if len(raw_results) == 0:
        print("No valid models found for plotting.")
        return raw_results

    sorted_clusters = sorted(raw_results.keys())
    sorted_clusters = [c for c in sorted_clusters if c in cluster_range]

    silhouette_vals = np.array([raw_results[c]["silhouette"] for c in sorted_clusters], dtype=float)
    davies_vals = np.array([raw_results[c]["davies_bouldin"] for c in sorted_clusters], dtype=float)
    calinski_vals = np.array([raw_results[c]["calinski_harabasz"] for c in sorted_clusters], dtype=float) / 1000.0

    first_file = raw_results[sorted_clusters[0]]["file"]
    parcellation, subject_group, movie_run = _extract_run_metadata(first_file)

    def minmax_scale(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if np.isclose(arr_min, arr_max):
            return np.ones_like(arr) * 0.5
        return (arr - arr_min) / (arr_max - arr_min)

    x = np.array(sorted_clusters)

    plt.figure(figsize=figsize)

    if normalize:
        silhouette_plot = minmax_scale(silhouette_vals)
        davies_plot = minmax_scale(davies_vals)
        calinski_plot = minmax_scale(calinski_vals)

        plt.plot(x, silhouette_plot, marker='o', label='Silhouette')
        plt.plot(x, davies_plot, marker='s', label='Davies-Bouldin')
        plt.plot(x, calinski_plot, marker='^', label='Calinski-Harabasz')

        if annotate_points:
            for xi, yi in zip(x, silhouette_plot):
                plt.annotate(f"Cluster {xi}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center')
            for xi, yi in zip(x, davies_plot):
                plt.annotate(f"Cluster {xi}", (xi, yi), textcoords="offset points", xytext=(0, -14), ha='center')
            for xi, yi in zip(x, calinski_plot):
                plt.annotate(f"Cluster {xi}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center')

        plt.ylabel("Normalized score")
    else:
        plt.plot(x, silhouette_vals, marker='o', label='Silhouette')
        plt.plot(x, davies_vals, marker='s', label='Davies-Bouldin')
        plt.plot(x, calinski_vals, marker='^', label='Calinski-Harabasz')

        if annotate_points:
            for xi, yi in zip(x, silhouette_vals):
                plt.annotate(f"Cluster {xi}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center')
            for xi, yi in zip(x, davies_vals):
                plt.annotate(f"Cluster {xi}", (xi, yi), textcoords="offset points", xytext=(0, -14), ha='center')
            for xi, yi in zip(x, calinski_vals):
                plt.annotate(f"Cluster {xi}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center')

        plt.ylabel("Score")

    plt.xlabel("Number of clusters")
    if ytop is not None:
        plt.ylim(bottom=-0.1, top=ytop)
    plt.xticks(list(cluster_range))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.title(
        f"Clustering scores for {movie_run} | Parcellation: {parcellation} | Subjects: {subject_group}"
    )

    footnote_text = (
    "Silhouette score ∈ [-1, 1] ↑ | "
    "Davies-Bouldin score ∈ [0, ∞) ↓ | "
    "Calinski-Harabasz score ∈ [0, ∞) ↑ (÷1000)"
    )

    plt.figtext(
        0.5, -0.02,  # x (center), y (slightly below plot)
        footnote_text,
        ha='center',
        fontsize=9
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return raw_results

# ___________________________________________________________________________
# Plot the relationships from the behavioural analysis
# ___________________________________________________________________________

# Plot the CCA and MLR scores across clusters
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json_results(json_path):
    """
    Load clustering results from a JSON file.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON dictionary.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(d, key_path):
    """
    Safely get a nested value from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to search.
    key_path : str
        Colon-separated path, e.g.
        'cca_removed_subjects_results:cv_mean_cc'

    Returns
    -------
    value or None
        The value if found, otherwise None.
    """
    keys = key_path.split(":")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def plot_cca_mlr_across_clusters(
    json_path,
    cca_metric="cca_removed_subjects_results:cv_mean_cc",
    mlr_metric="mlr_removed_subjects_results:mean_accuracy",
    title="CCA and MLR across clusters",
    xlabel="Cluster",
    ylabel="Score",
    save_path=None,
    annotate=False,
    figsize=(10, 6)
):
    """
    Plot two line graphs across clusters:
      - CCA values
      - MLR values

    Parameters
    ----------
    json_path : str or Path
        Path to JSON file.
    cca_metric : str
        Colon-separated path to the CCA metric.
    mlr_metric : str
        Colon-separated path to the MLR metric.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    save_path : str or Path or None
        If provided, save figure here.
    annotate : bool
        If True, annotate each point with its value.
    figsize : tuple
        Figure size.
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Sort clusters numerically
    sorted_items = sorted(
        results.items(),
        key=lambda item: int(item[0].split("_")[1])
    )

    cluster_numbers = []
    cca_values = []
    mlr_values = []

    for cluster_name, cluster_data in sorted_items:
        cluster_num = int(cluster_name.split("_")[1])

        cca_val = get_nested_value(cluster_data, cca_metric)
        mlr_val = get_nested_value(cluster_data, mlr_metric)
        
        cluster_numbers.append(cluster_num)

        # Skip clusters where either value is missing/null
        if cca_val is None or mlr_val is None:
            cca_values.append(0.0)
            mlr_values.append(0.0)
            continue

        cca_values.append(cca_val)
        mlr_values.append(mlr_val)

    plt.figure(figsize=figsize)

    plt.plot(cluster_numbers, cca_values, marker="o", label="CCA")
    plt.plot(cluster_numbers, mlr_values, marker="o", label="MLR")

    if annotate:
        for x, y in zip(cluster_numbers, cca_values):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center")
        for x, y in zip(cluster_numbers, mlr_values):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points", ha="center")

    plt.xticks(cluster_numbers, [f"Cluster {n}" for n in cluster_numbers], rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    footnote_text = (
    "Canonical Correlation Analysis (CCA) [0, 1] and Multinomial Logistic Regression (MLR) [0, 1] scores across clusters. Both cross-validated. "
    )

    plt.figtext(
        0.5, -0.02,  # x (center), y (slightly below plot)
        footnote_text,
        ha='center',
        fontsize=9
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return cluster_numbers, cca_values, mlr_values


# Plot the scores for the best clustering results
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(d, key_path):
    keys = key_path.split(":")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def format_list(lst, max_len=5):
    """
    Format list for annotation (avoid huge text blocks).
    """
    if not lst:
        return "None"
    if len(lst) <= max_len:
        return ", ".join(map(str, lst))
    return ", ".join(map(str, lst[:max_len])) + ", ..."


def plot_single_cluster_cca_mlr_variants(
    json_path,
    cluster_number,
    cca_value_key="cv_mean_cc",
    mlr_value_key="mean_accuracy",
    dead_value=0.0,
    title=None,
    save_path=None,
    annotate=True,
    figsize=(10, 7)
):
    """
    Plot one chosen cluster across:
      - all variables
      - selected variables
      - removed subjects

    Produces two lines:
      - CCA
      - MLR

    Adds figure text below the plot containing selected variables
    and removed subjects for both CCA and MLR.
    """
    results = load_json_results(json_path)

    cluster_key = f"Cluster_{cluster_number}_results"
    if cluster_key not in results:
        raise ValueError(f"{cluster_key} not found in JSON file.")

    cluster_data = results[cluster_key]

    categories = ["All variables", "Selected variables", "Removed subjects"]

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

    cca_values = []
    mlr_values = []

    for path in cca_paths:
        value = get_nested_value(cluster_data, path)
        cca_values.append(dead_value if value is None else value)

    for path in mlr_paths:
        value = get_nested_value(cluster_data, path)
        mlr_values.append(dead_value if value is None else value)

    x = list(range(len(categories)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, cca_values, marker="o", label="CCA")
    ax.plot(x, mlr_values, marker="o", label="MLR")

    if annotate:
        for xi, yi in zip(x, cca_values):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 7),
                        textcoords="offset points", ha="center")
        for xi, yi in zip(x, mlr_values):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, -14),
                        textcoords="offset points", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_title(title or f"Cluster {cluster_number}: CCA and MLR across result types")
    ax.grid(True)
    ax.legend()

    # Figure text below plot
    cca_selected = format_list(cluster_data.get("cca_selected_variables", []))
    mlr_selected = format_list(cluster_data.get("mlr_selected_variables", []))
    cca_removed = format_list(cluster_data.get("cca_removed_subjects", []))
    mlr_removed = format_list(cluster_data.get("mlr_removed_subjects", []))

    fig_text = (
        f"CCA selected variables: {cca_selected}\n"
        f"MLR selected variables: {mlr_selected}\n"
        f"CCA removed subjects: {cca_removed}\n"
        f"MLR removed subjects: {mlr_removed}"
    )

    fig.text(
        0.02, 0.01,
        fig_text,
        ha="left",
        va="bottom",
        fontsize=9
    )

    # Leave extra room at the bottom for figure text
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "categories": categories,
        "cca_values": cca_values,
        "mlr_values": mlr_values,
        "figure_text": fig_text
    }


# Find the common removed subjects and the common selected variables across 
import json
from pathlib import Path
from collections import Counter


def _count_occurrences(dict_of_sets):
    """
    Count how many times each element appears across sets.

    Example:
        {'Cluster_2': {A, B}, 'Cluster_3': {A, C}}
    -> Counter({A: 2, B: 1, C: 1})
    """
    counter = Counter()

    for s in dict_of_sets.values():
        counter.update(s)

    return counter

def load_json_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_set(value):
    """
    Convert a list-like value to a set.
    Returns empty set for None or missing values.
    """
    if value is None:
        return set()
    if isinstance(value, list):
        return set(value)
    return set()


def _cluster_sort_key(cluster_name):
    return int(cluster_name.split("_")[1])


def _extract_common_and_union(dict_of_sets):
    """
    Compute intersection and union across non-empty sets.
    """
    nonempty_items = {k: v for k, v in dict_of_sets.items() if len(v) > 0}

    if len(nonempty_items) == 0:
        return {
            "common": set(),
            "union": set(),
            "used_entries": [],
        }

    sets = list(nonempty_items.values())

    return {
        "common": set.intersection(*sets),
        "union": set.union(*sets),
        "used_entries": list(nonempty_items.keys()),
    }


def _normalize_keys(keys):
    """
    Accept either:
      - a string
      - a list/tuple of strings
    and always return a list of strings.
    """
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, (list, tuple)):
        return list(keys)
    raise TypeError("Keys must be a string or a list/tuple of strings.")


def _combine_keys_as_set(data_dict, keys):
    """
    Combine multiple list-valued keys from one cluster/run into one set.

    Example:
        keys = ["cca_removed_subjects", "mlr_removed_subjects"]

    Result:
        union of both lists as a set
    """
    keys = _normalize_keys(keys)

    combined = set()
    for key in keys:
        combined |= _safe_set(data_dict.get(key, []))
    return combined


def compare_common_across_clusters(
    json_path,
    variable_keys="cca_selected_variables",
    removed_subjects_keys="cca_removed_subjects",
    clusters_to_include=None,
    ignore_empty=True
):
    """
    Compare common selected variables and removed subjects across clusters
    within a single JSON file.

    Supports both single keys and multiple keys, e.g.
        variable_keys=["cca_selected_variables", "mlr_selected_variables"]
        removed_subjects_keys=["cca_removed_subjects", "mlr_removed_subjects"]

    Parameters
    ----------
    json_path : str or Path
    variable_keys : str or list[str]
        Key(s) for variables to combine before comparison.
    removed_subjects_keys : str or list[str]
        Key(s) for removed subjects to combine before comparison.
    clusters_to_include : list[int] or None
    ignore_empty : bool

    Returns
    -------
    dict
    """
    results = load_json_results(json_path)

    cluster_variable_sets = {}
    cluster_removed_sets = {}

    sorted_items = sorted(results.items(), key=lambda item: _cluster_sort_key(item[0]))

    for cluster_name, cluster_data in sorted_items:
        cluster_num = _cluster_sort_key(cluster_name)

        if clusters_to_include is not None and cluster_num not in clusters_to_include:
            continue

        short_name = f"Cluster_{cluster_num}"

        var_set = _combine_keys_as_set(cluster_data, variable_keys)
        rem_set = _combine_keys_as_set(cluster_data, removed_subjects_keys)

        cluster_variable_sets[short_name] = var_set
        cluster_removed_sets[short_name] = rem_set

    if ignore_empty:
        variable_summary = _extract_common_and_union(cluster_variable_sets)
        removed_summary = _extract_common_and_union(cluster_removed_sets)
    else:
        if len(cluster_variable_sets) > 0:
            variable_summary = {
                "common": set.intersection(*cluster_variable_sets.values()),
                "union": set.union(*cluster_variable_sets.values()),
                "used_entries": list(cluster_variable_sets.keys()),
            }
        else:
            variable_summary = {"common": set(), "union": set(), "used_entries": []}

        if len(cluster_removed_sets) > 0:
            removed_summary = {
                "common": set.intersection(*cluster_removed_sets.values()),
                "union": set.union(*cluster_removed_sets.values()),
                "used_entries": list(cluster_removed_sets.keys()),
            }
        else:
            removed_summary = {"common": set(), "union": set(), "used_entries": []}

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
        "variable_counts_sorted": sorted(variable_counts.items(), key=lambda x: -x[1]),
        "clusters_used_for_variables": variable_summary["used_entries"],
        "common_removed_subjects": sorted(removed_summary["common"]),
        "union_removed_subjects": sorted(removed_summary["union"]),
        "clusters_used_for_removed_subjects": removed_summary["used_entries"],
        "removed_subjects_counts": dict(removed_counts),
        "removed_subjects_counts_sorted": sorted(removed_counts.items(), key=lambda x: -x[1]),
    }


def compare_common_across_runs_same_cluster(
    json_paths,
    cluster_number,
    variable_keys="cca_selected_variables",
    removed_subjects_keys="cca_removed_subjects",
    run_names=None,
    ignore_empty=True
):
    """
    Compare common selected variables and removed subjects across multiple runs
    for the same cluster.

    Supports both single keys and multiple keys, e.g.
        variable_keys=["cca_selected_variables", "mlr_selected_variables"]
        removed_subjects_keys=["cca_removed_subjects", "mlr_removed_subjects"]
    """
    json_paths = [Path(p) for p in json_paths]

    if run_names is None:
        run_names = [p.stem for p in json_paths]

    if len(run_names) != len(json_paths):
        raise ValueError("run_names must have the same length as json_paths.")

    cluster_key = f"Cluster_{cluster_number}_results"

    run_variable_sets = {}
    run_removed_sets = {}

    for json_path, run_name in zip(json_paths, run_names):
        results = load_json_results(json_path)

        if cluster_key not in results:
            raise ValueError(f"{cluster_key} not found in {json_path}")

        cluster_data = results[cluster_key]

        run_variable_sets[run_name] = _combine_keys_as_set(cluster_data, variable_keys)
        run_removed_sets[run_name] = _combine_keys_as_set(cluster_data, removed_subjects_keys)

    if ignore_empty:
        variable_summary = _extract_common_and_union(run_variable_sets)
        removed_summary = _extract_common_and_union(run_removed_sets)
    else:
        variable_summary = {
            "common": set.intersection(*run_variable_sets.values()) if len(run_variable_sets) > 0 else set(),
            "union": set.union(*run_variable_sets.values()) if len(run_variable_sets) > 0 else set(),
            "used_entries": list(run_variable_sets.keys()),
        }
        removed_summary = {
            "common": set.intersection(*run_removed_sets.values()) if len(run_removed_sets) > 0 else set(),
            "union": set.union(*run_removed_sets.values()) if len(run_removed_sets) > 0 else set(),
            "used_entries": list(run_removed_sets.keys()),
        }

    variable_counts = _count_occurrences(variable_summary)
    removed_counts = _count_occurrences(removed_summary)

    return {
        "cluster_number": cluster_number,
        "variable_keys": _normalize_keys(variable_keys),
        "removed_subjects_keys": _normalize_keys(removed_subjects_keys),
        "per_run_variables": {k: sorted(v) for k, v in run_variable_sets.items()},
        "per_run_removed_subjects": {k: sorted(v) for k, v in run_removed_sets.items()},
        "common_variables": sorted(variable_summary["common"]),
        "union_variables": sorted(variable_summary["union"]),
        "variable_counts": dict(variable_counts),
        "variable_counts_sorted": sorted(variable_counts.items(), key=lambda x: -x[1]),
        "runs_used_for_variables": variable_summary["used_entries"],
        "common_removed_subjects": sorted(removed_summary["common"]),
        "union_removed_subjects": sorted(removed_summary["union"]),
        "removed_subjects_counts": dict(removed_counts),
        "removed_subjects_counts_sorted": sorted(removed_counts.items(), key=lambda x: -x[1]),
        "runs_used_for_removed_subjects": removed_summary["used_entries"],
    }


def print_comparison_summary(comparison_result, save_path=None):
    """
    Pretty-print the result.
    """
    if "json_file" in comparison_result:
        print(f"\nComparison across clusters in: {comparison_result['json_file']}")
        save_results = {
            "json_file": comparison_result["json_file"],
        }
    else:
        print(f"\nComparison across runs for Cluster {comparison_result['cluster_number']}")
        save_results = {
            "cluster_number": comparison_result["cluster_number"],
        }

    print(f"Variable keys: {comparison_result['variable_keys']}")
    print(f"Removed-subject keys: {comparison_result['removed_subjects_keys']}")

    print("\nCommon variables:")
    print(comparison_result["common_variables"])

    print("\nVariable counts across clusters/runs:")
    top_5_vars = comparison_result["variable_counts_sorted"][:5]
    for var, count in top_5_vars:
        print(f"{var}: {count}")

    print("\nCommon removed subjects:")
    print(comparison_result["common_removed_subjects"])

    print("\nRemoved subject counts across clusters/runs:")
    top_5_subjects = comparison_result["removed_subjects_counts_sorted"][:5]
    for subject, count in top_5_subjects:
        print(f"{subject}: {count}")

    save_results.update({
        "common_variables": comparison_result["common_variables"],
        "variable_counts_sorted": comparison_result["variable_counts_sorted"][:5],
        "common_removed_subjects": comparison_result["common_removed_subjects"],
        "removed_subjects_counts_sorted": comparison_result["removed_subjects_counts_sorted"][:5],
    })

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=4)
        print(f"\nComparison result saved to: {save_path}")


# Compare between the ages
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(d, key_path):
    """
    Safely get a nested value from a dictionary using a colon-separated path.
    Example:
        'cca_removed_subjects_results:cv_mean_cc'
    """
    keys = key_path.split(":")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current

import matplotlib.colors as mcolors

def adjust_lightness(color, amount=1.2):
    """
    amount > 1 → lighter
    amount < 1 → darker
    """
    import colorsys
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * amount))
    return colorsys.hls_to_rgb(h, l, s)

def plot_cca_mlr_two_jsons_across_clusters(
    young_json_path,
    old_json_path,
    cca_metric="cca_removed_subjects_results:cv_mean_cc",
    mlr_metric="mlr_removed_subjects_results:mean_accuracy",
    label_1="Young",
    label_2="Old",
    dead_value=0.0,
    include_all_clusters=True,
    annotate=False,
    title="CCA and MLR across clusters for two JSON files",
    xlabel="Cluster",
    ylabel="Score",
    save_path=None,
    figsize=(11, 6)
):
    """
    Plot 4 lines across clusters:
      - CCA from JSON 1
      - MLR from JSON 1
      - CCA from JSON 2
      - MLR from JSON 2

    Parameters
    ----------
    json_path_1 : str or Path
    json_path_2 : str or Path
    cca_metric : str
        Colon-separated path to the CCA metric inside each cluster.
    mlr_metric : str
        Colon-separated path to the MLR metric inside each cluster.
    label_1 : str
        Label prefix for first JSON file.
    label_2 : str
        Label prefix for second JSON file.
    dead_value : float
        Value used when metric is missing/null.
    include_all_clusters : bool
        If True, includes all cluster numbers found in either JSON.
        Missing clusters are plotted as dead_value.
        If False, only clusters present in both JSONs are included.
    annotate : bool
        If True, annotate each point with its numeric value.
    title : str
    xlabel : str
    ylabel : str
    save_path : str or Path or None
    figsize : tuple

    Returns
    -------
    dict
        Extracted plotting values.
    """
    results_young = load_json_results(young_json_path)
    results_old = load_json_results(old_json_path)

    clusters_young = {int(name.split("_")[1]) for name in results_young.keys()}
    clusters_old = {int(name.split("_")[1]) for name in results_old.keys()}

    if include_all_clusters:
        cluster_numbers = sorted(clusters_young | clusters_old)
    else:
        cluster_numbers = sorted(clusters_young & clusters_old)

    cca_values_young = []
    mlr_values_young = []
    cca_values_old = []
    mlr_values_old = []

    for cluster_num in cluster_numbers:
        cluster_key = f"Cluster_{cluster_num}_results"

        cluster_data_1 = results_young.get(cluster_key, {})
        cluster_data_2 = results_old.get(cluster_key, {})

        cca_young = get_nested_value(cluster_data_1, cca_metric)
        mlr_young = get_nested_value(cluster_data_1, mlr_metric)
        cca_old = get_nested_value(cluster_data_2, cca_metric)
        mlr_old = get_nested_value(cluster_data_2, mlr_metric)

        cca_values_young.append(dead_value if cca_young is None else cca_young)
        mlr_values_young.append(dead_value if mlr_young is None else mlr_young)
        cca_values_old.append(dead_value if cca_old is None else cca_old)
        mlr_values_old.append(dead_value if mlr_old is None else mlr_old)

    plt.figure(figsize=figsize)

    base_colors = {"CCA": "green", "MLR": "red"}

    plt.plot(cluster_numbers, cca_values_young, marker="o", label=f"{label_1} - CCA", color=adjust_lightness(base_colors["CCA"], 0.7))
    plt.plot(cluster_numbers, mlr_values_young, marker="o", label=f"{label_1} - MLR", color=adjust_lightness(base_colors["MLR"], 0.7))
    plt.plot(cluster_numbers, cca_values_old, marker="o", label=f"{label_2} - CCA", color=adjust_lightness(base_colors["CCA"], 1.3))
    plt.plot(cluster_numbers, mlr_values_old, marker="o", label=f"{label_2} - MLR", color=adjust_lightness(base_colors["MLR"], 1.3))

    if annotate:
        for x, y in zip(cluster_numbers, cca_values_young):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        for x, y in zip(cluster_numbers, mlr_values_young):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8)
        for x, y in zip(cluster_numbers, cca_values_old):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(10, 7), textcoords="offset points", ha="center", fontsize=8)
        for x, y in zip(cluster_numbers, mlr_values_old):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(10, -12), textcoords="offset points", ha="center", fontsize=8)

    plt.xticks(cluster_numbers, [f"Cluster {n}" for n in cluster_numbers], rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "cluster_numbers": cluster_numbers,
        "young_cca": cca_values_young,
        "young_mlr": mlr_values_young,
        "old_cca": cca_values_old,
        "old_mlr": mlr_values_old,
    }

# Compare the common selected variables and removed subjects across the ages 
import json
from pathlib import Path
from collections import Counter


def load_json_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _normalize_keys(keys):
    """
    Accept either a string or a list/tuple of strings.
    Always return a list.
    """
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, (list, tuple)):
        return list(keys)
    raise TypeError("Keys must be a string or a list/tuple of strings.")


def count_occurrences_across_two_jsons(
    young_json_path,
    old_json_path,
    keys=["cca_selected_variables", "mlr_selected_variables", "cca_removed_subjects", "mlr_removed_subjects"],
    clusters_to_include=None,
    remove_duplicates_within_cluster=False,
    sort_descending=True
):
    """
    Count how many times each variable/subject appears across all clusters
    in two JSON files combined.

    Example:
        keys = ["cca_selected_variables", "mlr_selected_variables"]
    or
        keys = ["cca_removed_subjects", "mlr_removed_subjects"]

    If there are 9 clusters in each file, the maximum count is 18 when
    remove_duplicates_within_cluster=True.

    Parameters
    ----------
    json_path_1 : str or Path
    json_path_2 : str or Path
    keys : str or list[str]
        JSON key(s) to count across all clusters.
    clusters_to_include : list[int] or None
        Optional subset of cluster numbers to include.
    remove_duplicates_within_cluster : bool
        If True, an item is counted at most once per cluster per file,
        even if it appears in both CCA and MLR lists for that cluster.
        This is usually what you want if max count should be 18.
    sort_descending : bool
        If True, sort most common first.

    Returns
    -------
    dict
        Count summary.
    """
    results_young = load_json_results(young_json_path)
    results_old = load_json_results(old_json_path)
    keys = _normalize_keys(keys)

    counter = Counter()
    total_cluster_entries = 0

    for results, file_label in [(results_young, "young"), (results_old, "old")]:
        for cluster_name, cluster_data in results.items():
            if not cluster_name.startswith("Cluster_"):
                continue

            cluster_num = int(cluster_name.split("_")[1])

            if clusters_to_include is not None and cluster_num not in clusters_to_include:
                continue

            total_cluster_entries += 1

            collected = []
            for key in keys:
                collected.extend(_safe_list(cluster_data.get(key, [])))

            if remove_duplicates_within_cluster:
                collected = set(collected)

            counter.update(collected)

    items = list(counter.items())
    if sort_descending:
        items.sort(key=lambda x: (-x[1], str(x[0])))
    else:
        items.sort(key=lambda x: str(x[0]))

    return {
        "young_json_path": str(young_json_path),
        "old_json_path": str(old_json_path),
        "keys": keys,
        "clusters_to_include": clusters_to_include,
        "remove_duplicates_within_cluster": remove_duplicates_within_cluster,
        "max_possible_count": total_cluster_entries,
        "counts": dict(counter),
        "counts_sorted": items,
    }