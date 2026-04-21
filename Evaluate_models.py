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