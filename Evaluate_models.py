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
        plt.savefig(os.path.join(save_path, "silhouette_scores.png"))
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
        plt.savefig(os.path.join(save_path, "davies_bouldin_scores.png"))
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
        plt.savefig(os.path.join(save_path, "calinski_harabasz_scores.png"))
    plt.show()