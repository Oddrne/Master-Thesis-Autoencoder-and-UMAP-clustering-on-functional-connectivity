from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
from scipy.io import loadmat
import numpy as np

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

def evaluate_single_clustering(functional_connectivity_matrix, labels_path):
    
    labels = np.loadtxt(labels_path)
    
    if len(np.unique(labels)) > 1:  # Ensure more than one cluster exists
        print(f"\n Scores for the given labels:")
        
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
        print("Only one cluster in the provided labels, silhouette coefficient not computed.")  
        
        
        
# Example
# PTSD = loadmat("C:\\Users\\oddar\\Downloads\\PTSD_connectivity.mat")
# PTSD is a dataset containing 87 samples (subjects) with 340 features (as vectorized functional connectivity matrices)
# The expected number of clusters are 3

# Define the functional connectivity matrix (example)
# Functional_connectivity_matrix = PTSD["connectivities"]  # Example matrix

# Define where to find the labels 
# labels_path = os.fsencode("Clusters")
# evaluate_clustering(Functional_connectivity_matrix, labels_path)