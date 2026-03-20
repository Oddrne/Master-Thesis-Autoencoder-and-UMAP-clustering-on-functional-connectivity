import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.datasets import load_digits
from scipy.io import loadmat
import os

# Load the data
PTSD = loadmat("C:\\Users\\oddar\\Downloads\\PTSD_connectivity.mat")
# PTSD is a dataset containing 87 samples (subjects) with 340 features (as vectorized functional connectivity matrices)
# The expected number of clusters are 3

# Define the functional connectivity matrix (example)
Functional_connectivity_matrix = PTSD["connectivities"]  # Example matrix

# Dummy data
X = Functional_connectivity_matrix  # [N,d] float tensor

def hdbscan_clustering(Functional_connectivity_matrix, save_labels=False):
    hdb = HDBSCAN(copy=True, min_cluster_size=10)
    hdb.fit(Functional_connectivity_matrix)
    
    print(hdb.labels_)
    if save_labels:
        # Add additional information to the filename if needed
        save_text = "test"
        
        # Save the labels to a text file
        np.savetxt(os.path.join("Clusters",f"HDBSCAN_labels_{save_text}.txt"), hdb.labels_, fmt='%d')


