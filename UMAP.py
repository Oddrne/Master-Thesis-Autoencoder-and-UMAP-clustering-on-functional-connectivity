# Define a UMAP class that can be used for dimensionality reduction and clustering

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(int)

standard_embedding = umap.UMAP(random_state=42).fit_transform(mnist.data)

plot = False


if plot==True:
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=mnist.target.astype(int), cmap='Spectral', s=0.1)
    # Show the plot
    plt.show()
    
