import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

""" filepath = "C:\\Users\\oddar\\Downloads\\200_schaefer_vectorized_fc.mat"

fc_200 = loadmat(filepath)

print(fc_200.keys())
print(fc_200['200_vectorized_fc'].shape)
print(fc_200['200_vectorized_fc'][0:5])
print(fc_200['200_vectorized_fc'][2].sum())  # Sum of the first sample's features

# Count number of values below zero in fc_200['200_vectorized_fc']
count_below_zero = np.sum(fc_200['200_vectorized_fc'] < 0)
print(f"Number of values below zero: {count_below_zero}")



filepath_functional = "C:\\Users\\oddar\\Downloads\\test_vectorized_fc.mat"

fc_functional = loadmat(filepath_functional)
print(fc_functional.keys())
print(fc_functional['vectorized_fc'].shape)
print(fc_functional['vectorized_fc'][0:5])
print(fc_functional['vectorized_fc'][2].sum())

# Count number of values below zero in fc_functional['vectorized_fc']
count_below_zero_functional = np.sum(fc_functional['vectorized_fc'] < 0)
print(f"Number of values below zero in functional data: {count_below_zero_functional}") 

filepath_example = "C:\\Users\\oddar\\Downloads\\PTSD_connectivity.mat"

fc_example = loadmat(filepath_example)
print(fc_example.keys())
print(fc_example['connectivities'].shape)
print(fc_example['connectivities'][0:5])
print(fc_example['connectivities'][2].sum())

# Count number of values below zero in fc_example['connectivities']
count_below_zero_example = np.sum(fc_example['connectivities'] < 0)
print(f"Number of values below zero in example data: {count_below_zero_example}") """

# Loading the data with all 200 features (for 200x200 data)
# 200_schaefer_vectorized_fc.mat contains the vectorized upper triangle of the 200x200 FC matrices, resulting in 19900 features per sample.

from scipy.io import loadmat
import torch
import numpy as np

# FC_test_mat = loadmat("C:\\Users\\oddar\\Downloads\\200_schaefer_vectorized_fc.mat")
FC_test_mat = loadmat("C:\\Mats og Odd Arne\\Prosjektoppgave\\sch407\\YA\\200_schaefer_vectorized_fc.mat")  # Load the .mat file

FC_test_array = FC_test_mat["200_vectorized_fc"]  # Example matrix
# np.fill_diagonal(FC_test_array, 1.0)  # Set diagonal to zero

print(FC_test_array[0:5].shape)  # Print the first 5 rows to verify

X = torch.from_numpy(FC_test_array).float()  # Example matrix

# Check for NaN values in X
if torch.isnan(X).any():
    print("X contains NaN values. Check the data loading process.")
else:
    print("X loaded successfully with shape:", X.shape)