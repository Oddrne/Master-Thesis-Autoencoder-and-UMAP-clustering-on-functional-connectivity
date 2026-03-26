import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import torch
import random


def set_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)
    
    # PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)
    
    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Vectorize the upper triangle of the FC matrix (excluding diagonal)
def vectorize_fc_matrix(fc_matrix):
    n = fc_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1) # Removes diagonal as well
    return fc_matrix[upper_tri_indices]


def vectorize_fc_400_matrix(fc_matrix):
    upper_tri_indices = np.triu_indices(400, k=1) # Removes diagonal and lower triangle, leaving 400*399/2 = 79800 features
    return fc_matrix[upper_tri_indices]


def load_fc_matrix():
    filepath = "C:\\Mats og Odd Arne\\Prosjektoppgave\\ISC_data\\FC1.mat"
    fc_mat = loadmat(filepath)
    fc_mat = fc_mat['FC1']  # Extract the FC1 variable from the loaded .mat file

    print(fc_mat.shape)  # Should print (200, 200, N_subjects)

    subjects = []
    for i in range(fc_mat.shape[2]):
        fc_matrix = fc_mat[:, :, i]  # Get the FC matrix for the i-th 
        print(f"Processing subject {i+1} with FC matrix shape: {fc_matrix.shape}")  # Should print (200, 200)
        vectorized_fc = vectorize_fc_matrix(fc_matrix)  # Vectorize the upper triangle
        print(f"Vectorized FC shape for subject {i+1}: {vectorized_fc.shape}")  # Should print (19900,)
        subjects.append(vectorized_fc)  # Append to the list of subjects
    
    print(f"Total number of subjects: {len(subjects)}") 
    print("One subject's vectorized FC looks like:\n", subjects[0])  # Print the first subject's vectorized FC

    savemat("C:\\Mats og Odd Arne\\Prosjektoppgave\\sch407\\YA\\200_schaefer_vectorized_fc.mat", {"200_vectorized_fc": subjects}) 

# load_fc_matrix()


def load_fc_mat_matrices():
    N_FC_Matrices_m2 = 216
    ## Load data from matlab files
    subjects = []
    for i in range(N_FC_Matrices_m2):
        filepath = f"C:\\Mats og Odd Arne\\Prosjektoppgave\\sch407\\YA\\zFCmat\\sub-11{i:03d}_task-video_run-2__zFCmat.mat"

        try:
            fc_mat_m2 = loadmat(filepath)
            
            fc_df = pd.DataFrame(fc_mat_m2['zfcmatrix'])

        except Exception as e:
            continue
        
        print(f"Loaded FC matrix from {filepath} with shape {fc_df.shape}")
        vectorized_fc = vectorize_fc_400_matrix(fc_df.values)
        print(f"Vectorized FC shape: {vectorized_fc.shape}")

        subjects.append(vectorized_fc)
        # Process fc_matrix as needed

    #savemat("C:\\Mats og Odd Arne\\Prosjektoppgave\\sch407\\YA\\test_vectorized_fc.mat", {"vectorized_fc": subjects}) 

    print(f"Loaded {len(subjects)} FC matrices.")

    print(f"one subject looks like:\n {subjects[0]}")

### TEST LOADING A SINGLE FILE
# FCmat_data = loadmat("C:\\Mats og Odd Arne\\Prosjektoppgave\\sch407\\YA\\zFCmat\\sub-11012_task-video_run-2__zFCmat.mat")

# print(FCmat_data.keys())

# fc_test_df = pd.DataFrame(FCmat_data['zfcmatrix'])
# print(subjects[0])


def load_npz_data(filepath):
    npz_data = np.load(filepath, allow_pickle=True)

    values = npz_data['data']
    columns = npz_data['columns']
    index = npz_data['index']

    features_df = pd.DataFrame(values, columns=columns, index=index)

    return features_df

def remove_duplicate_pairs(df):
    keep, seen = [], set()
    for c in df.columns:
        if "_mean_conn" in c:
            parts = c.replace("_mean_conn", "").split("_")
            key = tuple(sorted(parts[:2]))
            if key not in seen:
                keep.append(c)
                seen.add(key)
        else:
            keep.append(c)
    return df[keep]

filepath = "Prosjektoppgave-Odd-Arne-og-Mats-main\subject_features.npz" # All subjects, 50 features


def load_workable_fc(filepath):
    all_features = load_npz_data(filepath)

    print(all_features.shape)

    all_features.to_csv("subject_features_all.csv", index=True)
     
    cleaned_features = remove_duplicate_pairs(all_features)

    print(cleaned_features.shape)

    return cleaned_features


# load_fc_mat_matrices()

def load_static_functional_connectivities(filepath="Input Data\ADHD_connectivity.mat"):
    file = loadmat(filepath)
    fc = file['connectivities']  # 487x672
    fc_coordinates = file['coordinates']  # 672x1
    
    print(f"Original FC shape: {fc.shape}")

    # Keep every column in fc if the row in fc_coordinates begins with "SFC"
    fc = fc[:, [i for i in range(fc_coordinates.shape[0]) if fc_coordinates[i][0][0].startswith("SFC")]]
    
    print(f"Final FC shape: {fc.shape}")
    
    return fc

