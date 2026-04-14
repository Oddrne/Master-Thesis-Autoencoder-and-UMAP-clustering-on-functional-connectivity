import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt

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

filepath = "Prosjektoppgave-Odd-Arne-og-Mats-main\\subject_features.npz" # All subjects, 50 features


def load_workable_fc(filepath):
    all_features = load_npz_data(filepath)

    print(all_features.shape)

    all_features.to_csv("subject_features_all.csv", index=True)
     
    cleaned_features = remove_duplicate_pairs(all_features)

    print(cleaned_features.shape)

    return cleaned_features


# load_fc_mat_matrices()

def load_static_functional_connectivities(filepath="Input Data\\ADHD_connectivity.mat"):
    file = loadmat(filepath)
    fc = file['connectivities']  # 487x672
    fc_coordinates = file['coordinates']  # 672x1
    
    print(f"Original FC shape: {fc.shape}")

    # Keep every column in fc if the row in fc_coordinates begins with "SFC"
    fc = fc[:, [i for i in range(fc_coordinates.shape[0]) if fc_coordinates[i][0][0].startswith("SFC")]]
    
    print(f"Final FC shape: {fc.shape}")
    
    return fc


# Function for extracting res_scores from the CSV file. Used un the project thesis.
def extract_res_scores_from_csv(file_path: str) -> pd.DataFrame:
    """
    Extracts the 'Subject' and 'Emo_res' columns from the specified CSV file.
    Parameters    ----------
    file_path : str
    Returns    -------
    pd.DataFrame        A DataFrame containing the 'Subject' and 'Emo_res' columns.
    pd.DataFrame        The entire DataFrame loaded from the CSV file for further use if needed.
    """
    all_data = pd.read_csv(file_path)
    res_scores_df = all_data[["Subject", "Emo_res", "Sex", "dEV_neu", "dEV_neg", "DS", "rmet"]]

    return res_scores_df, all_data

def extract_cluster_labels_from_txt(file_path: str, df: pd.DataFrame, column_name: str = "Predicted_Labels") -> pd.DataFrame:
    """
    Extracts cluster labels from a specified text file.
    Parameters    ----------
    file_path : str
    subject_ids : list, optional
        A list of subject IDs to filter the cluster labels.
    Returns    -------
    pd.DataFrame        A DataFrame containing the cluster labels.
    """
    # Idea: Read the cluster labels from the text file from each clustering.
    cluster_labels = pd.read_csv(file_path, header=None)[0]

    df[column_name] = cluster_labels.astype(int)
    return df

def extract_all_cluster_labels_from_txt(start_string: str, df: pd.DataFrame, clusters_range: range = range(2, 11)) -> pd.DataFrame:
    """
    Extracts cluster labels from multiple text files and adds them to the DataFrame.
    Parameters    ----------
    start_string : str
        The common prefix for the cluster label text files.
    df : pd.DataFrame
        The DataFrame to which the cluster labels will be added.
    clusters_range : range, optional
        The range of cluster numbers to extract (default is range(2, 11)).
    Returns    -------
    pd.DataFrame
        The updated DataFrame with cluster labels added as new columns.
    """
    for i in clusters_range:
        cluster_path = f"{start_string}cluster_{i}_labels_predicted_labels_.txt"
        df = extract_cluster_labels_from_txt(cluster_path, df=df, column_name=f"Cluster_{i}")
    
    return df

def plot_correlation_heatmap(df: pd.DataFrame, plt_title: str = "Pearson Correlation Heatmap for [Age] subjects and run [x] ([emotion] movie)", save_path: str = None):
    df = df.drop(columns=["Subject"])
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(plt_title)
    plt.show()

    if save_path:
        plt.savefig(save_path)

def load_whole_behavioural_data(csv_path: str, included_ids_path: str, cluster_labels_path: str, clusters_range: range = range(2, 11)):
    """
    Load the whole behavioural data, including the res scores and the cluster labels for each clustering.

    Parameters    
    ----------
    csv_path : str
    included_ids_path : str
    cluster_labels_path : str
    clusters_range : range, optional
        The range of cluster numbers to extract (default is range(2, 11)).

    Returns    
    -------
    tuple
        A tuple containing the updated DataFrame and a list of appropriate column names.
    """
    csv_ids = pd.read_csv(included_ids_path)
    _, all_scores = extract_res_scores_from_csv(csv_path)

    all_scores = all_scores[all_scores["Subject"].isin(csv_ids["Subject"])]
    all_scores = all_scores.reset_index(drop=True)

    df = extract_all_cluster_labels_from_txt(start_string=cluster_labels_path, df=all_scores, clusters_range=clusters_range)

    columns = df.columns.tolist()
    columns.remove("Subject") 
    columns.remove("Group") 
    columns.remove("Gender")
    columns.remove("Handedness")  # Remove non-behavioural variables
    columns.remove("Medicine")  # Remove non-behavioural variables
    columns.remove("Cluster_2")  # Remove other cluster columns
    columns.remove("Cluster_3")  # Remove other cluster columns
    columns.remove("Cluster_4")  # Remove other cluster columns
    columns.remove("Cluster_5")  # Remove other cluster columns
    columns.remove("Cluster_6")  # Remove other cluster columns
    columns.remove("Cluster_7")  # Remove other cluster columns
    columns.remove("Cluster_8")  # Remove other cluster columns
    columns.remove("Cluster_9")  # Remove other cluster columns
    columns.remove("Cluster_10")  # Remove other cluster columns

    return df, columns

def combine_cca_mlr_pipeline_outputs(cca_pipeline_output: dict, mlr_pipeline_output: dict):
    """
    Combine CCA and MLR pipeline outputs in a fixed order.

    Expected keys in cca_pipeline_output:
        - cca_all_variables_results
        - cca_selected_variables_results
        - cca_removed_subjects_results
        - cca_selected_variables
        - cca_removed_subjects

    Expected keys in mlr_pipeline_output:
        - mlr_all_variables_results
        - mlr_selected_variables_results
        - mlr_removed_subjects_results
        - mlr_selected_variables
        - mlr_removed_subjects

    Returns
    -------
    dict
        Combined dictionary in the requested order.
    """
    combined_output = {
        "cca_all_variables_results": cca_pipeline_output["cca_all_variables_results"],
        "cca_selected_variables_results": cca_pipeline_output["cca_selected_variables_results"],
        "cca_removed_subjects_results": cca_pipeline_output["cca_removed_subjects_results"],
        "mlr_all_variables_results": mlr_pipeline_output["mlr_all_variables_results"],
        "mlr_selected_variables_results": mlr_pipeline_output["mlr_selected_variables_results"],
        "mlr_removed_subjects_results": mlr_pipeline_output["mlr_removed_subjects_results"],
        "cca_selected_variables": cca_pipeline_output["cca_selected_variables"],
        "mlr_selected_variables": mlr_pipeline_output["mlr_selected_variables"],
        "cca_removed_subjects": cca_pipeline_output["cca_removed_subjects"],
        "mlr_removed_subjects": mlr_pipeline_output["mlr_removed_subjects"],
    }

    return combined_output

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")