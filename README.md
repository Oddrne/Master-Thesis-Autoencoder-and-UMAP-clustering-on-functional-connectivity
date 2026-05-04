# Master Thesis: DCEC on Functional Connectivity

This repository contains code developed for a master’s thesis investigating the use of **Deep Convolutional Embedded Clustering (DCEC)** on functional connectivity matrices derived from fMRI data.

The project explores whether unsupervised deep clustering can identify meaningful structure in healthy subjects based on functional connectivity patterns, and whether the resulting clusters show any relationship to behavioural and questionnaire-based variables.

## Project description

Functional connectivity (FC) matrices describe statistical relationships between brain regions based on fMRI time series. In this project, FC matrices are used as input to a deep clustering pipeline.

The main method is based on **Deep Convolutional Embedded Clustering (DCEC)**. The model first learns a compressed representation of each FC matrix using a convolutional autoencoder. Cluster centres are then initialised in the latent space using KMeans, before the full model is trained using both reconstruction loss and clustering loss.

The project contains code for:

- preprocessing and collecting `.mat` files
- computing or loading functional connectivity matrices
- training convolutional autoencoders
- training DCEC models
- extracting latent embeddings
- saving predicted cluster labels
- evaluating clustering quality
- comparing clustering results with behavioural variables using CCA and MLR
- plotting clustering and evaluation results

## Repository structure

The most important files are:

```text
.
├── Convolutional_AE.py
├── Evaluate_models.py
├── Evaluate_clusterings_with_CCA.py
├── Evaluate_clusterings_with_MLR.py
├── Functional_Connectivity.py
├── Collect_matlab_files.py
├── data_loader.py
├── UMAP.py
├── My_HDBSCAN.py
├── Create_dummy_csv.py
├── test.py
├── test.ipynb
├── UMAP.ipynb
├── Prosjektoppgave-Odd-Arne-og-Mats-main/
└── README.md
```

### Main files

| File | Description |
|---|---|
| `Convolutional_AE.py` | Contains the DCEC model, convolutional autoencoder, clustering layer, pretraining, joint DCEC training, prediction utilities and plotting functions. |
| `Evaluate_models.py` | Contains functions for evaluating clustering results using internal clustering metrics and plotting score comparisons. |
| `Evaluate_clusterings_with_CCA.py` | Contains functions for comparing clustering results with behavioural variables using Canonical Correlation Analysis. |
| `Evaluate_clusterings_with_MLR.py` | Contains functions for comparing clustering results with behavioural variables using Multinomial Logistic Regression. |
| `Functional_Connectivity.py` | Contains functionality related to functional connectivity matrix handling. |
| `Collect_matlab_files.py` | Utility script for collecting and structuring MATLAB files. |
| `data_loader.py` | Utility code for loading data into the Python/PyTorch pipeline. |
| `UMAP.py` / `UMAP.ipynb` | Code for dimensionality reduction and visualisation using UMAP. |
| `My_HDBSCAN.py` | Code related to HDBSCAN clustering experiments. |
| `Create_dummy_csv.py` | Utility for creating dummy behavioural CSV data for testing the analysis pipeline. |

Some scripts and notebooks are experimental and may contain local paths or project-specific assumptions.

## Method overview

The main DCEC pipeline follows these steps:

1. Load functional connectivity matrices.
2. Convert the matrices into tensors suitable for convolutional neural networks.
3. Train a convolutional autoencoder using reconstruction loss.
4. Extract latent embeddings from the trained encoder.
5. Initialise cluster centres using KMeans.
6. Train the full DCEC model using:
   - reconstruction loss
   - KL-divergence clustering loss
7. Predict cluster labels for each subject.
8. Save cluster labels and latent embeddings.
9. Evaluate clustering quality.
10. Compare cluster labels with behavioural variables.

## DCEC model

The DCEC model consists of:

- a convolutional encoder
- a low-dimensional latent layer
- a trainable clustering layer
- a convolutional decoder

The clustering layer uses a Student’s t-distribution to compute soft cluster assignments. A target distribution is updated during training and used in the clustering loss.

The model is trained in two stages:

### 1. Autoencoder pretraining

The convolutional autoencoder is first trained to reconstruct the input FC matrices. This helps the encoder learn a useful latent representation before clustering is introduced.

### 2. Joint DCEC training

After pretraining, KMeans is applied to the latent embeddings to initialise the cluster centres. The model is then trained jointly using reconstruction loss and clustering loss.

## Evaluation

The repository contains two main types of evaluation.

### Internal clustering evaluation

Clustering quality is evaluated using:

- **Silhouette coefficient**
- **Davies-Bouldin score**
- **Calinski-Harabasz score**

These metrics are used to compare clustering solutions with different numbers of clusters.

### External behavioural evaluation

The relationship between clustering results and behavioural variables is evaluated using:

- **Canonical Correlation Analysis (CCA)**
- **Multinomial Logistic Regression (MLR)**

These methods are used to investigate whether the cluster assignments correspond to variation in behavioural or questionnaire-based data.

## Data

The original fMRI-derived data and behavioural data are not included in this repository.

This is because the data may be:

- too large for GitHub
- not publicly shareable
- subject to privacy or project restrictions

The code assumes that the required input data is available locally.

Typical local data may include:

```text
Input Data/
Clusters/
Models/
Figures/
Results/
```

These folders are not intended to be committed to Git.

## Expected data format

The project primarily works with functional connectivity matrices stored in MATLAB `.mat` files or NumPy/PyTorch-compatible formats.

The exact expected shape depends on the parcellation used. Examples from the project include FC matrices based on parcellations such as:

- 200 × 200
- 400 × 400
- 1000 × 1000

Some scripts assume specific naming conventions for subjects, runs and parcellations. These may need to be adjusted if the code is used on another dataset.

## Installation

Clone the repository:

```bash
git clone https://github.com/Oddrne/Master-Thesis-DCEC-on-functional-connectivity.git
cd Master-Thesis-DCEC-on-functional-connectivity
```

Create and activate a virtual environment.

Using `venv`:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy scipy pandas scikit-learn matplotlib torch h5py
```

Optional packages used in some parts of the project:

```bash
pip install umap-learn hdbscan
```

If using a GPU, install the correct PyTorch version for your CUDA setup from the official PyTorch installation instructions.

## Suggested `.gitignore`

Large data files, model weights and generated outputs should not be committed to GitHub.

A suitable `.gitignore` may include:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

# Virtual environments
.venv/
venv/
env/

# Data
Input Data/
Data/
*.mat
*.h5
*.npy
*.npz
*.csv

# Model weights
Models/
*.pt
*.pth

# Generated outputs
Clusters/
Figures/
Results/
*.png
*.jpg
*.jpeg
*.pdf

# Editor settings
.vscode/
```

If a data file has already been committed, adding it to `.gitignore` is not enough. It must also be removed from Git tracking using:

```bash
git rm --cached path/to/file
```

## Example usage

A simplified training workflow may look like this:

```python
from Convolutional_AE import (
    DCEC,
    DCECConfig,
    pretrain_cae,
    initialize_cluster_centers,
    train_dcec,
    predict_soft_assignments
)

cfg = DCECConfig(
    name="DCEC_1000x1000_subjects_run1",
    conv_layers_sizes=[1, 32, 64, 128, 256],
    n_clusters=2,
    latent_dim=10,
    alpha=1.0,
    gamma=0.1,
    epochs_pretrain=50,
    epochs_dcec=100,
    lr_pretrain=1e-3,
    lr_dcec=1e-4,
    update_interval=8,
    tol=1e-3,
    print_interval=10
)

model = DCEC(cfg)

pretrain_history = pretrain_cae(
    model=model,
    dataloader=dataloader,
    device="cuda",
    epochs=cfg.epochs_pretrain,
    lr=cfg.lr_pretrain,
    print_interval=cfg.print_interval
)

initialize_cluster_centers(
    model=model,
    dataloader=dataloader,
    device="cuda"
)

dcec_history = train_dcec(
    model=model,
    dataloader=dataloader,
    device="cuda",
    gamma=cfg.gamma,
    epochs=cfg.epochs_dcec,
    lr=cfg.lr_dcec,
    update_interval=cfg.update_interval,
    tol=cfg.tol,
    print_interval=cfg.print_interval
)

q, labels = predict_soft_assignments(
    model=model,
    dataloader=dataloader,
    device="cuda",
    save=True
)
```

This example assumes that a valid PyTorch `DataLoader` named `dataloader` has already been created.

## Outputs

The pipeline can produce several types of output:

- trained model weights
- reconstructed FC matrices
- latent embeddings
- predicted cluster labels
- clustering score plots
- t-SNE or UMAP visualisations
- CCA and MLR evaluation results
- JSON files containing evaluation summaries

Generated outputs should normally be stored locally and excluded from Git.

## Reproducibility notes

This repository contains research code developed during a master’s thesis. Some parts of the code may require manual adjustment depending on:

- local file paths
- input data format
- parcellation size
- number of subjects
- selected fMRI run
- selected behavioural variables
- available hardware

The code is therefore best understood as a research pipeline rather than a fully packaged Python library.

## Known limitations

- The original dataset is not included.
- Some scripts may contain local or project-specific paths.
- Some notebooks are exploratory.
- The code assumes specific file naming conventions in several places.
- Large model and data files should be stored outside the repository.
- Results may vary depending on random seed, cluster number, batch size, parcellation and preprocessing choices.

## Author

**Odd Arne Fosse**

Master’s student in Cybernetics and Robotics  
Norwegian University of Science and Technology (NTNU)

## Thesis context

This repository was developed as part of a master’s thesis on unsupervised deep clustering of functional connectivity matrices from fMRI data.

The broader aim of the thesis is to investigate whether differences between healthy subjects can be captured from functional connectivity patterns, and whether such differences relate to behavioural variables.

