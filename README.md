# Deep Convolutional Embedded Clustering on Functional Connectivity

This repository contains code developed for a master’s thesis exploring whether Deep Convolutional Embedded Clustering (DCEC) can identify meaningful structure in functional connectivity (FC) matrices derived from fMRI data.

The project uses convolutional autoencoders to learn low-dimensional representations of FC matrices, followed by embedded clustering in the latent space. The resulting clusters are evaluated both internally, using standard clustering metrics, and externally, by comparing cluster membership with behavioural variables.

## Project overview

The main goal of this project is to investigate whether unsupervised deep clustering can reveal differences between healthy subjects based on functional connectivity patterns.

The pipeline consists of:

1. Loading functional connectivity matrices
2. Training a convolutional autoencoder (CAE)
3. Initialising cluster centres using KMeans on the latent embeddings
4. Jointly training the DCEC model using reconstruction loss and clustering loss
5. Saving predicted cluster labels and latent embeddings
6. Evaluating clustering quality
7. Comparing clusters with behavioural data using CCA and multinomial logistic regression

## Repository structure

```text
.
├── Convolutional_AE.py        # DCEC model, training functions and plotting utilities
├── Evaluate_models.py         # Clustering evaluation and result plotting
├── README.md                  # Project documentation
├── test.ipynb                 # Notebook for testing/development
└── .vscode/                   # VS Code settings