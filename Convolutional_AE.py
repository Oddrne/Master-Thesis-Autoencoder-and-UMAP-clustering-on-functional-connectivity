from dataclasses import dataclass

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from typing import List



# --------------------------------------------------
# 0. Config class for easy hyperparameter management
# --------------------------------------------------
@dataclass
class DCECConfig:
    name: str                                               # Used for saving predicted labels, e.g. "dcec_results"   
    conv_layers_sizes: List[int]                            # Channel sizes for the convolutional layers in the CAE encoder/decoder
    
    n_clusters: int = 3                                     # Number of clusters for the clustering layer
    latent_dim: int = 10                                    # Dimensionality of the middle layer z
    alpha: float = 1.0                                      # Parameter for Student's t-distribution in clustering layer
    gamma: float = 0.1                                      # Weight for clustering loss in the joint DCEC training              
    epochs_pretrain: int = 50                               # Number of epochs for pretraining the CAE (reconstruction only)
    epochs_dcec: int = 100                                  # Number of epochs for joint DCEC training (reconstruction + clustering)    
    lr_pretrain: float = 1e-3                               # Learning rate for pretraining the CAE                
    lr_dcec: float = 1e-4                                   # Learning rate for joint DCEC training
    update_interval: int = 8                                # How often to update the target distribution p_ij during DCEC training
    tol: float = 1e-3                                       # Tolerance for early stopping based on cluster assignment stability during DCEC training                       
    print_interval: int = 10                                # How often to print training progress during pretraining and DCEC training


# --------------------------------------------------
# 2. Clustering layer (Student's t-distribution)
#    Corresponds to q_ij in the paper
# --------------------------------------------------

class ClusteringLayer(nn.Module):
    """
    Cluster centers mu_j as trainable parameters.
    Computes soft assignments q_ij using Student's t-distribution.
    """

    def __init__(self, n_clusters: int, embedding_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha

        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch_size, embedding_dim)
        returns q: (batch_size, n_clusters)
        
        Current: q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_j((1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)) 
        Actual: q_ij = (1 + ||z_i - mu_j||^2 )^(-1) / sum_j((1 + ||z_i - mu_j||^2)^(-1)
            z_i = embedding of sample i
            mu_j = cluster center j
        """
        if self.cluster_centers.shape[0] != self.n_clusters:
            print("Warning: cluster_centers shape is", self.cluster_centers.shape, "but expected (n_clusters, embedding_dim)")
        
        # Squared distance to each cluster center
        # dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)

        z1 = z.unsqueeze(1) - self.cluster_centers # (batch_size, 1, embedding_dim)
        z2 = torch.mul(z1, z1) # (batch_size, n_clusters, embedding_dim)
        x3 = torch.sum(z2, dim=2) # (batch_size, n_clusters)
        z4 = 1.0 + x3
        z5 = 1.0 /z4
        z6 = torch.t(z5) / torch.sum(z5, dim=1) # (n_clusters, batch_size) / (batch_size,) -> (n_clusters, batch_size)
        q = torch.t(z6) # (batch_size, n_clusters)
        
        # Student's t-distribution
        # numerator = (1.0 + dist_sq ) ** (-1.0)
        # q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q

    def set_weight(self, tensor):
        self.cluster_centers = nn.Parameter(tensor)

# --------------------------------------------------
# 3. Full DCEC model - Deep Convolutional Embedded Clustering
# --------------------------------------------------

class DCEC(nn.Module):
    """
    Convolutional Autoencoder with a low-dimensional embedded layer z.
    Assumes input images of shape (1, 200, 200), e.g. FC matrices.
    """
    def __init__(self, cfg: DCECConfig):
        super().__init__()
        self.cfg = cfg
        self.n_clusters = cfg.n_clusters
        self.z = None  # To store the final embeddings after training        
        self.latent_dim = cfg.latent_dim
        self.conv_layers_sizes = cfg.conv_layers_sizes  # Encoder channel sizes

        conv_layers_sizes = cfg.conv_layers_sizes
        if len(conv_layers_sizes) < 5:
            conv_layers_sizes = [1, 32, 64, 128, 256]
                    
        # Encoder
        self.enc_conv1 = nn.Conv2d(conv_layers_sizes[0], conv_layers_sizes[1], kernel_size=6, stride=2, padding=2)   # 400x400x1 -> 200x200x32
        self.enc_conv2 = nn.Conv2d(conv_layers_sizes[1], conv_layers_sizes[2], kernel_size=6, stride=2, padding=2)  # 200x200x32 -> 100x100x64
        self.enc_conv3 = nn.Conv2d(conv_layers_sizes[2], conv_layers_sizes[3], kernel_size=6, stride=2, padding=2) # 100x100x64 -> 50x50x128
        self.enc_conv4 = nn.Conv2d(conv_layers_sizes[3], conv_layers_sizes[4], kernel_size=5, stride=3, padding=3) # 50x50x128 -> 25x25x256
        self.enc_conv5 = nn.Conv2d(conv_layers_sizes[4], conv_layers_sizes[4], kernel_size=4, stride=2, padding=1) # 25x25x256 -> 12x12x256
        self.enc_conv6 = nn.Conv2d(conv_layers_sizes[4], conv_layers_sizes[4], kernel_size=5, stride=3, padding=1) # 12x12x256 -> 6x6x256

        self.flatten = nn.Flatten() # 1x2304
        self.fc_enc = nn.Linear(256 * 3 * 3, self.latent_dim) # 2304 -> latent_dim = 10
        self.clustering = ClusteringLayer(n_clusters=self.n_clusters, embedding_dim=self.latent_dim, alpha=1)
        
        # Normalizations
        self.bn1_1 = nn.BatchNorm2d(conv_layers_sizes[1])
        self.bn1_2 = nn.BatchNorm2d(conv_layers_sizes[2]) 
        self.bn1_3 = nn.BatchNorm2d(conv_layers_sizes[3]) 
        self.bn1_4 = nn.BatchNorm2d(conv_layers_sizes[4]) 
        self.bn1_5 = nn.BatchNorm2d(conv_layers_sizes[4])
        self.gn1_1 = nn.GroupNorm(8, conv_layers_sizes[1]) # 32 channels -> 8 groups
        self.gn1_2 = nn.GroupNorm(8, conv_layers_sizes[2]) # 64 channels -> 16 groups
        self.gn1_3 = nn.GroupNorm(16, conv_layers_sizes[3]) # 128 channels -> 32 groups
        self.gn1_4 = nn.GroupNorm(32, conv_layers_sizes[4]) # 256 channels -> 64 groups
        self.gn1_5 = nn.GroupNorm(64, conv_layers_sizes[4]) # 256 channels -> 64 groups



        # Decoder
        self.fc_dec = nn.Linear(self.latent_dim, 256 * 3 * 3) # latent_dim -> 2304

        self.dec_deconv1 = nn.ConvTranspose2d(conv_layers_sizes[-1], conv_layers_sizes[-1], kernel_size=5, stride=3, padding=1, output_padding=0)  # 3x3x256 -> 6x6x256
        self.dec_deconv2 = nn.ConvTranspose2d(conv_layers_sizes[-1], conv_layers_sizes[-1], kernel_size=4, stride=2, padding=1, output_padding=0)  # 6x6x256 -> 12x12x256
        self.dec_deconv3 = nn.ConvTranspose2d(conv_layers_sizes[-1], conv_layers_sizes[-2], kernel_size=5, stride=3, padding=3, output_padding=0)  # 12x12x256 -> 25x25x128
        self.dec_deconv4 = nn.ConvTranspose2d(conv_layers_sizes[-2], conv_layers_sizes[-3], kernel_size=6, stride=2, padding=2, output_padding=0)  # 25x25x128 -> 50x50x64 (if needed)
        self.dec_deconv5 = nn.ConvTranspose2d(conv_layers_sizes[-3], conv_layers_sizes[-4], kernel_size=6, stride=2, padding=2, output_padding=0)  # 50x50x64 -> 100x100x32 (if needed)
        self.dec_deconv6 = nn.ConvTranspose2d(conv_layers_sizes[-4], conv_layers_sizes[-5], kernel_size=6, stride=2, padding=2, output_padding=0)  # 100x100x32 -> 200x200x1 (if needed)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_conv1(x)
        x = self.gn1_1(x)
        x = F.relu(x)
        x = self.enc_conv2(x)
        x = self.gn1_2(x)
        x = F.relu(x)
        x = self.enc_conv3(x)
        x = self.gn1_3(x)
        x = F.relu(x)
        x = self.enc_conv4(x)
        x = self.gn1_4(x)
        x = F.relu(x)
        x = self.enc_conv5(x)
        x = self.gn1_5(x)
        x = F.relu(x)
        x = self.enc_conv6(x)
        x = F.relu(x)
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_dec(z)
        x = x.view(-1, self.conv_layers_sizes[-1], 3, 3)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        x = F.relu(self.dec_deconv4(x))
        x = F.relu(self.dec_deconv5(x))
        x_hat = torch.tanh(self.dec_deconv6(x))
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        q = self.clustering(z)
        x_hat = self.decode(z)
        
        return x_hat, q, z # Return the reconstructed x, the soft cluster assignments q, and the latent embeddings z



# --------------------------------------------------
# 4. Target distribution p_ij from q_ij
#    Matches the target distribution idea in the paper
# --------------------------------------------------

def target_distribution(q: np.ndarray) -> np.ndarray:
    """
    p_ij = (q_ij^2 / f_j) / sum_j(q_ij^2 / f_j)
    where f_j = sum_i q_ij
    
    q: (N_samples, n_clusters)
    returns p: (N_samples, n_clusters)
    """
    weight = q ** 2 / np.sum(q, axis=0)
    #p = weight / torch.sum(weight, dim=1, keepdim=True)
    p = np.transpose(np.transpose(weight) / np.sum(weight, axis=1))
    return p


# --------------------------------------------------
# 5. Utilities
# --------------------------------------------------

@torch.no_grad()
def extract_embeddings(model: DCEC, dataloader: DataLoader, device: str) -> np.ndarray:
    """ 
    Extract the latent embeddings z_i for all samples in the dataloader using the CAE encoder.
    Returns:
    - z_all: A numpy array of shape (N_samples, latent_dim) containing the embeddings for all samples.
    """
    was_training = model.training # Save if the model was in training mode
    model.eval()
    zs = []

    for (x,) in dataloader:
        x = x.to(device)
        z = model.encode(x)
        zs.append(z.cpu().numpy())

    if was_training:
        model.train() # Restore the original training mode of the model
        
    return np.concatenate(zs, axis=0)


@torch.no_grad() # No need to compute gradients when predicting cluster assignmentse
def predict_soft_assignments(model: DCEC, dataloader: DataLoader, device: str, save=False) -> Tuple[np.ndarray, np.ndarray]:
    """
        Predict the soft cluster assignments q_ij and the corresponding hard labels for all samples in the dataloader.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing: [q_final, labels]
    """
    was_training = model.training # Save if the model was in training mode
    
    model.eval()    
    qs = None
    zs = None

    for (x,) in dataloader:
        x = x.to(device) # x = input
        x_hat, q, z = model(x) # z 
        
        if qs is not None:
            zs = np.concatenate((zs, z.cpu().detach().numpy()), axis=0) # z_array
            qs = np.concatenate((qs, q.cpu().detach().numpy()), axis=0) # q_array = output_array
        else:
            zs = z.cpu().detach().numpy()
            qs = q.cpu().detach().numpy()

    labels = np.argmax(qs.data, axis=1) # preds
    
    if save:
        model.z = zs  # Store the final embeddings in the model for later use
        model_name = model.cfg.name
        
        # Count how many samples are assigned to each cluster and save to a dictionary
        label_counts = {}
        for label in np.unique(labels):
            count = (labels == label).sum()
            label_counts[f"label_{label}"] = count
        # Create a filename with the model name and label counts
        filename_labels = f"Clusters\\{model_name}_cluster_{model.n_clusters}_labels_predicted_labels_" + ".txt" # + "_".join([f"{name}_{count}" for name, count in label_counts.items()])
        filename_midlayer = f"Clusters\\{model_name}_cluster_{model.n_clusters}_middle_layer_predicted_labels_"  + ".txt" # + "_".join([f"{name}_{count}" for name, count in label_counts.items()])
        
        # Save the predicted labels to a text file
        np.savetxt(filename_labels, labels, fmt="%d")
        np.savetxt(filename_midlayer, zs, fmt="%.6f")
    
    if was_training:
        model.train() # Restore the original training mode of the model
        
    return qs, labels


def initialize_cluster_centers(
    model: DCEC,
    dataloader: DataLoader,
    device: str
):
    """
    Pretrain CAE, then run k-means on embeddings z_i, then load centers into clustering layer.
    """
    model.eval()
    
    z_all = extract_embeddings(model, dataloader, device)
    kmeans = KMeans(n_clusters=model.n_clusters, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z_all)

    centers = torch.from_numpy(kmeans.cluster_centers_)
    # Check if KMeans returns fewwer centers than n_clusters
    if centers.shape[0] < model.n_clusters:
        print(f"Warning: KMeans returned {centers.shape[0]} centers, but expected {model.n_clusters}. Check the input data and KMeans parameters.")
        print(f"Z_all shape: {z_all.shape}. Unique predicted labels: {np.unique(y_pred)}. Counts: {np.bincount(y_pred)}.")
        # If fewer centers are returned, we can pad with random centers from the existing ones
        raise ValueError("KMeans did not return the expected number of cluster centers. Check the input data and KMeans parameters.")
    
    model.clustering.set_weight(centers.to(device))

    return y_pred

def make_upper_triangle_mask(n=200, include_diagonal=False, device='cpu'):
    if include_diagonal:
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=0)
    else:
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    return mask

def masked_mse_loss(x_hat, x, mask):
    """Finds the mse-loss of the upper triangle

    Args:
        x_hat (_type_): The reconstructed FC matrix (batch_size, n, n)
        x (_type_): The original FC matrix (batch_size, n, n)
        mask (_type_): A boolean mask for the upper triangle (n, n)

    Returns:
        _type_: (x_hat - x)^2 averaged over the valid elements in the upper triangle and batch size
    """
    diff2 = ((x_hat - x) ** 2) * mask
    return diff2.sum() / mask.sum() / x.shape[0]  # Normalize by number of valid elements and batch size


# --------------------------------------------------
# 6. Pretraining of CAE (reconstruction only)
# --------------------------------------------------

def pretrain_cae(
    model: DCEC,
    dataloader: DataLoader,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 1e-2,
    print_interval: int = 10
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mask = make_upper_triangle_mask(n=400, include_diagonal=False, device=device).unsqueeze(0)  # Shape (1, 400, 400)
    #mse_loss = nn.MSELoss()

    
    model.to(device)

    print_pause = epochs // print_interval
    history = {"Recon loss": []}
    
    if print_pause == 0:
        raise ValueError("print_interval is too large for the number of epochs. Please set print_interval to a smaller value.")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_var = 0.0

        for (x,) in dataloader:
            x = x.to(device)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                x_hat, q, z = model(x)
                loss_rec = masked_mse_loss(x_hat, x, mask)
                #loss_var = -torch.var(z, dim=0).mean()  # Variance regularization to encourage non-collapsed embeddings
                loss = loss_rec #+ 1 * loss_var  # Total loss with variance
                loss.backward()
                optimizer.step()

            # running_loss_var += loss_var.item() * x.size(0)
            running_loss_rec += loss_rec.item() * x.size(0)
            running_loss += loss.item() * x.size(0)

        # print(f"Epoch {epoch+1}, Batch loss: {running_loss/len(dataloader.dataset):.6f}, Recon loss: {running_loss_rec/len(dataloader.dataset):.6f}, Var loss: {running_loss_var/len(dataloader.dataset):.6f}")
        epoch_loss = running_loss / len(dataloader.dataset)
        history["Recon loss"].append(epoch_loss)
        
        if (epoch + 1) % print_pause == 0 or epoch == 0:
            print(f"[Pretrain] Epoch {epoch+1:03d}/{epochs} - Recon loss: {epoch_loss:.6f}")

    return history

# --------------------------------------------------
# 7. Joint DCEC training
# --------------------------------------------------

def train_dcec(
    model: DCEC,
    dataloader: DataLoader,
    device: str = "cuda",
    gamma: float = 0.1,
    epochs: int = 100,
    lr: float = 1e-4,
    update_interval: int = 8, # For 9 bacthes, update every 3 batches means updating 3 times per epoch
    tol: float = 1e-3,
    print_interval: int = 10 #,
    # y_pred_initial: np.ndarray = None
):
    """
    Joint optimization of:
        L = L_r + gamma * L_c
    where
        L_r = MSE(x_hat, x)
        L_c = KL(P || Q)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mask = make_upper_triangle_mask(n=200, include_diagonal=False, device=device).unsqueeze(0)  # Shape (1, 200, 200)

    model.to(device)
    
    print_pause = epochs // print_interval
    if print_pause == 0:
        raise ValueError("print_interval is too large for the number of epochs. Please set print_interval to a smaller value.")

    history = {
        "Total loss": [], 
        "Recon loss": [],
        "KL loss": [],
        "Label change fraction": []}
    
    
    # Initial q/p and label estimate
    q_all, y_pred_last = predict_soft_assignments(model, dataloader, device)
    # q_all = torch.tensor(q_all, dtype=torch.float32, device=device)
    p_all = target_distribution(q_all)
    # y_pred_last = y_pred_initial # Take the initial predicted labels from KMeans as the starting point for tracking label changes.
    q_all_last = None # To track changes in q_all for early stopping
    
    finished = False
    update_number = 0
    low_updates = 0
    
    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0        
        
        
        for batch_num, (x,) in enumerate(dataloader): # batch_num starts at 0, but we want to start at 1 for the update_interval logic
            x = x.to(device)
            batch_size = x.size(0)
            update_number += 1
            
            # Update target distribution every few batches
            if (update_number) % update_interval == 0:   # Should update at batch 3, 6, 9 for batch_num starting at 1
                q_all, y_pred = predict_soft_assignments(model, dataloader, device)
                # q_all = torch.tensor(q_all_np, dtype=torch.float32, device=device)
                p_all = target_distribution(q_all)

                delta_label = np.mean(y_pred != y_pred_last)
                if q_all_last is None:
                    delta_q = np.inf
                else:
                    delta_q = np.mean(np.abs(q_all - q_all_last))
                    
                #print(f"[DCEC] Epoch {epoch:03d}/{epochs} Batch {batch_num+1:03d}/{len(dataloader)} Update {update_number} \n label change fraction: {delta_label:.6f} - Delta q: {delta_q if q_all_last is not None else 'N/A'}")
                 
                y_pred_last = np.copy(y_pred)
                q_all_last = np.copy(q_all)
                history["Label change fraction"].append(delta_label)

                
                if delta_q < tol and epoch > 0:
                    low_updates += 1
                else:
                    low_updates = 0
                    
                if low_updates > 2:
                    print("Stopping early: cluster assignments stabilized. Delta:", delta_q, "< Tol:", tol)
                    finished = True
                    break
                
            start_idx = (batch_num) * dataloader.batch_size
            end_idx = start_idx + batch_size
            p_batch = p_all[start_idx:end_idx, :]
            
            optimizer.zero_grad() # Zero gradients before backward pass

            with torch.set_grad_enabled(True):
                x_hat, q_batch, z_batch = model(x)
                p_batch = torch.tensor(p_batch, dtype=torch.float32, device=device)
                lr_loss = masked_mse_loss(x_hat, x, mask) # Reconstruction loss
                lc_loss = F.kl_div(torch.log(q_batch), p_batch, reduction="batchmean") # Clustering loss (KL divergence)
                loss = lr_loss + gamma * lc_loss # Total loss
                loss.backward()
                optimizer.step()

            if lc_loss.item() * batch_size < 0:
                print(f"KL loss: {lc_loss.item()}. q_batch: {q_batch}. p_batch: {p_batch}. This should not happen. Check the training process.") 
            
            running_total += loss.item() * batch_size
            running_recon += lr_loss.item() * batch_size
            running_kl += gamma * lc_loss.item() * batch_size
        
            
        n = len(dataloader.dataset)
        epoch_total_loss = running_total / n
        epoch_recon_loss = running_recon / n
        epoch_kl_loss =  running_kl / n
        
        history["Total loss"].append(epoch_total_loss)
        history["Recon loss"].append(epoch_recon_loss)
        history["KL loss"].append(epoch_kl_loss)
        
        if (epoch + 1) % print_pause == 0 or epoch == 0:
            print(
                f"[DCEC] Epoch {epoch+1:03d}/{epochs} - "
                f"Total: {epoch_total_loss:.6f}, "
                f"Recon: {epoch_recon_loss:.6f}, "
                f"KL: {epoch_kl_loss:.6f}"
            )
            """ if epoch_kl_loss <= 0:
                raise ValueError(f"KL loss is non-positive: {running_kl}. This should not happen. Check the training process.")
         """
        if finished:
            print("Stopped at epoch", epoch+1, "and batch", batch_num+1, "after", update_number, "updates.")
            break
            
    return history
        
        
# --------------------------------------------------
# 8. Visualization utilities
# --------------------------------------------------        
        
def plot_training_history(pretrain_history=None, dcec_history=None, save_path=None):
    plt.figure(figsize=(10, 6))
    
    if pretrain_history is not None:
        plt.plot(pretrain_history["Recon loss"], label="CAE Pretrain Recon Loss", color='blue')
    
    if dcec_history is not None:
        plt.plot(dcec_history["Total loss"], label="DCEC Total Loss", color='red')
        plt.plot(dcec_history["Recon loss"], label="DCEC Recon Loss", color='orange')
        plt.plot(dcec_history["KL loss"], label="DCEC KL Loss", color='green')
        
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
@torch.no_grad()
def plot_reconstruction(model: DCEC, dataloader: DataLoader, device: str, n_samples=5, save_path=None):
    """
    Plots original vs reconstructed FC matrices for a few samples from the dataloader.
    """
    # model.eval() #
    x_batch = next(iter(dataloader))[0][:n_samples].to(device)  # Get a batch of samples and limit to n_samples
    x_hat, q, z = model(x_batch)
    
    # Plot original and reconstructed FC matrices
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
    
    recon_losses = []
    for i in range(n_samples):
        loss = masked_mse_loss(
            x_hat=torch.tensor(x_hat[i:i+1, 0]),
            x=torch.tensor(x_batch[i, 0]), 
            mask=make_upper_triangle_mask(n=x_batch.shape[-1], include_diagonal=False, device=device).unsqueeze(0)
            ).item()
        recon_losses.append(loss)

    x_batch = x_batch.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

    if n_samples == 1:
        axes = np.array([axes])  # Ensure axes is 2D even for single sample
    
    for i in range(n_samples):
        subject_id = f"Subject {i}"
        
        axes[i, 0].imshow(x_batch[i, 0], aspect='auto')
        axes[i, 0].set_title(f"{subject_id} - Original FC")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(x_hat[i, 0], aspect='auto') 
        axes[i, 1].set_title(f"{subject_id} - Reconstructed FC\nLoss: {recon_losses[i]:.4f}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    
def plot_clustering(model: DCEC, dataloader: DataLoader, device: str, save_path=None):
    z_all = extract_embeddings(model, dataloader, device)
    _, labels = predict_soft_assignments(model, dataloader, device, save=True)
    
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    
    tsne = TSNE(
        random_state=42
        )
    z_2d = tsne.fit_transform(z_all)
    
    cluster_number = model.n_clusters
    
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=20, c=labels)
    plt.title(f"t-SNE Visualization of Clusters {cluster_number}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.text(0.95, 0.01, f"Model: {model.cfg.name} has {np.unique(labels).size} clusters", ha='right', va='bottom', transform=plt.gcf().transFigure, fontsize=8)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    
