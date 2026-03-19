import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


# --------------------------------------------------
# 1. Convolutional Autoencoder
# --------------------------------------------------

class CAE(nn.Module):
    """
    Convolutional Autoencoder with a low-dimensional embedded layer z.
    Assumes input images of shape (1, 28, 28), e.g. MNIST.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        
        conv_layers_sizes = [1, 32, 64, 128, 256]  # Encoder channel sizes
        self.conv_layers_sizes = conv_layers_sizes

        # Encoder
        self.enc_conv1 = nn.Conv2d(conv_layers_sizes[0], conv_layers_sizes[1], kernel_size=5, stride=2, padding=2)   # 200x200x1 -> 100x100x32
        self.enc_conv2 = nn.Conv2d(conv_layers_sizes[1], conv_layers_sizes[2], kernel_size=5, stride=2, padding=2)  # 100x100x32 -> 50x50x64
        self.enc_conv3 = nn.Conv2d(conv_layers_sizes[2], conv_layers_sizes[3], kernel_size=5, stride=2, padding=2) # 50x50x64 -> 25x25x128
        self.enc_conv4 = nn.Conv2d(conv_layers_sizes[3], conv_layers_sizes[4], kernel_size=3, stride=2, padding=0) # 25x25x128 -> 12x12x256
        self.enc_conv5 = nn.Conv2d(conv_layers_sizes[4], conv_layers_sizes[4], kernel_size=4, stride=2, padding=1) # 12x12x256 -> 6x6x256
        self.enc_conv6 = nn.Conv2d(conv_layers_sizes[4], conv_layers_sizes[4], kernel_size=4, stride=2, padding=1) # 6x6x256 -> 3x3x256

        self.flatten = nn.Flatten() # 1x1152
        self.fc_enc = nn.Linear(256 * 3 * 3, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 3 * 3)

        self.dec_deconv1 = nn.ConvTranspose2d(conv_layers_sizes[-1], conv_layers_sizes[-1], kernel_size=4, stride=2, padding=1, output_padding=0)  # 3x3x256 -> 6x6x256
        self.dec_deconv2 = nn.ConvTranspose2d(conv_layers_sizes[-1], conv_layers_sizes[-1], kernel_size=4, stride=2, padding=1, output_padding=0)  # 6x6x256 -> 12x12x256
        self.dec_deconv3 = nn.ConvTranspose2d(conv_layers_sizes[-1], conv_layers_sizes[-2], kernel_size=3, stride=2, padding=0, output_padding=0)  # 12x12x256 -> 25x25x128
        self.dec_deconv4 = nn.ConvTranspose2d(conv_layers_sizes[-2], conv_layers_sizes[-3], kernel_size=5, stride=2, padding=2, output_padding=1)  # 25x25x128 -> 50x50x64 (if needed)
        self.dec_deconv5 = nn.ConvTranspose2d(conv_layers_sizes[-3], conv_layers_sizes[-4], kernel_size=5, stride=2, padding=2, output_padding=1)  # 50x50x64 -> 100x100x32 (if needed)
        self.dec_deconv6 = nn.ConvTranspose2d(conv_layers_sizes[-4], conv_layers_sizes[-5], kernel_size=5, stride=2, padding=2, output_padding=1)  # 100x100x32 -> 200x200x1 (if needed)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = F.relu(self.enc_conv5(x))
        x = F.relu(self.enc_conv6(x))
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
        x = torch.tanh(self.dec_deconv6(x))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


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
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch_size, embedding_dim)
        returns q: (batch_size, n_clusters)
        
        Current: q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_j((1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)) 
        Actual: q_ij = (1 + ||z_i - mu_j||^2 )^(-1) / sum_j((1 + ||z_i - mu_j||^2)^(-1)
            z_i = embedding of sample i
            mu_j = cluster center j
        """
        if self.cluster_centers.shape[0] != 4:
            print("Warning: cluster_centers shape is", self.cluster_centers.shape, "but expected (n_clusters, embedding_dim)")
        
        # Squared distance to each cluster center
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)

        # Student's t-distribution
        numerator = (1.0 + dist_sq ) ** (-1.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q


# --------------------------------------------------
# 3. Full DCEC model - Deep Convolutional Embedded Clustering
# --------------------------------------------------

class DCEC(nn.Module):
    def __init__(self, n_clusters: int = 10, latent_dim: int = 10, alpha: float = 1.0):
        super().__init__()
        self.cae = CAE(latent_dim=latent_dim)
        self.n_clusters = n_clusters
        self.clustering = ClusteringLayer(
            n_clusters=n_clusters,
            embedding_dim=latent_dim,
            alpha=alpha
        )

    def forward(self, x: torch.Tensor):
        x_hat, z = self.cae(x)
        q = self.clustering(z)
        return x_hat, z, q


# --------------------------------------------------
# 4. Target distribution p_ij from q_ij
#    Matches the target distribution idea in the paper
# --------------------------------------------------

def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    p_ij = (q_ij^2 / f_j) / sum_j(q_ij^2 / f_j)
    where f_j = sum_i q_ij
    """
    weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
    p = weight / torch.sum(weight, dim=1, keepdim=True)
    return p


# --------------------------------------------------
# 5. Utilities
# --------------------------------------------------

@torch.no_grad()
def extract_embeddings(model: DCEC, dataloader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    zs = []

    for (x,) in dataloader:
        x = x.to(device)
        z = model.cae.encode(x)
        zs.append(z.cpu().numpy())

    return np.concatenate(zs, axis=0)


@torch.no_grad()
def predict_soft_assignments(model: DCEC, dataloader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    qs = []

    for (x,) in dataloader:
        x = x.to(device)
        z = model.cae.encode(x)
        q = model.clustering(z)
        qs.append(q.cpu().numpy())

    return np.concatenate(qs, axis=0)


def initialize_cluster_centers(
    model: DCEC,
    dataloader: DataLoader,
    device: str
):
    """
    Pretrain CAE, then run k-means on embeddings z_i, then load centers into clustering layer.
    """
    z_all = extract_embeddings(model, dataloader, device)
    kmeans = KMeans(n_clusters=model.n_clusters, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z_all)

    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    model.clustering.cluster_centers.data.copy_(centers)

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
    lr: float = 1e-3,
    print_interval: int = 10
):
    optimizer = torch.optim.Adam(model.cae.parameters(), lr=lr)
    mask = make_upper_triangle_mask(n=200, include_diagonal=False, device=device).unsqueeze(0)  # Shape (1, 200, 200)
    #mse_loss = nn.MSELoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for (x,) in dataloader:
            x = x.to(device)

            x_hat, _ = model.cae(x)
            loss = masked_mse_loss(x_hat, x, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"[Pretrain] Epoch {epoch+1:03d}/{epochs} - Recon loss: {epoch_loss:.6f}")


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
    update_interval: int = 5,
    tol: float = 1e-3,
    print_interval: int = 20
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

    #mse_loss = nn.MSELoss()

    model.to(device)

    # Initial q/p and label estimate
    q_all = predict_soft_assignments(model, dataloader, device)
    y_pred_last = q_all.argmax(axis=1)

    for epoch in range(epochs):
        # Update target distribution every few epochs
        if epoch % update_interval == 0:
            q_all = predict_soft_assignments(model, dataloader, device)
            q_tensor = torch.tensor(q_all, dtype=torch.float32, device=device)
            p_all = target_distribution(q_tensor).cpu().numpy()

            y_pred = q_all.argmax(axis=1)
            delta_label = np.mean(y_pred != y_pred_last)
            y_pred_last = y_pred.copy()

            print(
                f"[DCEC] Epoch {epoch:03d}/{epochs} - label change fraction: {delta_label:.6f}"
            )

            if epoch > 0 and delta_label < tol:
                print("Stopping early: cluster assignments stabilized. Delta:", delta_label, "< Tol:", tol)
                break

        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0

        start_idx = 0
        for (x,) in dataloader:
            batch_size = x.size(0)
            x = x.to(device)

            p_batch = torch.tensor(
                p_all[start_idx:start_idx + batch_size],
                dtype=torch.float32,
                device=device
            )
            start_idx += batch_size

            x_hat, z, q_batch = model(x)

            # Reconstruction loss
            lr_loss = masked_mse_loss(x_hat, x, mask)

            # KL(P || Q)
            # F.kl_div expects log-probs as first input
            lc_loss = F.kl_div(torch.log(q_batch + 1e-10), p_batch, reduction="batchmean")

            loss = lr_loss + gamma * lc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total += loss.item() * batch_size
            running_recon += lr_loss.item() * batch_size
            running_kl += lc_loss.item() * batch_size

        n = len(dataloader.dataset)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(
                f"[DCEC] Epoch {epoch+1:03d}/{epochs} - "
                f"Total: {running_total/n:.6f}, "
                f"Recon: {running_recon/n:.6f}, "
                f"KL: {running_kl/n:.6f}"
            )


# --------------------------------------------------
# 8. Example usage
# --------------------------------------------------

if __name__ == "__main__":
    # Example dummy data:
    # Replace this with your real image tensor of shape (N, 1, 28, 28)
    X = torch.rand(2000, 1, 28, 28)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DCEC(n_clusters=10, latent_dim=10, alpha=1.0)

    # Step 1: Pretrain CAE
    pretrain_cae(model, dataloader, device=device, epochs=20, lr=1e-3)

    # Step 2: Initialize clusters with k-means on latent embeddings
    initialize_cluster_centers(model, dataloader, device=device)

    # Step 3: Joint train DCEC
    train_dcec(
        model,
        dataloader,
        device=device,
        gamma=0.1,
        epochs=50,
        lr=1e-4,
        update_interval=5,
        tol=1e-3
    )

    # Final cluster assignments
    q_final = predict_soft_assignments(model, dataloader, device)
    labels = q_final.argmax(axis=1)
    print("Final labels shape:", labels.shape)