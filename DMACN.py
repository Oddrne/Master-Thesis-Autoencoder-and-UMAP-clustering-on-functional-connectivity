# Implementation of DMACN gathered from "Deep multi-kernel auto-encoder network for clustering brain functional connectivity data"
# https://www.sciencedirect.com/science/article/pii/S0893608020304226#fd10

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import polynomial_kernel


# --------------------------
# Kernels (build K NxN from Y NxD)
# --------------------------
def gaussian_kernel_matrix(X: torch.Tensor, t0: float, eps: float = 1e-12) -> torch.Tensor:
    # Gaussian Kernel. Should handle multi-dimensional data. 
    # K_ij(x_i, x_j) = exp( -(x_i*x_j)^T(x_i*x_i) / 2t^2 )
    
    # We have problems with large values leading to inf in K. Implementing normalization
    # X1 = torch.nn.functional.normalize(X, p=2, dim=1)  # Normalize rows to unit length to prevent large values
    
    pairwise_dists = torch.cdist(X, X, p=2) ** 2  # [N,N] squared Euclidean distances
    
    # t = D0*t0 where D0 is the maximum distance between the samples
    D0 = torch.sqrt(torch.clamp(pairwise_dists.max(), min=eps)) 
    t = torch.clamp(torch.as_tensor(t0, device=X.device, dtype=X.dtype) * D0, min=eps)
    
    K = torch.exp(-pairwise_dists / (2.0 * t ** 2))
    # Check if K is inf
    if not torch.isfinite(K).all():
        print("X:", X, "K:", K)
        raise ValueError("Gaussian kernel K contains Inf or NaN values. Check for numerical issues.")
    return K


def poly_kernel_matrix(X: torch.Tensor, a = 0, b = 2, eps: float = 1e-12) -> torch.Tensor:
    # Polynomial Kernel
    # K_ij(x_i, x_j) = (a + x_i^T*x_j)^b

    # We have problems with large values leading to inf in K. Implementing normalization
    # X = torch.nn.functional.normalize(X, p=2, dim=1)  # Normalize rows to unit length to prevent large values


    X_2 = X @ X.T  # [N,N] pairwise dot products
    a_X_2 = X_2 + torch.as_tensor(a, device=X.device, dtype=X.dtype)
    
    K = a_X_2 ** b
        
    if not torch.isfinite(K).all():
        print("X:", X, "K:", K)
        raise ValueError("Polynomial kernel K contains Inf or NaN values. Check for numerical issues.")
    
    return K


def build_kernel_matrix(Y: torch.Tensor, spec: Dict) -> torch.Tensor:
    kind = spec["kind"]
    if kind == "rbf":
        return gaussian_kernel_matrix(Y, t0=float(spec.get("t0", 1.0)))
    if kind == "poly":
        return poly_kernel_matrix(
            Y,
            a=int(spec.get("a", 0)),
            b=int(spec.get("b", 2)),
        )
    raise ValueError(f"Unknown kernel kind: {kind}")


# --------------------------
# Algorithm 2 (MKFC): Eq. (17)(18)(19)(23)
# MKFC: Multi-Kernel Fuzzy Clustering. Iteratively update u, omega, D until convergence.
# --------------------------
def compute_Z_sr(K: torch.Tensor, u: torch.Tensor, m_fuzz: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Eq. (18) RKHS distance to fuzzy centroid:
      Z_{s,r} = K_ii - 2 ubar_c^T K + ubar_j^T K ubar_c
    K: [N,N], u: [N,C] -> Z: [N,C]
    """
    # RKHS is a Reproducing Kernel Hilbert Space, where the kernel function K implicitly 
    # defines a mapping of data points into a high-dimensional space. The distance Z_{i,c} 
    # measures how far each data point i is from the fuzzy centroid of cluster c in this RKHS, 
    # which is crucial for the fuzzy clustersing process.

    um = torch.clamp(u, min=eps) ** m_fuzz                 # [N,C]
    denom = um.sum(dim=0, keepdim=True)                    # [1,C]
    ubar = um / denom                                      # [N,C]

    K_diag = torch.diagonal(K, 0)                          # K_ii [N]
    Ku = K @ ubar                                          # ubar^T * K_:i [N,C]
    uKu = ubar.T @ Ku                                      # [C,C]
    uKu_diag = torch.diagonal(uKu, 0)                      # [C]

    Z = K_diag[:, None] - 2.0 * Ku + uKu_diag[None, :]
    return torch.clamp(Z, min=eps)


def compute_D(Z_list: List[List[torch.Tensor]], omega: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Eq. (19): D_{i,c} = sum_{s=1..mid} sum_{r=1..h} omega_{s,r}^2 * Z_{i,c}^{(s,r)}
    Z_list: [mid][h] each [N,C], omega: [mid,h] -> D [N,C]
    """
    mid = len(Z_list)
    h = len(Z_list[0])
    N, C = Z_list[0][0].shape
    D = torch.zeros((N, C), device=Z_list[0][0].device, dtype=Z_list[0][0].dtype)
    for s in range(mid):
        for r in range(h):
            D = D + (omega[s, r] ** 2) * Z_list[s][r]
    return torch.clamp(D, min=eps)


def update_u(D: torch.Tensor, m_fuzz: float = 1.08, eps: float = 1e-12) -> torch.Tensor:
    """
    Eq. (17):
      u_ic = 1 / sum_{c'} (D_ic / D_i,c')^(1/(m-1))
    """
    if m_fuzz <= 1.0:
        raise ValueError("m_fuzz must be > 1.")
    power = 1.0 / (m_fuzz - 1.0)
    # D = torch.clamp(D, min=eps)
    ratio = (D[:, :, None] / D[:, None, :]) ** power      # [N,C,C]
    u = 1.0 / torch.clamp(ratio.sum(dim=2), min=eps)      # [N,C]
    return torch.clamp(u, min=eps, max=1.0)


def update_omega(
    Z_list: List[List[torch.Tensor]],
    u: torch.Tensor,
    m_fuzz: float,
    eps: float = 1e-12,
    renormalize_sum1: bool = True,
) -> torch.Tensor:
    """
    Eq. (23) as written:
      A_{s,r} = sum_{i,c} u_ic^m * Z_{i,c}^{(s,r)}
      omega_{s,r} = (1/A_{s,r}) / ( 2 * sum_{s,r} (1/A_{s,r}) )

    Paper also states constraint sum omega = 1, but Eq(23) with the "2" gives sum=0.5.
    We therefore optionally renormalize omega to sum=1 for stability/consistency.
    """
    mid = len(Z_list)
    h = len(Z_list[0])
    um = torch.clamp(u, min=eps) ** m_fuzz

    invA = torch.zeros((mid, h), device=u.device, dtype=u.dtype)
    for s in range(mid):
        for r in range(h):
            A_sr = torch.sum(um * Z_list[s][r])
            invA[s, r] = 1.0 / torch.clamp(A_sr, min=eps)

    omega = invA / torch.clamp(2.0 * invA.sum(), min=eps)
    omega = torch.clamp(omega, min=eps)

    if renormalize_sum1:
        omega = omega / torch.clamp(omega.sum(), min=eps)

    return omega

def compute_J2(
    Y_layers: List[torch.Tensor],
    u: torch.Tensor,
    omega: torch.Tensor,
    cfg: DMACNConfig,
    kernel_specs: List[Dict],
    m_fuzz: float
) -> torch.Tensor:
    """
    Compute the J2 term in the DMACN loss, which is:
      J2 = lam1/2 * sum_{i,c} u_ic^m * D_{i,c}
    """
    # First find K using Y_layers and kernel_specs, then Z_list using K and u, then D using omega and Z_list, and finally J2 using u, D.
    K_list = [[None for _ in range(len(kernel_specs))] for _ in range(len(Y_layers))]  # type: ignore
    for s in range(len(Y_layers)):
        for r in range(len(kernel_specs)):
            K_list[s][r] = build_kernel_matrix(Y_layers[s], kernel_specs[r])  # [N,N]
    
    # Find Z_list using K and u
    Z_list = [[None for _ in range(len(kernel_specs))] for _ in range(len(Y_layers))]  # type: ignore
    for s in range(len(Y_layers)):
        for r in range(len(kernel_specs)):
            Z_list[s][r] = compute_Z_sr(K_list[s][r], u, m_fuzz=m_fuzz)  # [N,C]
    
    # Lastly find D using omega and Z_list.
    D = compute_D(Z_list, omega)  # [N,C]
    
    J2 = 0.5 * cfg.lam1 * torch.linalg.vector_norm(D * (u ** m_fuzz), ord=2)  # sum_{i,c} u_ic^m * D_{i,c}
    return J2


@torch.no_grad()
def algorithm2_mkfc(
    Y_layers: List[torch.Tensor],          # length mid, each [N, d_s]
    kernel_specs: List[Dict],              # length h
    C: int,
    m_fuzz: float = 1.08,
    eps_stop: float = 1e-5,
    max_iters: int = 50,
    input_u: Optional[torch.Tensor] = None,  # [N,C] or None for uniform init
    renormalize_omega_sum1: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      u:     [N,C]
      omega: [mid,h]
    """
    device = Y_layers[0].device
    dtype = Y_layers[0].dtype
    mid = len(Y_layers)
    h = len(kernel_specs)
    N = Y_layers[0].shape[0]

    if input_u is not None: # We have input_u
        if input_u.shape != (N, C):
            raise ValueError(f"input_u must have shape [N,C], got {input_u.shape}")
        u = torch.clamp(input_u.to(device=device, dtype=dtype), min=1e-12)
    else: # We don't have input_u, initialize uniformly
        u = torch.rand((N, C), device=device, dtype=dtype)
        u = u / (u.sum(dim=1, keepdim=True))  # Normalize rows to sum to 1
        
    omega = torch.full((mid, h), 1.0 / (mid * h), device=device, dtype=dtype)
    if renormalize_omega_sum1:
        omega = omega / omega.sum()

    # Precompute kernel matrices K^{(s,r)} for current Y_layers
    K: List[List[torch.Tensor]] = [[None for _ in range(h)] for _ in range(mid)]  # type: ignore

    for s in range(mid):
        for r in range(h):
            K[s][r] = build_kernel_matrix(Y_layers[s], kernel_specs[r])  # [N,N]


    for _iter in range(max_iters):
        u_prev = u

        # Eq (18): Z_list[s][r] is [N,C]
        Z_list: List[List[torch.Tensor]] = [[None for _ in range(h)] for _ in range(mid)]  # type: ignore
        for s in range(mid):
            for r in range(h):
                Z_list[s][r] = compute_Z_sr(K[s][r], u, m_fuzz=m_fuzz)

        if any(z.isnan().any() for z_list in Z_list for z in z_list):
            print("Z_list contains NaN values. Check for numerical issues.")
            for s, z_list in enumerate(Z_list):
                for r, z in enumerate(z_list):
                    if z.isnan().any():
                        print(f"Z_list[{s}][{r}] contains NaN values")
            raise ValueError("Z_list contains NaN values. Check for numerical issues.")

        

        # Eq (19)
        # Distance D_{i,c} 
        D = compute_D(Z_list, omega)
        if D.isnan().any():
            print("D contains NaN values. Check for numerical issues.")
            # Print noteworthy values for debugging
            print("D min:", D.min().item(), "D max:", D.max().item(), "D mean:", D.mean().item())
            print("Z_list[0][0] min:", Z_list[0][0].min().item(), "max:", Z_list[0][0].max().item(), "mean:", Z_list[0][0].mean().item())
            if Z_list.isnan().any():
                print("Z_list contains NaN values. Check for numerical issues.")
            raise ValueError("D contains NaN values. Check for numerical issues.")

        # Eq (17)
        u = update_u(D, m_fuzz=m_fuzz)
        if u.isnan().any():
            print("u contains NaN values. Check for numerical issues.")
            print("D min:", D.min().item(), "D max:", D.max().item(), "D mean:", D.mean().item())
            raise ValueError("u contains NaN values. Check for numerical issues.")
        if not torch.isfinite(u).all():
            print("u contains non-finite values. Check for numerical issues.")
            print("u min:", u.min().item(), "u max:", u.max().item(), "u mean:", u.mean().item())
            raise ValueError("u contains non-finite values. Check for numerical issues.")
        
        # Eq (23)
        omega = update_omega(
            Z_list, u, m_fuzz=m_fuzz,
            renormalize_sum1=renormalize_omega_sum1
        )
        u_change = torch.norm(u - u_prev).item()
        if u_change < eps_stop:
            print(f"MKFC converged after {_iter+1} iterations with u change {u_change:.4e}")
            break

    # print(_iter+1, "MKFC iterations until convergence")
    return u, omega


# --------------------------
# Autoencoder with encoder taps (mid = L/2)
# --------------------------
class AEWithTaps(nn.Module):
    """
    Encoder has 'mid' Linear layers: mid = L/2
    We return outputs after each Linear encoder layer as Y^(s), s=1..mid.
    """
    def __init__(self, dims_enc: List[int], dims_dec: List[int], activation: str = "relu"):
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")
        act = nn.ReLU if activation == "relu" else nn.Tanh

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        # Encoder: Linear + activation except after last Linear (common)
        for i in range(len(dims_enc) - 1):
            self.enc.append(nn.Linear(dims_enc[i], dims_enc[i + 1]))
            if i < len(dims_enc) - 1:
                self.enc.append(act())

        # Decoder
        for i in range(len(dims_dec) - 1):
            self.dec.append(nn.Linear(dims_dec[i], dims_dec[i + 1]))
            if i < len(dims_dec) - 1:
                self.dec.append(act())

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        Ys: List[torch.Tensor] = []
        y = x
        for layer in self.enc:
            y = layer(y)
            if isinstance(layer, nn.Linear):
                Ys.append(y)

        z = y  # y here is y_mid
        for layer in self.dec:
            z = layer(z)
        x_hat = z

        return Ys, y, x_hat


# --------------------------
# DMACN wrapper
# --------------------------
@dataclass
class DMACNConfig:
    name: str                           # name for saving results
    C: int
    dims_enc: List[int]               # dimension of each encoder layer, length mid+1
    dims_dec: List[int]               # dimension of each decoder layer, length mid+1
    kernel_specs: List[Dict]          # length h
    m_fuzz: float = 1.08              # exponent in u^m
    lam1: float = 0.5
    lam2: float = 0.5
    lr: float = 1e-3
    epochs: int = 200

    mk_max_iters: int = 50
    mk_eps_stop: float = 1e-5
    renormalize_omega_sum1: bool = True

    # "mid-only in first/last" as described in the paper text
    mid_only_first: bool = True
    mid_only_last: bool = True


class DMACN(nn.Module):
    """
    A practical, paper-aligned DMACN implementation with:
      - Algorithm 1 training loop (fit)
      - Algorithm 2 MKFC embedded
      - omega renormalization (optional but recommended)
    """
    def __init__(self, cfg: DMACNConfig, activation: str = "relu"):
        super().__init__()
        self.cfg = cfg
        self.ae = AEWithTaps(cfg.dims_enc, cfg.dims_dec, activation=activation)

        # outputs after fit
        self.u_: Optional[torch.Tensor] = None
        self.omega_: Optional[torch.Tensor] = None
        
    def frobenius_norm(self, parameter_name: str) -> torch.Tensor:
        return torch.sqrt(sum(
            torch.sum(parameter ** 2)
            for name, parameter in self.named_parameters()
            if parameter_name in name and parameter.requires_grad
        ))

    def fit(self, X: torch.Tensor, verbose_every: int = 10) -> "DMACN":
        """
        X: [N,d] float tensor on CPU/GPU.

        Returns self (fitted). Stores:
          self.u_     [N,C]
          self.omega_ [mid,h] or [1,h] if mid-only mode used at end
        """
        cfg = self.cfg
        device = X.device

        # SGD matches "a = a - eta dJ/da" notation nicely
        optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr)

        print("Ready to train DMACN:")
        
        
        # 1 Obtain sself-expression layer through feedforward process formula (3)
            
        # perform multi-kernel mapping, and perform Algorithm 2
        

        for epoch in range(cfg.epochs):
            self.train()

            # (1) forward (Eq. 3): obtain encoder features and reconstruction
            Ys, y_mid, x_hat = self.ae(X)

            # Decide which layers to pass into Algorithm 2
            use_mid_only = (epoch == 0 and cfg.mid_only_first) or (epoch == cfg.epochs - 1 and cfg.mid_only_last)
            Y_layers = [Ys[-1]] if use_mid_only else Ys
            # Note: mid here is len(Y_layers); paper uses s=1..mid, OK.

            # Multi-kernel mapping + Algorithm 2 (MKFC) to get u, omega, D
            u, omega= algorithm2_mkfc(
                Y_layers=Y_layers,
                kernel_specs=cfg.kernel_specs,
                C=cfg.C,
                m_fuzz=cfg.m_fuzz,
                eps_stop=cfg.mk_eps_stop,
                max_iters=cfg.mk_max_iters,
                renormalize_omega_sum1=cfg.renormalize_omega_sum1,
                input_u = self.u_
            )

            # (Backpropagate) build J = J1 + J2 + J3
            # J1 = 1/2 ||x - x_hat||_F (^2) - We are dropping the ^2 as it is seen as a typo
            # Minimize the reconstruction error
            # Frobenius norm
            J1 = 0.5 * torch.sum((X - x_hat) ** 2) # old code
            #norm = torch.linalg.matrix_norm(X - x_hat, ord='fro')
            #J1 = 0.5 * norm ** 2




            # Using Eq. (20) idea: T(omega)=sum u^m * sum omega^2 Z
            # since D_{i,c} already equals sum omega^2 Z, we do:
            # J2 = lam1/2 * sum_{i,c} u_ic^m * D_{i,c}
            # Guides clustering trend and helps autoencoder extract features that are good for clustering.
            
            J2 = compute_J2(
                Y_layers=Y_layers,
                u=u,
                omega=omega,
                cfg,
                kernel_specs=cfg.kernel_specs,
                m_fuzz=cfg.m_fuzz,
            )

            

            # J3 = lam2/2 (||a||^2 + ||b||^2)
            # Implemented explicitly (paper-like).
            # (PyTorch autograd gives correct bias gradients even if paper has a typo.)
            # Control size of the network weight a
            
            # parameters contains both a and b, and we want to take the norm of one at a time
            #reg = torch.tensor(0.0, device=device, dtype=X.dtype) #old
            #for p in self.parameters(): #old
            #    reg = reg + torch.sum(p ** 2) #old
            #J3 = 0.5 * cfg.lam2 * reg #old
            a_norm = self.frobenius_norm("weight")
            b_norm = self.frobenius_norm("bias")
            
            J3 = 0.5 * cfg.lam2 * (a_norm ** 2 + b_norm ** 2)

            J = J1 + J2 + J3

            optimizer.zero_grad()
            J.backward()
            optimizer.step()

            # (5) "train coding part again ... forward to obtain self-expression table"
            # In practice, next epoch forward reflects updated a,b.
            # We store latest u, omega for predict.
            self.u_ = u.detach()
            self.omega_ = omega.detach()

            if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch == cfg.epochs - 1):
                omega_sum = float(self.omega_.sum().item()) if self.omega_ is not None else float("nan")
                mode = "mid-only" if use_mid_only else "multilayer"
                print(
                    f"epoch {epoch+1:4d}/{cfg.epochs} [{mode}]  "
                    f"J={J.item():.4e}  J1={J1.item():.4e}  J2={J2.item():.4e}  J3={J3.item():.4e}  "
                    f"omega_sum={omega_sum:.4f}"
                )

        return self

    @torch.no_grad()
    def predict_probability(self) -> torch.Tensor:
        """
        Returns soft memberships u from last fit: [N,C]
        """
        if self.u_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X) first.")
        return self.u_

    @torch.no_grad()
    def predict(self, save: bool = False) -> torch.Tensor:
        """
        Hard labels from last fit: argmax over u. Returns [N]
        """
        u = self.predict_probability()
        labels = torch.argmax(u, dim=1)
        
        if save:
            # Count samples in each cluster and save to variables like label_0, label_1, etc.
            label_counts = {}
            for label in labels.unique():
                count = (labels == label).sum().item()
                label_counts[f"label_{label.item()}"] = count
                
            # Create the save-string
            save_str = "__".join([f"{name}_count_{count}" for name, count in label_counts.items()])
            print(save_str)
            
            # Save labels to a text file in folder "Clusters"
        np.savetxt(os.path.join("Clusters",f"{self.cfg.name}_{len(label_counts)}__Clusters__{save_str}.txt"), labels.cpu().numpy(), fmt="%d")
        
        return labels 
