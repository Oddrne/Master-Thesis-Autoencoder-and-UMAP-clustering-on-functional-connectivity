from dataclasses import dataclass
import torch


# Algortime 2: Multi-view kernel fuzzy c-means (MKFC)
# --------------------------
# Kernel helpers (N x D -> N x N)
# --------------------------
def rbf_kernel_matrix(Y, gamma: float):
    # Y: [N,D]
    Y2 = (Y * Y).sum(dim=1, keepdim=True)           # [N,1]
    dist2 = Y2 + Y2.T - 2.0 * (Y @ Y.T)            # [N,N]
    return torch.exp(-gamma * torch.clamp(dist2, min=0.0))

def poly_kernel_matrix(Y, degree: int = 2, coef0: float = 1.0):
    return (Y @ Y.T + coef0) ** degree

def build_kernel_matrix(Y, spec: dict):
    kind = spec["kind"]
    if kind == "rbf":
        return rbf_kernel_matrix(Y, gamma=float(spec.get("gamma", 1.0)))
    if kind == "poly":
        return poly_kernel_matrix(Y, degree=int(spec.get("degree", 2)), coef0=float(spec.get("coef0", 1.0)))
    raise ValueError(f"Unknown kernel kind: {kind}")

# --------------------------
# Eq. (18): compute Z_{i,c}^{(s,r)} from kernel matrix K and memberships u
# --------------------------
def compute_Z_ic_from_K(K, u, m: float, eps: float = 1e-12):
    """
    K: [N,N] kernel matrix for a fixed (s,r)
    u: [N,C] memberships
    m: fuzzifier exponent in u^m
    returns Z: [N,C] where Z[i,c] is RKHS squared distance to cluster centroid
    """
    N, C = u.shape
    um = torch.clamp(u, min=eps) ** m              # [N,C]

    # Normalized membership weights for each cluster: \bar u_c
    denom = um.sum(dim=0, keepdim=True)            # [1,C]
    ubar = um / torch.clamp(denom, min=eps)        # [N,C]

    # Terms:
    # K_ii: [N]
    K_diag = torch.diagonal(K, 0)                  # [N]

    # term2: 2 * (ubar_c^T K_{:,i}) for each i,c
    # (K @ ubar): [N,C], entry [i,c] = sum_j K_{i,j} ubar[j,c]
    Ku = K @ ubar                                  # [N,C]

    # term3: ubar_c^T K ubar_c for each cluster c (scalar per c)
    # ubar^T (K @ ubar): [C,C], take diagonal -> [C]
    uKu = (ubar.T @ Ku)                             # [C,C]
    uKu_diag = torch.diagonal(uKu, 0)               # [C]

    # Z[i,c] = K_ii[i] - 2*(Ku[i,c]) + uKu_diag[c]
    Z = K_diag[:, None] - 2.0 * Ku + uKu_diag[None, :]
    return torch.clamp(Z, min=eps)

# --------------------------
# Eq. (19): D_{i,c} = sum_{s,r} ω_{s,r}^2 * Z_{i,c}^{(s,r)}
# --------------------------
def compute_D(Z_list, omega):
    """
    Z_list: nested list [mid][h] each item is [N,C]
    omega:  [mid,h]
    returns D: [N,C]
    """
    mid = len(Z_list)
    h = len(Z_list[0])
    N, C = Z_list[0][0].shape
    D = torch.zeros((N, C), device=Z_list[0][0].device, dtype=Z_list[0][0].dtype)
    for s in range(mid):
        for r in range(h):
            D = D + (omega[s, r] ** 2) * Z_list[s][r]
    return torch.clamp(D, min=1e-12)

# --------------------------
# Eq. (17): u update using D
#   u_ic = 1 / sum_{c'} (D_ic / D_i,c')^(1/(m-1))
# --------------------------
def update_u(D, m: float, eps: float = 1e-12):
    if m <= 1.0:
        raise ValueError("Fuzzy exponent m must be > 1.")
    power = 1.0 / (m - 1.0)
    D = torch.clamp(D, min=eps)
    ratio = (D[:, :, None] / D[:, None, :]) ** power   # [N,C,C]
    u = 1.0 / torch.clamp(ratio.sum(dim=2), min=eps)   # [N,C]
    return torch.clamp(u, min=eps, max=1.0)

# --------------------------
# Eq. (23): omega_{s,r}
#   A_{s,r} = sum_{i,c} u_ic^m * Z_{i,c}^{(s,r)}
#   omega_{s,r} = (1/A_{s,r}) / (2 * sum_{s,r} (1/A_{s,r})) Obs: Fjerner 2
# --------------------------
def update_omega(Z_list, u, m: float, eps: float = 1e-12):
    mid = len(Z_list)
    h = len(Z_list[0])
    um = torch.clamp(u, min=eps) ** m  # [N,C]

    invA = torch.zeros((mid, h), device=u.device, dtype=u.dtype)
    for s in range(mid):
        for r in range(h):
            A_sr = torch.sum(um * Z_list[s][r])      # scalar
            invA[s, r] = 1.0 / torch.clamp(A_sr, min=eps)

    denom = torch.sum(invA) #*2.0 # Obs: Fjerner 2
    omega = invA / torch.clamp(denom, min=eps)
    return torch.clamp(omega, min=eps)

# --------------------------
# Full Algorithm 2 loop
# --------------------------
def mkfc_algorithm2(
    Y_layers,               # list length mid, each is Tensor [N, d_s]
    kernel_specs,           # list length h; each spec is dict, e.g. {"kind":"rbf","gamma":1.0}
    C: int,                 # number of clusters
    m: float = 1.08,        # fuzzifier, the degree of membership (u)
    eps_stop: float = 1e-5, # stopping criterion
    max_iters: int = 100,   # max iterations
):
    """
    Implements Algorithm 2:
      init omega_{s,r}=1/(mid*h)
      repeat:
        Z^{(s,r)} via Eq(18)
        D via Eq(19)
        u via Eq(17)
        omega via Eq(23)
      until ||u^t - u^(t-1)|| < eps
    Returns: u [N,C], omega [mid,h], D [N,C]
    """
    device = Y_layers[0].device
    dtype = Y_layers[0].dtype

    mid = len(Y_layers)
    h = len(kernel_specs)
    N = Y_layers[0].shape[0]

    # init u uniform, omega uniform
    u = torch.full((N, C), 1.0 / C, device=device, dtype=dtype)
    omega = torch.full((mid, h), 1.0 / (mid * h), device=device, dtype=dtype)

    # Precompute kernel matrices K^{(s,r)} (you can recompute each epoch if Y changes)
    K_mats = [[None for _ in range(h)] for __ in range(mid)]
    for s in range(mid):
        for r in range(h):
            K_mats[s][r] = build_kernel_matrix(Y_layers[s], kernel_specs[r])  # [N,N]

    for t in range(1, max_iters + 1):
        u_prev = u

        # Eq (18): Z_list[s][r] is [N,C]
        Z_list = [[None for _ in range(h)] for __ in range(mid)]
        for s in range(mid):
            for r in range(h):
                Z_list[s][r] = compute_Z_ic_from_K(K_mats[s][r], u, m=m)

        # Eq (19): D
        D = compute_D(Z_list, omega)

        # Eq (17): update u
        u = update_u(D, m=m)

        # Eq (23): update omega
        omega = update_omega(Z_list, u, m=m)

        # stop
        if torch.norm(u - u_prev).item() < eps_stop:
            break

    return u, omega, D




# Algorithm 1: Deep Multi-view Autoencoder Clustering Network (DMACN)
# ============ Autoencoder som gjev ut encoder-layer features ============

class AEWithTaps(nn.Module):
    """
    Encoder: mid lag. Decoder: spegla.
    Vi returnerer:
      - Ys: liste av encoder outputs per lag (lengd mid)
      - y_mid: siste encoder output
      - x_hat: rekonstruksjon
    """
    def __init__(self, dims_enc, dims_dec, activation="relu"):
        super().__init__()
        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()

        act = nn.ReLU if activation == "relu" else nn.Tanh

        # encoder
        for i in range(len(dims_enc) - 1):
            self.enc_layers.append(nn.Linear(dims_enc[i], dims_enc[i+1]))
            if i < len(dims_enc) - 2:
                self.enc_layers.append(act())

        # decoder
        for i in range(len(dims_dec) - 1):
            self.dec_layers.append(nn.Linear(dims_dec[i], dims_dec[i+1]))
            if i < len(dims_dec) - 2:
                self.dec_layers.append(act())

    def forward(self, x):
        Ys = []
        y = x
        # encoder: samle outputs etter kvar Linear (før aktivering er ofte uinteressant her)
        for layer in self.enc_layers:
            y = layer(y)
            if isinstance(layer, nn.Linear):
                Ys.append(y)
        y_mid = y

        # decoder
        z = y_mid
        for layer in self.dec_layers:
            z = layer(z)
        x_hat = z
        return Ys, y_mid, x_hat
    

# ============ Algorithm 1: DMACN trening ============

def train_dmacn(
    X,                      # [N,d]
    C: int,
    dims_enc,               # t.d. [d, 512, 128]  -> mid = len([512,128]) = 2 linearlag
    dims_dec,               # t.d. [128, 512, d]
    kernel_specs,           # lengd h
    lam1: float,
    lam2: float,
    lr: float,
    epochs: int,
    m_fuzz: float = 2.0,
    mk_iters: int = 50,
):
    """
    Følgjer Algorithm 1:
      (1) feedforward -> MK mapping -> Algorithm 2 (u, omega)
      (2)-(4) gradient (autograd) og oppdatering av a,b (optimizer step)
      (5) ny feedforward for å få oppdatert "self-expression table" (her: encoder-features)
    """
    device = X.device
    model = AEWithTaps(dims_enc=dims_enc, dims_dec=dims_dec).to(device)

    # Vi bruker SGD for å spegle "a^l = a^l - eta dJ/da^l" osv.
    # (du kan bytte til Adam om du vil, men SGD er nærast papernotasjon)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        # (1) feedforward (formel 3)
        Ys, y_mid, x_hat = model(X)

        # multi-layer features inn i Algorithm 2:
        # mid = tal encoder linear-lag (Ys inneheld berre Linear-output)
        # Om du vil bruke berre L/2-laget slik teksten nemner, bruk berre [Ys[-1]]
        Y_layers = Ys  # liste lengd mid

        # MKFC / Algorithm 2 (oppdater u, omega)
        with torch.no_grad():
            u, omega, D = mkfc_algorithm2(
                Y_layers=Y_layers,
                kernel_specs=kernel_specs,
                C=C,
                m_fuzz=m_fuzz,
                max_iters=mk_iters,
            )

        # Bygg J1, J2, J3
        # J1 = 1/2 ||x - xhat||_F^2
        J1 = 0.5 * torch.sum((X - x_hat) ** 2)

        # For J2 brukar vi Eq. (20)-ideen: sum_i sum_c u_ic^m * sum_{s,r} omega^2 Z
        # Merk: D_{i,c} ER allereie sum_{s,r} omega^2 Z, så:
        # J2 = lam1/2 * sum_{i,c} u_ic^m * D_{i,c}
        um = (u ** m_fuzz)
        J2 = 0.5 * lam1 * torch.sum(um * D)

        # J3 = lam2/2 (||a||^2 + ||b||^2)
        # Her tek vi L2 over alle parameters (vekter + bias)
        J3 = 0.0
        for p in model.parameters():
            J3 = J3 + torch.sum(p * p)
        J3 = 0.5 * lam2 * J3

        J = J1 + J2 + J3

        # (3) gradient (formel 7) og (4) update a,b (formel 10)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        # (5) “train coding part again ... forward propagation”
        # I praksis: neste epoch startar med ny forward med oppdaterte a,b.
        # Om du vil tvinge ein ekstra forward her (utan step), kan du kalle model(X) på nytt.

        if (epoch + 1) % 10 == 0:
            # rapport
            print(
                f"epoch {epoch+1:4d} | J={J.item():.4e} | J1={J1.item():.4e} | J2={J2.item():.4e} | J3={J3.item():.4e} "
                f"| omega_sum={omega.sum().item():.4f}"
            )

    # output: clustering results -> hard labels frå u
    labels = torch.argmax(u, dim=1)
    return model, u, omega, labels