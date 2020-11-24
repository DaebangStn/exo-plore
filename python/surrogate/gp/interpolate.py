import torch


class KNNInterpolator:
    """
    PyTorch-only K-NN + interpolation surrogate.
    Parameters
    ----------
    train_x : (N, D) tensor
    train_y : (N) tensor   # arbitrary output shape
    k       : int               # #nearest neighbours
    p       : float|int         # Minkowski-p (p=2 ⇒ Euclidean)
    eps     : float             # small value to avoid div-by-0
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 k: int = 8, p: float = 2.0, eps: float = 1e-8):
        assert train_x.ndim == 2, "train_x shape (N,D)"
        if torch.cuda.is_available():
            print("[KNNInterpolator] Using GPU")
            self.device = torch.device("cuda")
        else:
            print("[KNNInterpolator] Using CPU")
            self.device = torch.device("cpu")
        self.x, self.y = train_x.clone().to(self.device), train_y.clone().to(self.device)
        self.k, self.p, self.eps = k, p, eps

    # ---- helper: pairwise distance ---- #
    def _dist(self, q: torch.Tensor):
        # (B, D) vs (N, D)  → (B, N)
        return torch.cdist(q, self.x, p=self.p)

    # ---- predict for batch q: (B,D) ---- #
    def forward(self, q: torch.Tensor, mode: str = "idw", power: float = 2.0):
        """
        mode = 'idw'  (inverse-distance weighting)
             = 'mean' (simple mean of neighbours)
        """
        prev_device = q.device
        q = q.to(self.device)
        d = self._dist(q)                    # (B,N)
        knn_d, knn_idx = torch.topk(d, self.k, dim=-1, largest=False)
        neigh_y = self.y[knn_idx]            # (B, k, ...)
        q = q.to(prev_device)

        if mode == "mean":
            return neigh_y.mean(dim=1)       # simple average

        # inverse-distance weighting
        w = 1.0 / (knn_d + self.eps)**power  # (B,k)
        w = w / w.sum(dim=1, keepdim=True)   # normalize
        
        # Handle broadcasting for arbitrary output shapes
        # w: (B, k), neigh_y: (B, k, ...)
        # Need to expand w to match neigh_y dimensions
        for _ in range(neigh_y.ndim - w.ndim):
            w = w.unsqueeze(-1)
        
        return (w * neigh_y).sum(dim=1).to(prev_device).squeeze(-1)

    def __call__(self, q: torch.Tensor, **kw):
        return self.forward(q, **kw)
