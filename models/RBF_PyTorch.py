
# models/RBFPyTorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

def initialize_centroids(X: torch.Tensor, num_neurons: int) -> torch.Tensor:
    """
    Brug KMeans til at få meningsfulde centroids i stedet for random.
    Input X er en torch.Tensor med shape (n_samples, input_dim).
    Returnerer en torch.Tensor med shape (num_neurons, input_dim).
    """
    # Skift til NumPy til KMeans
    X_np = X.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_neurons, n_init='auto').fit(X_np)
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    return centroids


class RBFLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        betas: float = 0.5,
        initial_centroids: torch.Tensor = None
    ):
        """
        input_dim: dimensionen af hver input-vector
        num_neurons: antal RBF‑neuroner
        betas: skaleringsparameter for radial basis
        initial_centroids: Tensor (num_neurons, input_dim) til initiering
        """
        super().__init__()
        self.num_neurons = num_neurons

        if initial_centroids is not None:
            # Brug forudberegnede centroids
            self.centroids = nn.Parameter(initial_centroids.clone())
        else:
            # Random init
            self.centroids = nn.Parameter(torch.randn(num_neurons, input_dim))

        # Betas som lærbar parameter (én per neuron)
        self.betas = nn.Parameter(torch.full((num_neurons,), betas))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor med shape (batch_size, input_dim)
        Returnerer Tensor med shape (batch_size, num_neurons)
        """
        # Beregn ||x - c_j||^2 for alle j
        # x.unsqueeze(1): (batch, 1, input_dim)
        # c.unsqueeze(0): (1, num_neurons, input_dim)
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=2)  # (batch_size, num_neurons)
        # exp( - beta_j * dist_sq )
        return torch.exp(- self.betas.unsqueeze(0) * dist_sq)


class RBFNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_hidden_neurons: int,
        output_dim: int,
        betas: float = 0.5,
        initial_centroids: torch.Tensor = None
    ):
        """
        input_dim: dimension af input
        num_hidden_neurons: antal RBF‐neuroner
        output_dim: antal outputs
        betas: initielt beta‐værdi
        initial_centroids: Tensor til RBF‐centroids
        """
        super().__init__()
        self.rbf = RBFLayer(input_dim, num_hidden_neurons, betas, initial_centroids)
        self.linear = nn.Linear(num_hidden_neurons, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rbf(x)
        return self.linear(x)
