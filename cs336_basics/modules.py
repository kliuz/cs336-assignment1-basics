import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Parameters:
            in_dim: final dimension of the input
            out_dim: final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        std = math.sqrt(2 / (in_dim + out_dim))
        weights = torch.empty(out_dim, in_dim, dtype=dtype, device=device)
        self.weight: nn.Parameter = nn.Parameter(nn.init.trunc_normal_(weights, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Parameters:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors, i.e., d_model
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        weights = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        self.weight: nn.Parameter = nn.Parameter(nn.init.trunc_normal_(weights, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

