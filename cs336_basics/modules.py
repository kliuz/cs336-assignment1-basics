import math
import torch
import torch.nn as nn
from einops import einsum, reduce


class Linear(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            in_dim: Final dimension of the input
            out_dim: Final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        std = math.sqrt(2 / (in_dim + out_dim))
        weights = torch.empty(out_dim, in_dim, dtype=dtype, device=device)
        self.weight = nn.Parameter(nn.init.trunc_normal_(weights, std=std, a=-3 * std, b=3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors, i.e., d_model
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        weights = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        self.weight = nn.Parameter(nn.init.trunc_normal_(weights, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms: torch.Tensor = x**2
        rms = reduce(rms, "... dim -> ... 1", "mean") + self.eps
        rms = torch.sqrt(rms)

        result = x * self.weight / rms
        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: Dimensionality of the feedforward input and output.
            d_ff: Dimensionality of the up-project happening internally to your swiglu.
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.w1.forward(x)
        result *= torch.sigmoid(result)
        result *= self.w3.forward(x)
        return self.w2.forward(result)
