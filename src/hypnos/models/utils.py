import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class ConvLayerNorm(nn.Module):
    """Layer norm for convolutional layers with channels first."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.eps = eps

    def forward(self, x_BCT: Tensor) -> Tensor:
        mu_BCT = x_BCT.mean(1, keepdim=True)
        sigma_BCT = (x_BCT - mu_BCT).pow(2).mean(1, keepdim=True)
        x_BCT = (x_BCT - mu_BCT) / torch.sqrt(sigma_BCT + self.eps)
        x_BCT = self.weight * x_BCT + self.bias
        return x_BCT


class ConvRMSNorm(nn.Module):
    """RMS normalization for convolutional layers."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.eps = eps

    def forward(self, x_BCT: Tensor) -> Tensor:
        sigma_BCT = x_BCT.pow(2).mean(1, keepdim=True)
        x_BCT = x_BCT / torch.sqrt(sigma_BCT + self.eps)
        x_BCT = self.weight * x_BCT
        return x_BCT


class CausalInstanceNorm1d(nn.Module):
    """Causal instance normalization.

    Uses rolling windows to compute normalisation statistics.
    Statistics can be computed in chunks of [N, C] to limit memory usage.
    """

    def __init__(
        self,
        num_features: int,
        window_size: int | None = None,
        eps: float = 1e-5,
        affine: bool = True,
        max_nc_chunk: int | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.window_size = window_size
        self.max_nc_chunk = max_nc_chunk  # [N, C] chunks to process in parallel (hardware dependent)
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if self.window_size is None:
            return self.forward_rolling(x)
        else:
            return self.forward_windowed(x)

    def forward_rolling(self, x: Tensor) -> Tensor:
        """Normalise using stats up to current position."""
        _, _, L = x.shape

        # Cumulative sums for mean
        S = F.pad(x, (1, 0)).cumsum(dim=-1)
        counts = torch.arange(1, L + 1, device=x.device, dtype=torch.float32).view(1, 1, L)
        mean = S[..., 1:] / counts

        # Cumulative sums for variance
        S2 = F.pad(x * x, (1, 0)).cumsum(dim=-1)
        var = S2[..., 1:] / counts - mean.pow(2)

        # Normalize
        x = (x - mean) * torch.rsqrt(var + self.eps)

        if self.affine:
            x = x * self.weight + self.bias
        return x

    def forward_windowed(self, x: Tensor) -> Tensor:
        _, _, L = x.shape
        W = self.window_size

        # Prefix sums (length L+1 with a leading zero)
        S = F.pad(x, (1, 0)).cumsum(dim=-1)
        mean = S[..., 1:] - F.pad(S[..., :-W], (W - 1, 0))
        del S

        # Counts (min(t+1, W)) via the same trick (broadcasted)
        ones_cum = F.pad(torch.ones(1, 1, L, device=x.device, dtype=torch.float32), (1, 0)).cumsum(-1)
        counts = ones_cum[..., 1:] - F.pad(ones_cum[..., :-W], (W - 1, 0))
        del ones_cum
        mean /= counts

        # Windowed sums via prefix-sum differences (causal, size up to W)
        S2 = F.pad(x * x, (1, 0)).cumsum(dim=-1)
        var = S2[..., 1:] - F.pad(S2[..., :-W], (W - 1, 0))
        del S2
        var = (var / counts) - mean.pow(2)
        del counts

        x = (x - mean) * torch.rsqrt(var.clamp_min(0.0) + self.eps)
        del mean

        if self.affine:
            x = x * self.weight + self.bias
        return x


def get_activation(name: str, **kwargs):
    """Return an activation function from its name."""
    if name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "leaky":
        return nn.LeakyReLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(**kwargs)
    elif name == "elu":
        return nn.ELU(**kwargs)
    elif name == "silu" or name == "swish":
        return nn.SiLU(**kwargs)
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"{name=} is unsupported.")


def get_norm(name: str | None = "batch", causal: bool = False, *args, **kwargs) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == "layer":
        return ConvLayerNorm(*args, **kwargs)
    elif name == "rms":
        return ConvRMSNorm(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    elif name == "instance":  # and not causal:
        return nn.InstanceNorm1d(*args, **kwargs)
    elif name == "instance" and causal:  # IGNORE FOR NOW
        return CausalInstanceNorm1d(*args, **kwargs)
    elif name.startswith("instance") and causal:
        window_size = int(name.split("_")[-1])
        return CausalInstanceNorm1d(*args, **kwargs, window_size=window_size)
    else:
        raise ValueError(f"Normalisation with {name=} and {causal=} unknown.")
