"""Vector quantization modules for discrete tokenization.

Implements Residual Vector Quantization (RVQ) following the Moshi/Mimi reference
implementation (Défossez et al., 2024), with optional rotation trick gradient
estimator (Fifty et al., ICLR 2025).

References:
    - Moshi VQ: https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/quantization/core_vq.py
    - Rotation trick: https://github.com/cfifty/rotation_trick
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _apply_rotation(e: Tensor, q: Tensor) -> Tensor:
    """Rotation trick: differentiable rotation from encoder output to codebook entry.

    Replaces the straight-through estimator (STE) with a Householder rotation that
    encodes angular and magnitude information into the gradient.

    Forward: returns q exactly (rotation maps e to q).
    Backward: gradients flow through the rotation, giving the encoder directional
    information about how to adjust toward codebook entries.

    Uses efficient O(BD) formulation via dot products instead of explicit D×D matrix:
        R @ ê = ê - 2(w·ê)w + 2(û·ê)q̂
    where û = detach(ê), w = normalize(û + q̂).

    Args:
        e: Encoder output (*, D) — requires grad.
        q: Quantized codebook entry (*, D) — should be detached.

    Returns:
        Rotated output (*, D) — equals q in forward, has rotation gradients in backward.
    """
    e_hat = F.normalize(e, dim=-1, eps=1e-8)
    q_hat = F.normalize(q, dim=-1, eps=1e-8)
    q_scale = q.norm(dim=-1, keepdim=True)

    # Rotation matrix components (all constant w.r.t. e)
    u = e_hat.detach()
    w = F.normalize(u + q_hat, dim=-1)

    # R @ e_hat via dot products (R is constant, e_hat is differentiable)
    dot_we = (w * e_hat).sum(dim=-1, keepdim=True)
    dot_ue = (u * e_hat).sum(dim=-1, keepdim=True)
    rotated = e_hat - 2 * dot_we * w + 2 * dot_ue * q_hat

    return rotated * q_scale


def _sample_vectors(samples: Tensor, num: int) -> Tensor:
    """Randomly sample `num` vectors from `samples`."""
    n_samples = samples.size(0)
    if n_samples >= num:
        indices = torch.randperm(n_samples, device=samples.device)[:num]
    else:
        indices = torch.randint(0, n_samples, (num,), device=samples.device)
    return samples[indices]


def _kmeans(samples: Tensor, num_clusters: int, n_iters: int = 50) -> tuple[Tensor, Tensor]:
    """Run k-means clustering on samples.

    Args:
        samples: (N, D) input vectors.
        num_clusters: Number of clusters.
        n_iters: Number of k-means iterations.

    Returns:
        means: (K, D) cluster centroids.
        bins: (K,) number of samples assigned to each cluster.
    """
    dim = samples.size(-1)
    means = _sample_vectors(samples, num_clusters)
    bins = torch.zeros(num_clusters, device=samples.device)

    for _ in range(n_iters):
        dists = torch.cdist(samples.unsqueeze(0), means.unsqueeze(0), p=2).squeeze(0)  # (N, K)
        buckets = dists.argmin(dim=-1)  # (N,)
        bins = torch.bincount(buckets, minlength=num_clusters).float()
        zero_mask = bins == 0
        bins.clamp_(min=1)

        new_means = torch.zeros_like(means)
        new_means.scatter_add_(0, buckets.unsqueeze(1).expand(-1, dim), samples)
        new_means /= bins.unsqueeze(1)

        # Replace empty clusters with random samples
        resampled = _sample_vectors(samples, num_clusters)
        means = torch.where(zero_mask.unsqueeze(1), resampled, new_means)

    return means, bins


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer with EMA codebook updates.

    Matches the Moshi EuclideanCodebook implementation:
    - K-means initialization on first forward pass
    - EMA codebook updates (no gradient to codebook)
    - Commitment loss to pull encoder outputs toward codebook
    - Relative dead code threshold with periodic checking
    - No STE applied here — the RVQ applies STE (or rotation trick) to the total

    Args:
        dim: Embedding dimension.
        codebook_size: Number of codebook entries.
        commitment_cost: Weight for commitment loss (encoder -> codebook).
        ema_decay: EMA decay rate for codebook updates.
        threshold_usage_ratio: A code is "dead" if its usage is below this fraction
            of the average usage. Adapts automatically to batch size/codebook size.
        replaced_usage_ratio: Replaced codes get this fraction of mean usage as their
            initial cluster size (1.0 = mean usage).
        check_unused_every: Only check for dead codes every N forward passes.
        epsilon: Small constant for numerical stability.
        rotation_trick: Use rotation trick (Fifty et al., 2025) instead of returning
            raw detached quantized. Provides richer gradient signal to the encoder.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        check_unused_every: int = 5,
        epsilon: float = 1e-5,
        rotation_trick: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self.epsilon = epsilon
        self.rotation_trick = rotation_trick

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking buffers
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_sum", torch.zeros(codebook_size, dim))
        self.register_buffer("initialized", torch.tensor(False))

        # Raw assignment counts for unbiased marginal-entropy logging.
        # Unlike `ema_cluster_size`, these are not touched by dead-code resets,
        # so they reflect actual code usage frequency. Caller resets between
        # measurement windows (e.g. once per validation epoch).
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))

        # Dead code check counter (not a buffer — resets on checkpoint load, which is fine)
        self._next_unused_check = check_unused_every

    def _init_from_data(self, flat_input: Tensor) -> None:
        """Initialize codebook from first batch using k-means."""
        if self.initialized:
            return

        embeddings, cluster_counts = _kmeans(flat_input.detach().float(), self.codebook_size)
        self.embedding.weight.data.copy_(embeddings)
        self.ema_cluster_size.copy_(cluster_counts)
        self.ema_embed_sum.copy_(embeddings * cluster_counts.unsqueeze(1))
        self.initialized.fill_(True)

    def _quantize(self, flat_input: Tensor) -> Tensor:
        """Find nearest codebook entry for each input vector.

        Args:
            flat_input: (N, D) flattened input embeddings.

        Returns:
            indices: (N,) codebook indices.
        """
        # ||x - e||^2 = ||x||^2 - 2*x*e + ||e||^2
        dists = (
            flat_input.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_input @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        return dists.argmin(dim=1)

    def _replace_expired_codes(self, flat_input: Tensor) -> None:
        """Replace dead codebook entries with random encoder outputs.

        Uses relative threshold: a code is dead if its usage is below
        `threshold_usage_ratio` of the mean usage. Replaced codes get
        `replaced_usage_ratio` of the mean usage as their initial cluster size.

        Only runs every `check_unused_every` forward passes.
        """
        if not self.initialized:
            return

        self._next_unused_check -= 1
        if self._next_unused_check > 0:
            return
        self._next_unused_check = self.check_unused_every

        # Relative threshold — adapts to batch size and codebook size
        avg_usage = self.ema_cluster_size.sum() / self.codebook_size
        threshold = self.threshold_usage_ratio * avg_usage
        dead_codes = self.ema_cluster_size < threshold

        if not dead_codes.any():
            return

        # Replace dead codes with random encoder outputs
        new_vectors = _sample_vectors(flat_input.detach(), self.codebook_size)
        replace_usage = self.replaced_usage_ratio * avg_usage

        self.ema_embed_sum[dead_codes] = new_vectors[dead_codes].to(self.ema_embed_sum.dtype) * replace_usage
        self.ema_cluster_size[dead_codes] = replace_usage

    def _ema_update(self, flat_input: Tensor, indices: Tensor) -> None:
        """Update codebook using exponential moving average.

        Args:
            flat_input: (N, D) input embeddings.
            indices: (N,) assigned codebook indices.
        """
        # Count assignments per code
        cluster_size = torch.zeros(self.codebook_size, device=flat_input.device)
        cluster_size.scatter_add_(0, indices, torch.ones_like(indices, dtype=cluster_size.dtype))

        # EMA update cluster sizes
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

        # EMA update embedding sums
        embed_sum = torch.zeros_like(self.ema_embed_sum)
        embed_sum.scatter_add_(0, indices.unsqueeze(1).expand(-1, self.dim), flat_input.to(embed_sum.dtype))
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        # Derive embeddings from EMA state
        self.embedding.weight.data.copy_(
            self.ema_embed_sum / self.ema_cluster_size.clamp(min=self.epsilon).unsqueeze(1)
        )

    def forward(self, x: Tensor, initialize: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize input embeddings.

        Returns the raw quantized output (no STE). The RVQ applies STE to the
        total across all quantizer levels.

        Args:
            x: Input embeddings (B, T, D) or (B, D).
            initialize: Whether to allow k-means initialization on this call.
                Used by RVQ to prevent all layers from initializing on the same batch.

        Returns:
            z_q: Quantized embeddings (same shape as x, detached from codebook).
            indices: Codebook indices (B, T) or (B,).
            commitment_loss: Scalar commitment loss.
        """
        input_shape = x.shape
        flat_input = x.reshape(-1, self.dim)

        # Initialize from data (gated by `initialize` for cascaded RVQ init)
        if self.training and initialize and not self.initialized:
            self._init_from_data(flat_input)

        # Find nearest codebook entries
        with torch.no_grad():
            indices = self._quantize(flat_input)
            z_q = self.embedding(indices)
            # Accumulate raw assignment counts for marginal-entropy logging.
            # Done in both train and eval; caller resets between measurement windows.
            self.usage_count.scatter_add_(0, indices, torch.ones_like(indices, dtype=self.usage_count.dtype))

        # Training: expire dead codes first, then EMA update (Moshi order)
        if self.training:
            with torch.no_grad():
                self._replace_expired_codes(flat_input)
                self._ema_update(flat_input, indices)

        # Commitment loss: pull encoder outputs toward codebook
        commitment_loss = self.commitment_cost * F.mse_loss(flat_input, z_q.detach())

        # Apply rotation trick for differentiable gradient, or return raw detached
        if self.rotation_trick and self.training:
            z_q = _apply_rotation(flat_input, z_q)

        # Reshape outputs
        z_q = z_q.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])

        return z_q, indices, commitment_loss

    def encode(self, x: Tensor) -> Tensor:
        """Encode to indices only (no gradients needed).

        Args:
            x: Input embeddings (B, T, D).

        Returns:
            indices: Codebook indices (B, T).
        """
        flat_input = x.reshape(-1, self.dim)
        indices = self._quantize(flat_input)
        return indices.reshape(x.shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        """Decode indices to embeddings.

        Args:
            indices: Codebook indices (B, T) or (B,).

        Returns:
            z_q: Quantized embeddings (B, T, D) or (B, D).
        """
        return self.embedding(indices)

    def reset_usage_count(self) -> None:
        """Zero the raw assignment counter. Call before a measurement window."""
        self.usage_count.zero_()

    def marginal_entropy(self) -> Tensor:
        """Marginal entropy (in nats) of code usage since the last reset.

        Computed from raw assignment counts, NOT from EMA cluster sizes (which
        are inflated by dead-code resets and so under-report skew). Returns 0
        if no tokens have been accumulated yet.
        """
        total = self.usage_count.sum()
        if total == 0:
            return torch.zeros((), device=self.usage_count.device)
        p = self.usage_count.float() / total.float()
        nz = p > 0
        return -(p[nz] * p[nz].log()).sum()


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantization (RVQ) with multiple quantizer levels.

    Each level quantizes the residual from the previous level, enabling
    coarse-to-fine representation with multiple discrete codes per token.

    By default, STE (straight-through estimator) is applied once to the total
    quantized output, giving ∂z_q/∂x = I regardless of the number of quantizer
    levels (matching Moshi's implementation).

    With ``rotation_trick=True``, the STE is replaced by per-layer rotation
    (Fifty et al., ICLR 2025) that provides richer directional gradient
    information to the encoder.

    Args:
        dim: Embedding dimension.
        codebook_size: Number of entries per codebook.
        num_quantizers: Number of RVQ levels.
        commitment_cost: Weight for commitment loss.
        ema_decay: EMA decay rate for codebook updates.
        quantizer_dropout: Probability of dropping quantizers during training.
        rotation_trick: Use rotation trick instead of STE for gradient flow.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        quantizer_dropout: float = 0.0,
        rotation_trick: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.quantizer_dropout = quantizer_dropout
        self.rotation_trick = rotation_trick

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    dim=dim,
                    codebook_size=codebook_size,
                    commitment_cost=commitment_cost,
                    ema_decay=ema_decay,
                    rotation_trick=rotation_trick,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        num_quantizers: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Quantize input with residual vector quantization.

        Args:
            x: Input embeddings (B, T, D).
            num_quantizers: Override number of quantizers (for inference).

        Returns:
            z_q: Quantized embeddings (B, T, D) with STE gradient.
            indices: Codebook indices (B, T, num_quantizers).
            losses: Dict with 'commitment_loss' and 'residual_norm'.
        """
        n_q = num_quantizers if num_quantizers is not None else self.num_quantizers

        # Optional quantizer dropout during training (reduce number of active levels)
        if self.training and self.quantizer_dropout > 0:
            n_q = max(1, int(n_q * (1 - torch.rand(1).item() * self.quantizer_dropout)))

        z_q = torch.zeros_like(x)
        residual = x
        all_indices = []
        total_commitment = 0.0

        # Cascaded initialization (Moshi): layer i only k-means-initializes if
        # layer i-1 was already initialized from a *previous* batch.  This prevents
        # later layers from initializing on unrepresentative near-zero residuals.
        prev_initialized = True

        for i in range(n_q):
            if self.training:
                this_initialized = bool(self.quantizers[i].initialized)

            z_q_i, indices_i, commitment_i = self.quantizers[i](residual, initialize=prev_initialized)

            if self.training:
                prev_initialized = this_initialized

            if self.rotation_trick:
                # Rotation provides per-layer gradients — detach only for residual
                residual = residual - z_q_i.detach()
                z_q = z_q + z_q_i
            else:
                # Standard Moshi: detach everything, STE on total after loop
                z_q_i_detached = z_q_i.detach()
                residual = residual - z_q_i_detached
                z_q = z_q + z_q_i_detached

            all_indices.append(indices_i)
            total_commitment = total_commitment + commitment_i

        # Gradient method: STE on total (standard) or rotation (already applied per-layer)
        if self.training and not self.rotation_trick:
            z_q = x + (z_q - x).detach()

        indices = torch.stack(all_indices, dim=-1)

        return (
            z_q,
            indices,
            {
                "commitment_loss": total_commitment / n_q,
                "residual_norm": residual.detach().norm(dim=-1).mean(),
            },
        )

    def encode(self, x: Tensor, num_quantizers: int | None = None) -> Tensor:
        """Encode to discrete indices only (for inference).

        Args:
            x: Input embeddings (B, T, D).
            num_quantizers: Number of quantizers to use.

        Returns:
            indices: Codebook indices (B, T, num_quantizers).
        """
        n_q = num_quantizers if num_quantizers is not None else self.num_quantizers

        residual = x
        all_indices = []

        for i in range(n_q):
            indices_i = self.quantizers[i].encode(residual)
            z_q_i = self.quantizers[i].decode(indices_i)
            residual = residual - z_q_i
            all_indices.append(indices_i)

        return torch.stack(all_indices, dim=-1)

    def decode(self, indices: Tensor) -> Tensor:
        """Decode from indices to continuous embeddings.

        Args:
            indices: Codebook indices (B, T, num_quantizers).

        Returns:
            z_q: Quantized embeddings (B, T, D).
        """
        z_q = torch.zeros(
            indices.shape[0],
            indices.shape[1],
            self.dim,
            device=indices.device,
            dtype=self.quantizers[0].embedding.weight.dtype,
        )

        for i in range(indices.shape[-1]):
            z_q = z_q + self.quantizers[i].decode(indices[..., i])

        return z_q

    def get_codebook_usage(self) -> dict[str, Tensor]:
        """Get codebook usage statistics for monitoring.

        Uses perplexity (exp of entropy) which is batch-size invariant.
        Reports as fraction of codebook utilized (0 to 1).

        NOTE: This is computed from `ema_cluster_size`, which is inflated by
        dead-code resets (replaced codes get average usage). It therefore
        UNDER-reports skew. For an unbiased measurement use `get_marginal_stats`.

        Returns:
            Dict with usage stats per quantizer level.
        """
        stats = {}
        for i, q in enumerate(self.quantizers):
            cluster_size = q.ema_cluster_size.clamp(min=1e-10)
            probs = cluster_size / cluster_size.sum()
            entropy = -(probs * probs.log()).sum()
            perplexity = entropy.exp()
            stats[f"codebook_usage_q{i}"] = perplexity / q.codebook_size
        return stats

    def reset_marginal_stats(self) -> None:
        """Reset raw assignment counters across all levels."""
        for q in self.quantizers:
            q.reset_usage_count()

    def get_marginal_stats(self) -> dict[str, Tensor]:
        """Per-quantizer marginal entropy stats from raw assignment counts.

        Unlike `get_codebook_usage`, this is NOT biased by dead-code resets, so
        it can detect "codebook collapsed to uniform" — the failure mode where
        the encoder drifts away from the codebook and resets force every code
        to be hit equally often.

        For each level k, reports:
            marginal_entropy_q{k}      — H_k in nats
            marginal_entropy_norm_q{k} — H_k / log(V), in [0, 1]; ~1.0 means uniform
            effective_vocab_q{k}       — exp(H_k) / V, in [0, 1]; effective fraction of codes
        """
        stats: dict[str, Tensor] = {}
        for i, q in enumerate(self.quantizers):
            H = q.marginal_entropy()
            log_V = torch.log(torch.tensor(float(q.codebook_size), device=H.device))
            stats[f"marginal_entropy_q{i}"] = H
            stats[f"marginal_entropy_norm_q{i}"] = H / log_V
            stats[f"effective_vocab_q{i}"] = H.exp() / q.codebook_size
        return stats
