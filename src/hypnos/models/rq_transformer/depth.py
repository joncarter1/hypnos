"""Depth transformer for the RQ-Transformer.

Small causal transformer that generates K codebook tokens per timestep,
conditioned on temporal context. Following Lee et al.'s RQ-Transformer, the
input at depth position k is the per-timestep conditioning vector plus the
cumulative sum of codebook embeddings for the already-generated levels.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from hypnos.models.attention import _make_ffn, _run_checkpointed_layers, gradient_checkpoint


class DepthTransformerLayer(nn.Module):
    """Causal transformer layer for the depth transformer.

    Pre-norm with plain causal SDPA — no RoPE needed for short K-token sequences.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        swiglu: bool = False,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = _make_ffn(d_model, dim_feedforward, swiglu)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # Math backend: K is small (≈8) and Flash hits CUDA grid limits when
        # B*S*heads > 65535.
        with sdpa_kernel(SDPBackend.MATH):
            h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).reshape(B, L, D)
        h = self.out_proj(h)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)

        return x


def _build_depth_input(cond: Tensor, token_embeds: list[Tensor]) -> Tensor:
    """Construct depth-position inputs via cumulative residual sum.

    Args:
        cond: (B, S, D) per-timestep conditioning vector (added to every position).
        token_embeds: length-(K-1) list of (B, S, D) embeddings for q_0..q_{K-2}.

    Returns:
        (B, S, K, D) where position k = cond + Σ_{k'<k} token_embeds[k'].
    """
    B, S, D = cond.shape
    K = len(token_embeds) + 1
    if K == 1:
        return cond.unsqueeze(2)
    stacked = torch.stack(token_embeds, dim=2)  # (B, S, K-1, D)
    residuals = stacked.cumsum(dim=2)  # (B, S, K-1, D)
    zeros = cond.new_zeros(B, S, 1, D)
    residual_with_zero = torch.cat([zeros, residuals], dim=2)  # (B, S, K, D)
    return cond.unsqueeze(2) + residual_with_zero


def _sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token indices from ``(B, V)`` logits.

    ``temperature <= 0`` → greedy argmax. ``top_k > 0`` keeps the k highest-logit
    entries. ``0 < top_p < 1`` keeps the smallest set of entries whose cumulative
    probability first exceeds ``top_p`` (nucleus sampling). Returns ``(B,)`` long.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        # Keep the top token always; drop entries once cumulative prob (excluding
        # the current one) already exceeds top_p.
        remove = (cumprobs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class SharedDepthTransformer(nn.Module):
    """Depth transformer with shared layers and per-group tables.

    Transformer layers, projections, pos_embed, modality_embeddings, and
    null_mod_ctx are shared across modalities. Per-(signal_type, K, V) groups
    own their token-embedding tables and output heads. Grouping is computed
    externally by ``MultiModalRQTransformer`` and passed in as ``weight_group``
    (modality_name → leader_name).

    Args:
        modality_configs: Per-modality configs (name, K, V, signal_type).
        weight_group: modality_name → leader_name. Leaders own embeddings/heads.
        embed_dim: Input dimension from the temporal transformer.
        depth_dim: Internal depth dim.
        depth: Number of layers.
        num_heads: Attention heads.
        dim_feedforward: FFN hidden dim.
        dropout: Dropout probability.
        use_activation_checkpointing: Gradient checkpointing.
        swiglu: Use SwiGLU FFN.
        has_cls_context: If True, accept a CLS context tensor and add it to
            the per-timestep conditioning.
        mod_context_dropout_p: Per-(B, S) probability of replacing the modality
            temporal context (mod_ctx) with ``null_mod_ctx`` during training.
            Requires ``has_cls_context=True`` so the model still has a usable
            conditioning signal. The modality embedding is added after the
            replacement so depth still knows which modality it's generating.
    """

    def __init__(
        self,
        modality_configs: list,  # list[ModalityConfig] — avoid circular import
        weight_group: dict[str, str],
        embed_dim: int,
        depth_dim: int | None = None,
        depth: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        use_activation_checkpointing: bool = False,
        swiglu: bool = False,
        has_cls_context: bool = False,
        mod_context_dropout_p: float = 0.0,
    ):
        super().__init__()
        if not (0.0 <= mod_context_dropout_p < 1.0):
            raise ValueError(f"mod_context_dropout_p must be in [0, 1), got {mod_context_dropout_p}")

        self.modality_configs = modality_configs
        self._weight_group = weight_group
        self.use_activation_checkpointing = use_activation_checkpointing
        self.has_cls_context = has_cls_context
        self.mod_context_dropout_p = mod_context_dropout_p
        depth_dim = depth_dim or embed_dim
        self.depth_dim = depth_dim

        self._modality_name_to_idx = {mc.name: i for i, mc in enumerate(modality_configs)}
        self._K_of_modality = [mc.num_quantizers for mc in modality_configs]

        num_modalities = len(modality_configs)
        max_K = max(self._K_of_modality)

        self.per_group_embeddings = nn.ModuleDict()
        self.per_group_output_heads = nn.ModuleDict()
        for mc in modality_configs:
            if weight_group[mc.name] == mc.name:  # leader
                self.per_group_embeddings[mc.name] = nn.ModuleList(
                    [nn.Embedding(mc.codebook_size, depth_dim) for _ in range(mc.num_quantizers)]
                )
                self.per_group_output_heads[mc.name] = nn.ModuleList(
                    [nn.Linear(depth_dim, mc.codebook_size) for _ in range(mc.num_quantizers)]
                )

        self.context_proj = nn.Linear(embed_dim, depth_dim) if embed_dim != depth_dim else nn.Identity()
        if has_cls_context:
            self.cls_context_proj = nn.Linear(embed_dim, depth_dim) if embed_dim != depth_dim else nn.Identity()

        # Shared pos_embed sized for the longest modality's K.
        self.pos_embed = nn.Embedding(max_K, depth_dim)

        self.layers = nn.ModuleList(
            [
                DepthTransformerLayer(
                    d_model=depth_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    swiglu=swiglu,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(depth_dim)

        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, depth_dim) * 0.02)

        if has_cls_context and mod_context_dropout_p > 0.0:
            self.null_mod_ctx = nn.Parameter(torch.randn(depth_dim) * 0.02)

    def _run_layers(self, x: Tensor) -> Tensor:
        if self.use_activation_checkpointing:
            x = _run_checkpointed_layers(self.layers, x, lambda layer, x: layer(x))
        else:
            for layer in self.layers:
                x = layer(x)
        return self.norm(x)

    def _leader_for(self, modality_idx: int) -> str:
        return self._weight_group[self.modality_configs[modality_idx].name]

    def output_heads_for(self, modality_idx: int) -> nn.ModuleList:
        return self.per_group_output_heads[self._leader_for(modality_idx)]

    def forward(
        self,
        modality_idx: int,
        temporal_context: Tensor,
        target_tokens: Tensor,
        cls_context: Tensor | None = None,
    ) -> tuple[Tensor, nn.ModuleList]:
        """Teacher-forced forward for one modality.

        Returns (hidden, output_heads) where hidden is (B, S, K_i, depth_dim).
        """
        B, S, _ = temporal_context.shape
        K_i = self._K_of_modality[modality_idx]
        leader = self._leader_for(modality_idx)
        embeddings_i = self.per_group_embeddings[leader]
        output_heads_i = self.per_group_output_heads[leader]

        mod_ctx = self.context_proj(temporal_context)  # (B, S, D)

        # Replace mod_ctx with the learnable null at random (B, S) positions —
        # forces the model to lean on cls_context + modality embedding.
        if self.training and self.has_cls_context and cls_context is not None and self.mod_context_dropout_p > 0.0:
            drop_mask = torch.bernoulli(
                torch.full((B, S, 1), self.mod_context_dropout_p, device=mod_ctx.device, dtype=mod_ctx.dtype)
            ).bool()
            mod_ctx = torch.where(drop_mask, self.null_mod_ctx.to(mod_ctx.dtype), mod_ctx)

        cond = mod_ctx + self.modality_embeddings[modality_idx]
        if cls_context is not None and self.has_cls_context:
            cond = cond + self.cls_context_proj(cls_context)

        token_embeds = [embeddings_i[k](target_tokens[:, :, k]) for k in range(K_i - 1)]
        depth_input = _build_depth_input(cond, token_embeds)  # (B, S, K_i, D)

        positions = torch.arange(K_i, device=depth_input.device)
        depth_input = depth_input + self.pos_embed(positions)

        depth_input = depth_input.reshape(B * S, K_i, self.depth_dim)

        if self.use_activation_checkpointing and depth_input.requires_grad:
            chunk_size = 65536
            N = depth_input.size(0)
            if N > chunk_size:
                outputs = [
                    gradient_checkpoint(self._run_layers, c, use_reentrant=False) for c in depth_input.split(chunk_size)
                ]
                x = torch.cat(outputs, dim=0)
            else:
                x = gradient_checkpoint(self._run_layers, depth_input, use_reentrant=False)
        else:
            x = self._run_layers(depth_input)

        x = x.reshape(B, S, K_i, self.depth_dim)
        return x, output_heads_i

    @torch.no_grad()
    def sample(
        self,
        modality_idx: int,
        temporal_context: Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        cls_context: Tensor | None = None,
    ) -> Tensor:
        """Autoregressively sample K_i tokens for one modality at a single timestep.

        Uses the modality's weight-group embedding tables / output heads plus the shared
        ``modality_embeddings[modality_idx]`` conditioning (matching
        :meth:`SharedDepthTransformer.forward`).

        Args:
            modality_idx: Index into ``modality_configs``.
            temporal_context: (B, embed_dim) context for the timestep to predict.
            cls_context: (B, embed_dim) optional CLS context.

        Returns:
            (B, K_i) sampled token indices.
        """
        B = temporal_context.size(0)
        K_i = self._K_of_modality[modality_idx]
        leader = self._leader_for(modality_idx)
        embeddings_i = self.per_group_embeddings[leader]
        output_heads_i = self.per_group_output_heads[leader]
        device = temporal_context.device

        cond = self.context_proj(temporal_context) + self.modality_embeddings[modality_idx]
        if cls_context is not None and self.has_cls_context:
            cond = cond + self.cls_context_proj(cls_context)
        cond = cond.unsqueeze(1)  # (B, 1, depth_dim) — single timestep

        sampled: list[Tensor] = []
        for k in range(K_i):
            token_embeds = [embeddings_i[j](sampled[j]).unsqueeze(1) for j in range(k)]
            depth_input = _build_depth_input(cond, token_embeds)  # (B, 1, k+1, D)
            positions = torch.arange(k + 1, device=device)
            depth_input = (depth_input + self.pos_embed(positions)).reshape(B, k + 1, self.depth_dim)
            x = depth_input
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = output_heads_i[k](x[:, -1])  # (B, V)
            sampled.append(_sample_from_logits(logits, temperature, top_k, top_p, generator))

        return torch.stack(sampled, dim=1)  # (B, K_i)
