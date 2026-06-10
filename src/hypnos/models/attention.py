"""Shared attention primitives for transformer models.

Provides rotary position embeddings (RoPE), compiled FlexAttention wrappers,
sliding-window mask factories, block-mask caching, a unified pre-norm
transformer layer (RoPETransformerLayer), and a full transformer stack
(RoPETransformer) with optional sliding window.
"""

from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint as _raw_checkpoint


def _preserve_default_dtype_context_fn():
    """``context_fn`` that captures the current default dtype and re-enters
    it across both the forward and recompute legs of a checkpoint. See
    :func:`gradient_checkpoint` for the full story.
    """
    captured = torch.get_default_dtype()

    @contextmanager
    def _cm():
        prev = torch.get_default_dtype()
        torch.set_default_dtype(captured)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)

    return _cm(), _cm()


def gradient_checkpoint(fn, *args, **kwargs):
    """``torch.utils.checkpoint.checkpoint`` with default-dtype preservation.

    Activation-checkpointing helper used only when ``use_activation_checkpointing`` is
    enabled (training); inference does not call it. Forces ``use_reentrant=False`` and pipes
    a context manager into both the forward and recompute legs so ``default_dtype`` matches
    across them (``torch.utils.checkpoint`` preserves RNG/autocast state across recompute but
    not ``default_dtype``, which can otherwise cause a metadata mismatch in nested checkpoints
    that save dtype-dependent placeholder tensors).
    """
    kwargs['use_reentrant'] = False
    kwargs.setdefault('context_fn', _preserve_default_dtype_context_fn)
    return _raw_checkpoint(fn, *args, **kwargs)

# ---------------------------------------------------------------------------
# Compiled FlexAttention wrappers
# ---------------------------------------------------------------------------
# Calling flex_attention without torch.compile falls back to a slow eager
# implementation that materialises the full (B, H, T, T) attention matrix
# instead of fusing the kernel. For long sequences (e.g. tokenizer inference
# on overnight PSG recordings) this causes OOMs, so we compile on accelerators.
# torch.compile is not supported on CPU, so there we use the eager path (slower
# and heavier, but fine for local/CPU inference on shorter recordings).
_compiled_flex_attention_accel = None
_compiled_create_block_mask_accel = None


def _compiled_flex_attention(query, key, value, **kwargs):
    if query.device.type == 'cpu':
        return flex_attention(query, key, value, **kwargs)
    global _compiled_flex_attention_accel
    if _compiled_flex_attention_accel is None:
        _compiled_flex_attention_accel = torch.compile(flex_attention)
    return _compiled_flex_attention_accel(query, key, value, **kwargs)


def _compiled_create_block_mask(mask_fn, **kwargs):
    device = kwargs.get('device')
    if getattr(device, 'type', None) == 'cpu' or str(device) == 'cpu':
        return create_block_mask(mask_fn, **kwargs)
    global _compiled_create_block_mask_accel
    if _compiled_create_block_mask_accel is None:
        _compiled_create_block_mask_accel = torch.compile(create_block_mask)
    return _compiled_create_block_mask_accel(mask_fn, **kwargs)

# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Precomputes complex-valued rotation frequencies for applying
    rotary embeddings to query and key tensors in attention.

    Buffers are extended dynamically if the sequence length exceeds
    the precomputed size, so there is no hard maximum sequence length.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Initial sequence length to precompute.
        theta: Base frequency for rotary embeddings.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin caches up to the given sequence length."""
        t = torch.arange(seq_len, device=self.freqs.device)
        freqs_table = torch.outer(t, self.freqs)
        self.register_buffer('cos_cached', freqs_table.cos())
        self.register_buffer('sin_cached', freqs_table.sin())

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        """Return cos and sin tables for the given sequence length.

        Extends the precomputed cache if seq_len exceeds it.

        Returns:
            cos: (seq_len, dim//2)
            sin: (seq_len, dim//2)
        """
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to a tensor.

    Args:
        x: (..., seq_len, dim) tensor.
        cos: (seq_len, dim//2) cosine table.
        sin: (seq_len, dim//2) sine table.

    Returns:
        Rotated tensor of same shape as x.
    """
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return rotated.flatten(-2)


# ---------------------------------------------------------------------------
# Sliding-window mask factories
# ---------------------------------------------------------------------------


def _make_causal_sliding_window_mask(window_size: int):
    """Create a mask function for causal sliding window attention."""

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

    return mask_fn


def _make_bidirectional_sliding_window_mask(window_size: int):
    """Create a mask function for bidirectional sliding window attention."""
    half_w = window_size // 2

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= half_w

    return mask_fn


# ---------------------------------------------------------------------------
# Block-mask caching
# ---------------------------------------------------------------------------


def _dense_attention_mask(mask_fn, seq_len: int, device: torch.device) -> Tensor:
    """Materialise a dense ``(seq_len, seq_len)`` boolean mask from a FlexAttention
    mask function, for the MPS fallback path (FlexAttention has no MPS backend).

    ``True`` means the query/key pair may attend — the same convention
    ``F.scaled_dot_product_attention`` uses for a boolean ``attn_mask``. Costs
    O(seq_len**2) memory, the same as eager FlexAttention, so callers bound
    *seq_len* via chunking.
    """
    q_idx = torch.arange(seq_len, device=device).view(seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, seq_len)
    return mask_fn(None, None, q_idx, kv_idx)


def get_block_mask(
    mask_fn,
    seq_len: int,
    device: torch.device,
    cache: dict,
) -> BlockMask | Tensor | None:
    """Get an attention mask, creating or reusing a cached one.

    On CUDA/CPU this is a FlexAttention :class:`BlockMask`. On MPS — where
    FlexAttention has no backend — it is a dense ``(seq_len, seq_len)`` boolean
    mask; :class:`RoPETransformerLayer` dispatches to SDPA on that type.

    Args:
        mask_fn: The mask function, or ``None`` for no masking.
        seq_len: Current sequence length.
        device: Device for mask creation.
        cache: Mutable dict with keys ``'block_mask'`` and ``'seq_len'``
            for caching.  Pass the same dict across calls to enable caching.

    Returns:
        BlockMask (CUDA/CPU), dense bool Tensor (MPS), or None if *mask_fn* is None.
    """
    if mask_fn is None:
        return None
    if cache.get('block_mask') is not None and cache.get('seq_len') == seq_len:
        return cache['block_mask']
    if getattr(device, 'type', None) == 'mps' or str(device) == 'mps':
        mask = _dense_attention_mask(mask_fn, seq_len, device)
    else:
        mask = _compiled_create_block_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
    cache['block_mask'] = mask
    cache['seq_len'] = seq_len
    return mask


# ---------------------------------------------------------------------------
# Checkpointing utilities
# ---------------------------------------------------------------------------


def _run_checkpointed_layers(
    layers: nn.ModuleList,
    x: Tensor,
    run_layer: Callable[[nn.Module, Tensor], Tensor],
) -> Tensor:
    """Run transformer layers with per-layer activation checkpointing.

    Args:
        layers: ModuleList of layers to run.
        x: Input tensor — must be the only tensor that requires gradients.
        run_layer: ``run_layer(layer, x) -> x`` that executes a single layer.
            May capture non-gradient tensors (cos, sin, masks) via closure.
    """
    if not x.requires_grad:
        for layer in layers:
            x = run_layer(layer, x)
        return x

    for layer in layers:
        x = gradient_checkpoint(run_layer, layer, x, use_reentrant=False)
    return x


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (LLaMA / PaLM style).

    Uses three projections (gate, up, down) with SiLU gating.
    Hidden dim is auto-adjusted to 2/3 of the requested dim_feedforward
    so that total parameter count matches a standard GELU FFN with the
    same dim_feedforward.

    Args:
        d_model: Input/output dimension.
        dim_feedforward: Nominal FFN hidden dimension (before 2/3 adjustment).
    """

    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        hidden = int(2 * dim_feedforward / 3)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _make_ffn(d_model: int, dim_feedforward: int, swiglu: bool) -> nn.Module:
    """Create an FFN block — SwiGLU or standard GELU."""
    if swiglu:
        return SwiGLUFFN(d_model, dim_feedforward)
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.GELU(),
        nn.Linear(dim_feedforward, d_model),
    )


# ---------------------------------------------------------------------------
# Unified transformer layer
# ---------------------------------------------------------------------------


class RoPETransformerLayer(nn.Module):
    """Pre-norm transformer layer with RoPE, LayerScale, and optional KV cache.

    Unified layer used by both the tokenizer (RoPETransformer) and the
    temporal transformer.  Supports three attention dispatch modes:

    1. FlexAttention with *block_mask* (sliding-window patterns).
    2. Full SDPA when *past_kv* is provided (generation with KV cache).
    3. Standard SDPA with *is_causal* flag (no dense mask needed).

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        layer_scale_init: Initial value for LayerScale parameters.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        layer_scale_init: float = 0.01,
        dropout: float = 0.0,
        qk_norm: bool = False,
        swiglu: bool = False,
        xsa: bool = False,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.xsa = xsa

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention projections (no bias, matching Mimi)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-norm: per-head RMSNorm on Q/K to bound attention logits (ViT-22B)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        # FFN
        self.ffn = _make_ffn(d_model, dim_feedforward, swiglu)

        # LayerScale
        self.layer_scale_attn = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.layer_scale_ffn = nn.Parameter(torch.ones(d_model) * layer_scale_init)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        block_mask: BlockMask | Tensor | None = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            x: (B, S, D) input embeddings.
            cos: (S, head_dim//2) RoPE cosine table.
            sin: (S, head_dim//2) RoPE sine table.
            past_kv: Optional cached (K, V) from previous steps for generation.
            block_mask: FlexAttention block mask for sliding-window patterns.
            is_causal: Use efficient causal SDPA (when no block_mask or past_kv).

        Returns:
            output: (B, S, D) contextualized embeddings.
            kv: (K, V) key-value tensors for caching.
        """
        B, S, D = x.shape

        # Pre-norm + QKV projection
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.nhead, self.head_dim)
        k = self.k_proj(h).view(B, S, self.nhead, self.head_dim)
        v = self.v_proj(h).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # QK-norm (per-head, before RoPE — RoPE preserves norms)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)

        # Apply RoPE to Q, K
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Concatenate with past KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)

        # Attention dispatch
        if block_mask is not None and past_kv is None:
            # RoPE may upcast Q/K to fp32; both kernels need Q/K/V to share a dtype.
            q, k = q.to(v.dtype), k.to(v.dtype)
            if isinstance(block_mask, BlockMask):
                h = _compiled_flex_attention(q, k, v, block_mask=block_mask)
            else:
                # Dense boolean mask (MPS fallback): SDPA has an MPS backend, FlexAttention does not.
                h = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        elif past_kv is not None:
            # Single-step generation: q attends to all past + current keys
            h = F.scaled_dot_product_attention(q, k, v)
        else:
            h = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Exclusive Self-Attention: project out self-value component (arXiv:2603.09078)
        if self.xsa:
            v_cur = v[:, :, -S:]  # handle KV cache: only current positions
            v_norm = F.normalize(v_cur, dim=-1)
            h = h - (h * v_norm).sum(dim=-1, keepdim=True) * v_norm

        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        h = self.dropout(h)
        x = x + self.layer_scale_attn * h

        # Pre-norm + FFN
        h = self.norm2(x)
        h = self.ffn(h)
        h = self.dropout(h)
        x = x + self.layer_scale_ffn * h

        return x, new_kv


# ---------------------------------------------------------------------------
# Transformer stacks
# ---------------------------------------------------------------------------


class ModalityAttentionLayer(nn.Module):
    """Pre-norm bidirectional self-attention layer for cross-modality interaction.

    Designed for short sequences (M=2-4 modality tokens per timestep).
    No RoPE (modality positions are unordered). Uses learned positional
    embeddings and SDPA MATH backend for small sequences.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        num_modalities: Maximum number of modalities (for positional embeddings).
        layer_scale_init: Initial value for LayerScale parameters.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_modalities: int,
        layer_scale_init: float = 0.01,
        dropout: float = 0.0,
        swiglu: bool = False,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_ffn = use_ffn

        self.norm1 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.pos_embed = nn.Embedding(num_modalities, d_model)

        self.layer_scale_attn = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.dropout = nn.Dropout(dropout)

        if use_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = _make_ffn(d_model, dim_feedforward, swiglu)
            self.layer_scale_ffn = nn.Parameter(torch.ones(d_model) * layer_scale_init)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """Bidirectional self-attention across modalities.

        Args:
            x: (B_eff, M, D) where B_eff = B * S (batch * time flattened).
            attn_mask: Pre-computed float additive mask for SDPA.
                Shape (1, 1, M, M) for static or (B_eff, 1, M, M) for per-sample.
                Values are 0.0 (attend) or -inf (block). None = full attention.
            position_ids: (M,) long, the modality indices used to look up
                ``pos_embed``. None = ``arange(M)`` (default, identity case
                during training). Used by callers that pass a sliced ``x``
                holding a subset of the original modalities — pass the
                original indices so each modality keeps its learned pos
                embedding.
        """
        B_eff, M, D = x.shape

        # Add learned modality positional embeddings
        if position_ids is None:
            position_ids = torch.arange(M, device=x.device)
        x = x + self.pos_embed(position_ids)

        # Self-attention
        h = self.norm1(x)
        q = self.q_proj(h).view(B_eff, M, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B_eff, M, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B_eff, M, self.nhead, self.head_dim).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if attn_mask is not None:
                h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B_eff, M, D)
        h = self.out_proj(h)
        h = self.dropout(h)
        x = x + self.layer_scale_attn * h

        if self.use_ffn:
            h = self.norm2(x)
            h = self.ffn(h)
            h = self.dropout(h)
            x = x + self.layer_scale_ffn * h

        return x


class IdentityAttention(nn.Module):
    """Identity module used when no attention is desired."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class RoPETransformer(nn.Module):
    """RoPE + LayerScale transformer stack with optional sliding window.

    Architecture follows the Mimi neural codec (Défossez et al. 2024).
    Used as a drop-in contextualizer wherever a sequence of embeddings
    needs self-attention with rotary position encoding.

    Uses FlexAttention for sliding window patterns (O(T*W) memory) and
    efficient SDPA for causal-only or full attention (no dense mask creation).

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        window_size: Sliding window size for attention (None = full attention).
        causal: Whether to use causal masking.
        layer_scale_init: LayerScale initialization value.
        max_seq_len: Initial sequence length for RoPE precomputation (extended dynamically).
        dropout: Dropout probability.
        use_activation_checkpointing: Trade compute for memory by recomputing activations during backward.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        window_size: int | None = 250,
        causal: bool = True,
        layer_scale_init: float = 0.01,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.use_activation_checkpointing = use_activation_checkpointing
        head_dim = embed_dim // num_heads

        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    layer_scale_init=layer_scale_init,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Build the mask function (if needed) and cache the block mask
        self._mask_fn = self._build_mask_fn()
        self._mask_cache: dict = {}

    def _build_mask_fn(self):
        """Build a FlexAttention mask function from causal/window settings.

        Returns None when no mask is needed (full bidirectional attention)
        or when only causal masking is needed (handled via is_causal flag).
        """
        if self.window_size is not None:
            if self.causal:
                return _make_causal_sliding_window_mask(self.window_size)
            return _make_bidirectional_sliding_window_mask(self.window_size)
        # No window — causal-only handled via is_causal flag, full attention needs no mask
        return None

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer to embeddings.

        Args:
            x: Input embeddings (B, T, D)

        Returns:
            Contextualized embeddings (B, T, D)
        """
        seq_len = x.size(1)
        cos, sin = self.rope(seq_len)

        # Determine attention mode
        block_mask = get_block_mask(self._mask_fn, seq_len, x.device, self._mask_cache)
        is_causal = self.causal and self.window_size is None

        if self.use_activation_checkpointing:
            def run_layer(layer, x):
                x, _ = layer(x, cos, sin, block_mask=block_mask, is_causal=is_causal)
                return x

            x = _run_checkpointed_layers(self.layers, x, run_layer)
        else:
            for layer in self.layers:
                x, _ = layer(x, cos, sin, block_mask=block_mask, is_causal=is_causal)

        return self.norm(x)
