"""Multi-modal temporal transformer with axial attention + per-modality CLS head.

Backbone alternates between:
1. Temporal self-attention (causal, per-modality, with RoPE) — each modality
   processes its time series independently.
2. Modality self-attention (per-timestep, bidirectional) — modalities attend
   across each other at each timestep.

A separate ``ModalityCrossAttentionHead`` runs once after the backbone. A
single learned CLS embedding is broadcast to ``M`` query positions (one per
modality) and cross-attends to the backbone's modality outputs; each query's
key/value subset is restricted to members of the same CRP group as that
modality. Output shape ``(B, M, S, D)`` is static regardless of the per-sample
group count. Within a group the M rows are identical by construction — the
diversity lives across groups, which gives multiple multi-modal subset
representations per forward for free.

``use_cls=False`` disables the head entirely — no CLS tensor is returned.

``modality_attn_start_layer`` skips cross-modality attention for the first N
temporal blocks. ``modality_grouping_alpha`` (train only) draws a per-sample
CRP partition over modalities; both the backbone modality↔modality attention
and the CLS cross-attention are restricted to within-group pairs.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hypnos.models.attention import (
    ModalityAttentionLayer,
    RoPETransformerLayer,
    RotaryEmbedding,
    _make_bidirectional_sliding_window_mask,
    _make_causal_sliding_window_mask,
    _make_ffn,
    get_block_mask,
    gradient_checkpoint,
)


def _sample_crp_batch(B: int, M: int, alpha: float, device: torch.device) -> Tensor:
    """Sample B independent Chinese Restaurant Process partitions over M elements.

    Each row is a partition of {0, ..., M-1} encoded as canonical first-appearance
    group IDs (group containing element 0 is always group 0, the next new group is
    1, and so on). Smaller ``alpha`` biases toward few large groups; larger ``alpha``
    biases toward many small groups.

    Args:
        B: Batch size (number of independent partitions to draw).
        M: Number of elements per partition.
        alpha: CRP concentration parameter (must be > 0).
        device: Device for the returned tensor.

    Returns:
        (B, M) long tensor of group IDs.
    """
    if M == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=device)

    group_ids = torch.zeros(B, M, dtype=torch.long, device=device)
    group_sizes = torch.zeros(B, M, device=device)
    group_sizes[:, 0] = 1.0
    num_groups = torch.ones(B, dtype=torch.long, device=device)

    slot_idx = torch.arange(M, device=device).unsqueeze(0)
    ones_col = torch.ones(B, 1, device=device)

    for m in range(1, M):
        probs = group_sizes.clone()
        probs.scatter_(1, num_groups.unsqueeze(1), alpha)
        valid = (slot_idx <= num_groups.unsqueeze(1)).to(probs.dtype)
        probs = probs * valid

        chosen = torch.multinomial(probs, 1).squeeze(-1)
        group_ids[:, m] = chosen

        group_sizes.scatter_add_(1, chosen.unsqueeze(1), ones_col)
        num_groups = num_groups + (chosen == num_groups).long()

    return group_ids


class ModalityCrossAttentionHead(nn.Module):
    """Per-modality CLS cross-attention head.

    Broadcasts a single learned CLS token to ``M`` query positions (one per
    modality) and cross-attends to the backbone's modality outputs. Each
    query's key/value subset is selected by the attention mask — at train
    time, restricted to members of the same CRP group as that modality;
    at eval time, all present modalities. Keys/values carry a learned
    modality-id embedding so the CLS can disambiguate sources.

    Queries are identical across positions by design — the per-query subset
    mask gives them different outputs. Within a group the M rows are
    identical by construction.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        num_modalities: Number of modalities ``M``.
        layer_scale_init: Initial value for LayerScale parameters.
        dropout: Dropout probability.
        swiglu: Use SwiGLU FFN.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.num_modalities = num_modalities

        self.cls_embed = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        self.modality_id_embed = nn.Embedding(num_modalities, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = _make_ffn(d_model, dim_feedforward, swiglu)

        self.layer_scale_attn = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.layer_scale_ffn = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_out: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run the CLS cross-attention head.

        Args:
            modality_out: (B, M, S, D) backbone outputs (keys/values source).
            attn_mask: Additive float mask (0 attend, -inf block), shape
                (B, 1, M, M). Expanded across S internally. ``None`` = full
                attention across all modalities.

        Returns:
            (B, M, S, D) per-modality CLS representations.
        """
        B, M, S, D = modality_out.shape

        q_tokens = self.cls_embed.expand(B, M, S, D)

        positions = torch.arange(M, device=modality_out.device)
        mod_id = self.modality_id_embed(positions)  # (M, D)
        kv_tokens = modality_out + mod_id.view(1, M, 1, D)

        q_flat = q_tokens.permute(0, 2, 1, 3).reshape(B * S, M, D)
        kv_flat = kv_tokens.permute(0, 2, 1, 3).reshape(B * S, M, D)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, S, -1, -1)
            attn_mask = attn_mask.reshape(B * S, 1, M, M)

        q_n = self.norm_q(q_flat)
        kv_n = self.norm_kv(kv_flat)
        q = self.q_proj(q_n).view(B * S, M, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_n).view(B * S, M, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_n).view(B * S, M, self.nhead, self.head_dim).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if attn_mask is not None:
                h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B * S, M, D)
        h = self.out_proj(h)
        h = self.dropout(h)
        out = q_flat + self.layer_scale_attn * h

        ffn_out = self.ffn(self.norm_ffn(out))
        ffn_out = self.dropout(ffn_out)
        out = out + self.layer_scale_ffn * ffn_out

        return out.reshape(B, S, M, D).permute(0, 2, 1, 3).contiguous()


class MultiModalTemporalTransformer(nn.Module):
    """Axial temporal-modality transformer with a per-modality CLS cross-attention head.

    Backbone alternates temporal attention per-modality (causal, RoPE) with
    modality attention per-timestep (bidirectional within CRP group). A
    post-backbone ``ModalityCrossAttentionHead`` produces per-modality CLS
    representations when ``use_cls=True``.

    Args:
        embed_dim: Model dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        num_modalities: Number of modalities (M).
        modality_attn_every_n: Apply modality attention every N blocks (1 = every block).
        modality_attn_start_layer: Skip modality attention for the first N temporal
            blocks. 0 = current behavior.
        modality_grouping_alpha: CRP concentration for per-sample modality grouping at
            train time. None = disabled. When set, the same per-sample CRP partition
            drives both backbone modality↔modality attention and the CLS cross-attention.
            Disabled in eval mode.
        random_subset_masking: When True (train only), bypass CRP attention grouping and
            instead sample a single random subset per sample via CRP, picking a
            uniform-random modality and using its CRP group as a binary
            ``modality_mask``. Requires ``modality_grouping_alpha`` to be set (used as
            the alpha for the subset-sampling CRP). Mutually exclusive with multi-group
            CRP attention masking — group_ids passed to attention masks become None.
        modality_dropout_p: When set (train only), independently drop each modality
            per sample with this probability (Bernoulli). Rows whose dataloader
            ``modality_mask`` had at least one present modality always retain at least
            one (a uniformly-chosen survivor is rescued); rows already empty in the
            dataloader mask are left empty. Train-only alternative to the CRP-correlated
            ``random_subset_masking`` mode. Mutually exclusive with
            ``modality_grouping_alpha``, ``random_subset_masking``, and
            ``disable_cross_modal_attention``.
        disable_cross_modal_attention: Ablation switch. When True, every modality is
            forced into its own singleton group so each modality attends only to itself
            in both the backbone modality attention and the CLS cross-attention — i.e.
            no cross-modal information exchange at all. Unlike CRP grouping this applies
            in BOTH train and eval (deterministic, not sampled), so the model is trained
            and evaluated under the same isolation. Overrides / disables CRP grouping;
            mutually exclusive with ``random_subset_masking`` and ``modality_dropout_p``.
        layer_scale_init: LayerScale initial value.
        max_seq_len: Initial sequence length for RoPE precomputation.
        dropout: Dropout probability.
        sliding_window: Local window size for temporal attention (None = full causal).
        global_every_n: Every Nth layer uses full causal instead of sliding window.
        global_window: Window size for "global" layers selected by ``global_every_n``.
        use_activation_checkpointing: Gradient checkpointing for memory savings.
        qk_norm: Apply QK-norm in temporal attention layers.
        use_cls: When True (default), instantiate the CLS cross-attention head and
            return per-modality CLS outputs from ``forward``. When False, no CLS
            head is created and ``forward`` returns ``(backbone_out, None)``.
        cls_independent_crp: When True, the CLS cross-attention head samples its
            OWN CRP partition per forward (independent of the backbone's). Each
            modality *i*'s ``cls_context[i]`` then fuses a subset that ``mod_ctx[i]``
            did not see, so CLS carries genuinely additive information. Default
            False: CLS head reuses the backbone's partition (current behaviour —
            CLS represents exactly the same group depth already consumes).
            Only meaningful when ``use_cls=True`` and
            ``modality_grouping_alpha`` is set.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_modalities: int = 2,
        modality_attn_every_n: int = 1,
        modality_attn_start_layer: int = 0,
        modality_grouping_alpha: float | None = None,
        random_subset_masking: bool = False,
        modality_dropout_p: float | None = None,
        disable_cross_modal_attention: bool = False,
        layer_scale_init: float = 0.01,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        sliding_window: int | None = None,
        global_every_n: int | None = None,
        global_window: int | None = None,
        use_activation_checkpointing: bool = False,
        qk_norm: bool = False,
        swiglu: bool = False,
        xsa: bool = False,
        use_cls: bool = True,
        cls_independent_crp: bool = False,
        modality_use_ffn: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        if modality_attn_start_layer < 0:
            raise ValueError(f"modality_attn_start_layer must be >= 0, got {modality_attn_start_layer}")
        if modality_grouping_alpha is not None and modality_grouping_alpha <= 0:
            raise ValueError(f"modality_grouping_alpha must be > 0 when set, got {modality_grouping_alpha}")
        if random_subset_masking and modality_grouping_alpha is None:
            raise ValueError("random_subset_masking=True requires modality_grouping_alpha to be set")
        if disable_cross_modal_attention and random_subset_masking:
            raise ValueError("disable_cross_modal_attention and random_subset_masking are mutually exclusive")
        if modality_dropout_p is not None:
            if not 0.0 <= modality_dropout_p <= 1.0:
                raise ValueError(f"modality_dropout_p must be in [0, 1] when set, got {modality_dropout_p}")
            if modality_grouping_alpha is not None:
                raise ValueError("modality_dropout_p and modality_grouping_alpha are mutually exclusive")
            if random_subset_masking:
                raise ValueError("modality_dropout_p and random_subset_masking are mutually exclusive")
            if disable_cross_modal_attention:
                raise ValueError("modality_dropout_p and disable_cross_modal_attention are mutually exclusive")

        self.num_modalities = num_modalities
        self.causal = causal
        self.use_cls = use_cls
        self.cls_independent_crp = cls_independent_crp
        self.modality_attn_every_n = modality_attn_every_n
        self.modality_attn_start_layer = modality_attn_start_layer
        self.modality_grouping_alpha = modality_grouping_alpha
        self.random_subset_masking = random_subset_masking
        self.modality_dropout_p = modality_dropout_p
        self.disable_cross_modal_attention = disable_cross_modal_attention
        self.depth = depth
        self.use_activation_checkpointing = use_activation_checkpointing
        head_dim = embed_dim // num_heads

        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

        # Static structural modality attention mask — full attention between all
        # M modalities. Kept as a buffer (and referenced for device lookup) so
        # callers that pass modality_mask=None without a CUDA tensor still work.
        self.register_buffer(
            "modality_attn_mask",
            torch.ones(num_modalities, num_modalities, dtype=torch.bool),
        )

        self.temporal_layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    layer_scale_init=layer_scale_init,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    swiglu=swiglu,
                    xsa=xsa,
                )
                for _ in range(depth)
            ]
        )

        num_modality_layers = sum(1 for i in range(depth) if self._should_run_modality_attn(i))
        self.modality_layers = nn.ModuleList(
            [
                ModalityAttentionLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    num_modalities=num_modalities,
                    layer_scale_init=layer_scale_init,
                    dropout=dropout,
                    swiglu=swiglu,
                    use_ffn=modality_use_ffn,
                )
                for _ in range(num_modality_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        if use_cls:
            self.cls_head = ModalityCrossAttentionHead(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                num_modalities=num_modalities,
                layer_scale_init=layer_scale_init,
                dropout=dropout,
                swiglu=swiglu,
            )

        if sliding_window is None and global_window is None:
            self._per_layer_masks: list[tuple] | None = None
        else:
            make_sliding = _make_causal_sliding_window_mask if causal else _make_bidirectional_sliding_window_mask
            local_mask_fn = make_sliding(sliding_window) if sliding_window is not None else None
            global_mask_fn = make_sliding(global_window) if global_window is not None else None
            self._per_layer_masks = []
            for i in range(depth):
                if global_every_n is not None and (i + 1) % global_every_n == 0:
                    self._per_layer_masks.append((global_mask_fn, {}))
                else:
                    self._per_layer_masks.append((local_mask_fn, {}))

    def _should_run_modality_attn(self, i: int) -> bool:
        """Whether temporal block ``i`` is followed by a modality attention layer."""
        if i < self.modality_attn_start_layer:
            return False
        relative = i - self.modality_attn_start_layer
        return (relative + 1) % self.modality_attn_every_n == 0

    def _modality_layer_idx(self, i: int) -> int:
        """Compute modality-layer index from temporal-layer index (deterministic, no state)."""
        return sum(1 for j in range(i) if self._should_run_modality_attn(j))

    def sample_group_ids(self, B: int, device: torch.device) -> Tensor | None:
        """Sample a per-sample CRP modality partition when grouping is active.

        Returns ``None`` when grouping is disabled (eval mode,
        ``modality_grouping_alpha=None``, ``random_subset_masking=True``,
        ``modality_dropout_p`` set, or ``disable_cross_modal_attention=True`` —
        the latter forces singleton isolation directly in the mask builders
        instead). The same ``group_ids`` tensor should be used for both
        ``build_modality_attn_mask`` and ``build_cross_attn_mask`` in a single
        forward.
        """
        use_grouping = (
            self.training
            and self.modality_grouping_alpha is not None
            and not self.random_subset_masking
            and not self.disable_cross_modal_attention
            and self.num_modalities > 1
        )
        if not use_grouping:
            return None
        return _sample_crp_batch(B, self.num_modalities, self.modality_grouping_alpha, device)

    def sample_random_subset_mask(self, B: int, device: torch.device) -> Tensor | None:
        """Sample a per-sample binary modality mask matched to one CRP group.

        Draws a CRP partition with the configured ``modality_grouping_alpha``,
        picks a uniform-random modality per row, and returns the mask of
        modalities sharing that modality's group. The resulting subset
        distribution is identical to what a single modality "sees" under CRP —
        so this exposes the same per-forward subset distribution as CRP
        attention masking, but with only one subset per forward instead of M.

        Returns ``None`` when not in training mode or when
        ``random_subset_masking`` is disabled.
        """
        active = self.training and self.random_subset_masking and self.num_modalities > 1
        if not active:
            return None
        M = self.num_modalities
        group_ids = _sample_crp_batch(B, M, self.modality_grouping_alpha, device)  # (B, M)
        chosen_idx = torch.randint(0, M, (B,), device=device)  # (B,)
        chosen_group = group_ids.gather(1, chosen_idx.unsqueeze(1))  # (B, 1)
        return group_ids == chosen_group  # (B, M) bool

    def sample_modality_dropout_mask(
        self,
        B: int,
        device: torch.device,
        modality_mask: Tensor | None = None,
    ) -> Tensor | None:
        """Sample a per-sample binary modality mask via independent Bernoulli dropout.

        Each modality is independently kept with probability ``1 - modality_dropout_p``.
        The dropout mask is ANDed with the dataloader-provided ``modality_mask`` so
        dropped-out and dataloader-absent modalities are both treated as absent.

        Rescue rule: for any row that loses every modality after the AND but had at
        least one present in ``modality_mask``, one survivor is drawn uniformly from
        that row's original present set. Rows whose dataloader mask was already empty
        stay empty (the loss path already handles all-absent rows).

        Returns ``None`` when not in training mode, ``modality_dropout_p is None``, or
        ``num_modalities <= 1``.
        """
        active = (
            self.training
            and self.modality_dropout_p is not None
            and self.modality_dropout_p > 0.0
            and self.num_modalities > 1
        )
        if not active:
            return None
        M = self.num_modalities
        keep_prob = 1.0 - float(self.modality_dropout_p)
        dropout_mask = torch.bernoulli(torch.full((B, M), keep_prob, device=device)).bool()
        if modality_mask is not None:
            dropout_mask = dropout_mask & modality_mask
            candidates = modality_mask
        else:
            candidates = torch.ones(B, M, dtype=torch.bool, device=device)
        empty = ~dropout_mask.any(dim=1)  # (B,)
        rescuable = candidates.any(dim=1)  # (B,)
        need_rescue = empty & rescuable
        if need_rescue.any():
            # multinomial requires nonzero weights per row — give dummy weights to
            # rows we won't actually use (we mask them out via torch.where below).
            weights = candidates.float()
            weights = torch.where(need_rescue.unsqueeze(1), weights, torch.ones_like(weights))
            rescue_idx = torch.multinomial(weights, 1).squeeze(-1)  # (B,)
            eye = torch.eye(M, dtype=torch.bool, device=device)
            rescue = eye[rescue_idx]  # (B, M)
            dropout_mask = torch.where(need_rescue.unsqueeze(1), rescue, dropout_mask)
        return dropout_mask

    def build_modality_attn_mask(
        self,
        modality_mask: Tensor | None,
        B: int,
        group_ids: Tensor | None = None,
    ) -> Tensor:
        """Pre-compute backbone modality attention mask. Call OUTSIDE torch.compile region.

        Always returns ``(B, 1, M, M)`` for consistent compiled graph caching.
        The temporal forward expands across S internally.

        Args:
            modality_mask: (B, M) bool where True = present, or None = all present.
            B: batch size.
            group_ids: (B, M) from ``sample_group_ids``, or None to skip grouping.

        Returns:
            Float additive mask (B, 1, M, M) ready for SDPA.
            Values are 0.0 (attend) or -inf (block).
        """
        device = self.modality_attn_mask.device if modality_mask is None else modality_mask.device
        M = self.num_modalities

        # Ablation: force singleton groups so each modality attends only to itself
        # (no cross-modal attention), in both train and eval.
        if self.disable_cross_modal_attention:
            group_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, M)

        # Fast path: all present, no grouping → zero mask (attend all).
        if group_ids is None and (modality_mask is None or modality_mask.all()):
            return torch.zeros(B, 1, M, M, device=device)

        if modality_mask is None:
            full_present = torch.ones(B, M, dtype=torch.bool, device=device)
        else:
            full_present = modality_mask

        bool_mask = full_present.unsqueeze(2) & full_present.unsqueeze(1)  # (B, M, M)

        if group_ids is not None:
            same_group = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)  # (B, M, M)
            bool_mask = bool_mask & same_group

        return torch.where(bool_mask, 0.0, float("-inf")).unsqueeze(1)

    def build_cross_attn_mask(
        self,
        modality_mask: Tensor | None,
        B: int,
        group_ids: Tensor | None = None,
    ) -> Tensor:
        """Pre-compute CLS-head cross-attention mask. Call OUTSIDE torch.compile region.

        Each CLS-query at modality index ``i`` attends only to members of the
        same CRP group as ``i`` (when grouping is active) and only to present
        modalities. Absent query rows are given self-only attention so softmax
        stays well-defined — those outputs are discarded downstream.

        Args:
            modality_mask: (B, M) bool where True = present, or None = all present.
            B: batch size.
            group_ids: (B, M) from ``sample_group_ids``, or None to skip grouping.

        Returns:
            Float additive mask (B, 1, M, M) for the cross-attention head.
        """
        device = self.modality_attn_mask.device if modality_mask is None else modality_mask.device
        M = self.num_modalities

        # Ablation: force singleton groups so each CLS query attends only to its own
        # modality (no cross-modal attention), in both train and eval.
        if self.disable_cross_modal_attention:
            group_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, M)

        # Fast path: all present, no grouping → zero mask (attend all).
        if group_ids is None and (modality_mask is None or modality_mask.all()):
            return torch.zeros(B, 1, M, M, device=device)

        bool_mask = torch.ones(B, M, M, dtype=torch.bool, device=device)

        if group_ids is not None:
            same_group = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)  # (B, M, M)
            bool_mask = bool_mask & same_group

        if modality_mask is not None:
            present_k = modality_mask.unsqueeze(1).expand(-1, M, -1)  # block absent keys
            bool_mask = bool_mask & present_k

            # Absent query rows → self-only diagonal to keep softmax well-defined.
            absent_q = ~modality_mask  # (B, M)
            if absent_q.any():
                eye = torch.eye(M, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
                keep_row = modality_mask.unsqueeze(2)  # (B, M, 1) True where query present
                safe_row = absent_q.unsqueeze(2) & eye  # (B, M, M) diagonal where absent
                bool_mask = (bool_mask & keep_row) | safe_row

        return torch.where(bool_mask, 0.0, float("-inf")).unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        modality_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass with axial temporal-modality attention + CLS head.

        Args:
            x: (B, M, S, D) per-modality temporal embeddings.
            modality_attn_mask: From ``build_modality_attn_mask`` — (B, 1, M, M).
            cross_attn_mask: From ``build_cross_attn_mask`` — (B, 1, M, M).
                Only consumed when ``use_cls=True``.
            position_ids: (M,) long, modality indices used to look up the
                modality positional embeddings in each ``ModalityAttentionLayer``.
                None ⇒ ``arange(M)`` (default training behaviour). Callers that
                pass a sliced ``x`` containing only a subset of the original
                modalities must pass the original modality indices here so
                the learned per-modality positional embeddings are preserved.

        Returns:
            (backbone_out, cls_per_modality):
                backbone_out: (B, M, S, D) contextualised modality outputs.
                cls_per_modality: (B, M, S, D) per-modality CLS outputs when
                    ``use_cls=True``, else ``None``.
        """
        B, M, S, D = x.shape

        cos, sin = self.rope(S)

        block_masks = []
        for i in range(len(self.temporal_layers)):
            if self._per_layer_masks is not None:
                mask_fn, cache = self._per_layer_masks[i]
                block_masks.append(get_block_mask(mask_fn, S, x.device, cache))
            else:
                block_masks.append(None)

        if modality_attn_mask is not None:
            mod_attn_mask = modality_attn_mask.unsqueeze(2).expand(-1, -1, S, -1, -1)
            mod_attn_mask = mod_attn_mask.reshape(B * S, 1, M, M)
        else:
            mod_attn_mask = None

        def _run_step(x, i):
            """Run temporal layer i + optional modality layer. Returns (B, M, S, D)."""
            x_flat = x.reshape(B * M, S, D)
            x_flat, _ = self.temporal_layers[i](
                x_flat,
                cos,
                sin,
                block_mask=block_masks[i],
                is_causal=self.causal,
            )
            x = x_flat.reshape(B, M, S, D)

            if self._should_run_modality_attn(i):
                modality_layer = self.modality_layers[self._modality_layer_idx(i)]
                x_mod = x.permute(0, 2, 1, 3).reshape(B * S, M, D)
                x_mod = modality_layer(x_mod, attn_mask=mod_attn_mask, position_ids=position_ids)
                x = x_mod.reshape(B, S, M, D).permute(0, 2, 1, 3)

            return x

        if self.use_activation_checkpointing and x.requires_grad:

            def _run_step_flat(x_flat, i):
                x = x_flat.reshape(B, M, S, D)
                x = _run_step(x, i)
                return x.reshape(B * M, S, D)

            x_flat = x.reshape(B * M, S, D)

            for i in range(len(self.temporal_layers)):
                x_flat = gradient_checkpoint(_run_step_flat, x_flat, i, use_reentrant=False)

            x = x_flat.reshape(B, M, S, D)
        else:
            for i in range(len(self.temporal_layers)):
                x = _run_step(x, i)

        x_flat = x.reshape(B * M, S, D)
        x_flat = self.norm(x_flat)
        backbone_out = x_flat.reshape(B, M, S, D)

        cls_per_modality = None
        if self.use_cls:
            if self.use_activation_checkpointing and backbone_out.requires_grad:
                cls_per_modality = gradient_checkpoint(
                    self.cls_head,
                    backbone_out,
                    cross_attn_mask,
                    use_reentrant=False,
                )
            else:
                cls_per_modality = self.cls_head(backbone_out, attn_mask=cross_attn_mask)

        return backbone_out, cls_per_modality
