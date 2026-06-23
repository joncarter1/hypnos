"""MultiModalRQTransformer: multi-modal next-token prediction over RVQ token streams.

Two-level architecture (Moshi/Kyutai style): a shared temporal transformer with
axial temporal-modality attention aggregates per-modality embeddings over time, and a shared
depth transformer generates the K codebook tokens per timestep. For embedding inference only
the temporal transformer is used (see ``hypnos.embedding.infer``).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hypnos.models.attention import gradient_checkpoint
from hypnos.models.rq_transformer.depth import SharedDepthTransformer
from hypnos.models.rq_transformer.multimodal_temporal import MultiModalTemporalTransformer
from hypnos.settings import NUM_KNOWN_CHANNELS


@dataclass
class ModalityConfig:
    """Configuration for a single modality.

    ``name`` is the unique modality identifier (e.g. ``'eeg_c3'``) used for
    model bookkeeping. ``signal_type`` is the broad signal class (e.g.
    ``'eeg'``) used for canonical metric paths so that runs sharing a
    signal_type can be compared regardless of run shape.
    """

    name: str
    num_quantizers: int
    codebook_size: int
    signal_type: str = ""


def _compute_per_level_loss(
    hidden: Tensor,
    target_tokens: Tensor,
    output_heads: nn.ModuleList,
    K: int,
    codebook_size: int,
    device: torch.device,
    use_activation_checkpointing: bool = False,
    sample_mask: Tensor | None = None,
    token_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute per-level cross-entropy loss and accuracy.

    When ``use_activation_checkpointing`` is True, each level's linear
    projection + cross-entropy is wrapped in ``gradient_checkpoint`` so
    that the logits (~2 GB per level) are not retained for backward.
    They are recomputed during backward instead.

    Per-sample loss/accuracy are metric-only reductions over `(S, K)` for
    each batch element, computed without grad — the trainer uses them for
    per-channel breakdowns and never backpropagates through them.

    Args:
        hidden: (B, S, K, depth_dim) depth transformer hidden states.
        target_tokens: (B, S, K) ground truth token indices.
        output_heads: ModuleList of K linear layers projecting to codebook_size.
        K: Number of quantizer levels.
        codebook_size: Vocabulary size.
        device: Device for output tensors.
        use_activation_checkpointing: Checkpoint the logit projection + CE.
        sample_mask: (B,) bool where True = include in loss. None = all included.
        token_mask: (B, S) bool where True = score this position. None = all positions.
            Used by the masked objective to restrict CE to masked timesteps. ANDed with
            ``sample_mask`` (broadcast over S) when both are given.

    Returns:
        Tuple of (per_level_loss, per_level_accuracy, per_sample_loss, per_sample_correct,
        per_level_logit_max, per_level_logz).
        per_sample_loss and per_sample_correct have shape (B,). For samples
        masked out by ``sample_mask`` the per-sample values are still
        populated (unmasked) and meaningless; callers must filter using the
        same mask. per_level_logit_max and per_level_logz are diagnostics for
        tied-embedding norm-growth instability (see PaLM z-loss / Wortsman 2023):
        logit_max tracks |logit|.max() per level; logz tracks mean
        logsumexp(logits) per level — the quantity z-loss would penalize.
    """
    B, S = hidden.shape[:2]
    per_level_loss = torch.zeros(K, device=device)
    per_level_accuracy = torch.zeros(K, device=device)
    per_sample_loss = torch.zeros(B, device=device)
    per_sample_correct = torch.zeros(B, device=device)
    per_level_logit_max = torch.zeros(K, device=device)
    per_level_logz = torch.zeros(K, device=device)

    # Build per-token weight mask: (B,) and/or (B, S) → (B*S,) for masked loss averaging.
    # sample_mask gates whole samples (absent modalities); token_mask gates positions
    # (masked timesteps in the masked objective). Either may be None.
    if sample_mask is not None or token_mask is not None:
        weight = torch.ones(B, S, device=device)
        if sample_mask is not None:
            weight = weight * sample_mask.float().unsqueeze(1)
        if token_mask is not None:
            weight = weight * token_mask.float()
        token_weight = weight.reshape(-1)  # (B*S,)
        n_tokens = token_weight.sum().clamp(min=1)
    else:
        token_weight = None
        n_tokens = None

    for k in range(K):
        h = hidden[:, :, k].reshape(-1, hidden.size(-1))
        targets = target_tokens[:, :, k].reshape(-1)

        if use_activation_checkpointing and h.requires_grad:

            def _level_loss(h, targets, _head=output_heads[k], _tw=token_weight, _nt=n_tokens):
                loss = F.cross_entropy(_head(h), targets, reduction="none")
                if _tw is not None:
                    return (loss * _tw).sum() / _nt
                return loss.mean()

            per_level_loss[k] = gradient_checkpoint(_level_loss, h, targets, use_reentrant=False)

            # Metric-only second forward: combined loss/accuracy/per-sample reductions.
            with torch.no_grad():
                logits = output_heads[k](h)
                loss_full = F.cross_entropy(logits, targets, reduction="none")
                correct = (logits.argmax(dim=-1) == targets).float()
                if token_weight is not None:
                    per_level_accuracy[k] = (correct * token_weight).sum() / n_tokens
                else:
                    per_level_accuracy[k] = correct.mean()
                per_sample_loss += loss_full.reshape(B, S).mean(dim=1) / K
                per_sample_correct += correct.reshape(B, S).mean(dim=1) / K
                per_level_logit_max[k] = logits.abs().max()
                per_level_logz[k] = torch.logsumexp(logits, dim=-1).mean()
        else:
            logits = output_heads[k](h).reshape(-1, codebook_size)
            loss = F.cross_entropy(logits, targets, reduction="none")
            correct = (logits.argmax(dim=-1) == targets).float()
            if token_weight is not None:
                per_level_loss[k] = (loss * token_weight).sum() / n_tokens
                per_level_accuracy[k] = (correct * token_weight).sum() / n_tokens
            else:
                per_level_loss[k] = loss.mean()
                per_level_accuracy[k] = correct.mean()
            per_sample_loss = per_sample_loss + loss.detach().reshape(B, S).mean(dim=1) / K
            per_sample_correct = per_sample_correct + correct.reshape(B, S).mean(dim=1) / K
            with torch.no_grad():
                per_level_logit_max[k] = logits.detach().abs().max()
                per_level_logz[k] = torch.logsumexp(logits.detach(), dim=-1).mean()

    return (
        per_level_loss,
        per_level_accuracy,
        per_sample_loss,
        per_sample_correct,
        per_level_logit_max,
        per_level_logz,
    )


def _init_weights(module: nn.Module) -> None:
    """Initialize weights following GPT-2 conventions."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


class _ProjectedDepthEmbedding(nn.Module):
    """Depth code-embedding as a shared linear projection of the trunk codebook.

    Trick 2: holds a *reference* to the trunk's ``nn.Embedding`` (shared, not
    duplicated) and one ``proj`` (``embed_dim -> depth_dim``) reused across all
    RVQ levels, so the codebook lives only in the trunk's ``token_embeddings``.
    """

    def __init__(self, trunk_emb: nn.Embedding, proj: nn.Module):
        super().__init__()
        self.trunk_emb = trunk_emb
        self.proj = proj

    def forward(self, idx: Tensor) -> Tensor:
        return self.proj(self.trunk_emb(idx))


class _TiedTrunkHead(nn.Module):
    """Output head whose unembedding is the shared trunk codebook.

    Combined trick 1 ∘ trick 2: ``logits = down_proj(h) @ trunk_emb.weight.T +
    bias`` where ``down_proj`` is ``depth_dim -> embed_dim``. No per-level head
    weight matrix — only a per-level bias and the (shared) projection.
    """

    def __init__(self, trunk_emb: nn.Embedding, down_proj: nn.Module, codebook_size: int):
        super().__init__()
        self.trunk_emb = trunk_emb
        self.down_proj = down_proj  # depth_dim -> embed_dim
        self.bias = nn.Parameter(torch.zeros(codebook_size))

    def forward(self, h: Tensor) -> Tensor:
        return F.linear(self.down_proj(h), self.trunk_emb.weight, self.bias)


class MultiModalRQTransformer(nn.Module):
    """Hypnos RQ-Transformer with axial temporal-modality attention.

    Uses a shared temporal transformer with alternating temporal and modality
    attention, plus independent per-modality depth transformers.

    Args:
        modality_configs: List of ModalityConfig for each modality.
        embed_dim: Temporal transformer embedding dimension.
        temporal_depth: Number of temporal transformer blocks.
        temporal_heads: Number of temporal attention heads.
        temporal_mlp_ratio: MLP ratio for temporal FFN.
        depth_dim: Depth transformer dimension (None = embed_dim).
        depth_depth: Number of depth transformer layers per modality.
        depth_heads: Number of depth attention heads.
        depth_mlp_ratio: MLP ratio for depth FFN.
        modality_attn_every_n: Apply modality attention every N blocks.
        layer_scale_init: LayerScale initial value.
        max_seq_len: Maximum sequence length for RoPE.
        dropout: Dropout probability.
        sliding_window: Sliding window size for temporal attention.
        use_activation_checkpointing: Gradient checkpointing.
        qk_norm: QK-norm in temporal attention.
        channel_embeddings: Enable learned channel embeddings.
        modality_loss_weights: Per-modality loss weights (None = uniform).
        use_cls: If True (default), enable the per-modality CLS cross-attention
            head inside the temporal transformer. Each modality gets a CLS
            representation (group-specific at train time, all-modalities at eval)
            that is fed as extra conditioning to that modality's depth transformer.
            False removes the head entirely and omits ``'cls'`` from the returned
            ``embeddings`` dict.
    """

    def __init__(
        self,
        modality_configs: list[ModalityConfig],
        embed_dim: int = 512,
        temporal_depth: int = 8,
        temporal_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,
        depth_dim: int | None = None,
        depth_depth: int = 4,
        depth_heads: int = 4,
        depth_mlp_ratio: float = 4.0,
        modality_attn_every_n: int = 1,
        modality_attn_start_layer: int = 0,
        modality_grouping_alpha: float | None = None,
        random_subset_masking: bool = False,
        modality_dropout_p: float | None = None,
        disable_cross_modal_attention: bool = False,
        mod_context_dropout_p: float = 0.0,
        layer_scale_init: float = 0.01,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        sliding_window: int | None = None,
        global_every_n: int | None = None,
        global_window: int | None = None,
        use_activation_checkpointing: bool = False,
        qk_norm: bool = False,
        channel_embeddings: bool = False,
        modality_loss_weights: dict[str, float] | None = None,
        swiglu: bool = False,
        xsa: bool = False,
        use_cls: bool = True,
        cls_independent_crp: bool = False,
        modality_use_ffn: bool = True,
        tie_depth_io: bool = False,
        share_trunk_depth_emb: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        self.modality_configs = modality_configs
        self.embed_dim = embed_dim
        self.modality_loss_weights = modality_loss_weights
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_cls = use_cls
        self.cls_independent_crp = cls_independent_crp

        depth_dim = depth_dim or embed_dim
        num_modalities = len(modality_configs)

        # Validate signal_type is set when there are multiple modalities. Without it, the grouping key
        # falls back to the empty string and all modalities with the same (K, V) get
        # forced to share a single codebook regardless of their actual tokenizer.
        if num_modalities > 1:
            missing = [mc.name for mc in modality_configs if not mc.signal_type]
            if missing:
                raise ValueError(
                    f"Setups with multiple modalities require non-empty signal_type on every ModalityConfig. "
                    f"Missing signal_type for: {missing}"
                )

        # Compute depth layout: flat mapping from token column index → (modality_name, quantizer_idx)
        self.total_depth = sum(mc.num_quantizers for mc in modality_configs)
        self._modality_offsets: dict[str, tuple[int, int]] = {}  # name → (start, end) in total_K
        offset = 0
        for mc in modality_configs:
            self._modality_offsets[mc.name] = (offset, offset + mc.num_quantizers)
            offset += mc.num_quantizers

        # Group modalities by tokenizer identity for weight sharing.
        # Modalities sharing (signal_type, num_quantizers, codebook_size) are assumed to
        # come from the same tokenizer — they share embedding tables and output heads.
        # Using signal_type in the key prevents distinct tokenizers that happen to match
        # (K, V) (e.g. EEG vs EOG both with K=8, V=const) from being forced into one group.
        tokenizer_groups: dict[tuple[str, int, int], str] = {}  # (signal_type, K, V) → leader name
        for mc in modality_configs:
            key = (mc.signal_type, mc.num_quantizers, mc.codebook_size)
            if key not in tokenizer_groups:
                tokenizer_groups[key] = mc.name
        self._weight_group: dict[str, str] = {
            mc.name: tokenizer_groups[(mc.signal_type, mc.num_quantizers, mc.codebook_size)] for mc in modality_configs
        }

        # Token embeddings and BOS — only create for group leaders
        self.token_embeddings = nn.ModuleDict()
        self.bos_embeds = nn.ParameterDict()
        for mc in modality_configs:
            if self._weight_group[mc.name] == mc.name:
                self.token_embeddings[mc.name] = nn.ModuleList(
                    [nn.Embedding(mc.codebook_size, embed_dim) for _ in range(mc.num_quantizers)]
                )
                self.bos_embeds[mc.name] = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Optional channel embeddings
        self.channel_embed = nn.Embedding(NUM_KNOWN_CHANNELS, embed_dim) if channel_embeddings else None

        # Temporal transformer (axial attention)
        dim_feedforward = int(embed_dim * temporal_mlp_ratio)
        self.temporal_transformer = MultiModalTemporalTransformer(
            embed_dim=embed_dim,
            depth=temporal_depth,
            num_heads=temporal_heads,
            dim_feedforward=dim_feedforward,
            num_modalities=num_modalities,
            modality_attn_every_n=modality_attn_every_n,
            modality_attn_start_layer=modality_attn_start_layer,
            modality_grouping_alpha=modality_grouping_alpha,
            random_subset_masking=random_subset_masking,
            modality_dropout_p=modality_dropout_p,
            disable_cross_modal_attention=disable_cross_modal_attention,
            layer_scale_init=layer_scale_init,
            max_seq_len=max_seq_len,
            dropout=dropout,
            sliding_window=sliding_window,
            global_every_n=global_every_n,
            global_window=global_window,
            use_activation_checkpointing=use_activation_checkpointing,
            qk_norm=qk_norm,
            swiglu=swiglu,
            xsa=xsa,
            use_cls=use_cls,
            cls_independent_crp=cls_independent_crp,
            modality_use_ffn=modality_use_ffn,
            causal=causal,
        )

        # Single shared depth transformer: shared layers/projections/pos_embed/null_mod_ctx,
        # per-group embedding tables and output heads, per-modality modality_embeddings.
        self.depth_transformer = SharedDepthTransformer(
            modality_configs=modality_configs,
            weight_group=self._weight_group,
            embed_dim=embed_dim,
            depth_dim=depth_dim,
            depth=depth_depth,
            num_heads=depth_heads,
            dim_feedforward=int(depth_dim * depth_mlp_ratio),
            dropout=dropout,
            use_activation_checkpointing=use_activation_checkpointing,
            swiglu=swiglu,
            has_cls_context=use_cls,
            mod_context_dropout_p=mod_context_dropout_p,
        )

        # Codebook sharing flags (trick 1 / trick 2).
        self.tie_depth_io = tie_depth_io
        self.share_trunk_depth_emb = share_trunk_depth_emb
        leaders = [mc.name for mc in modality_configs if self._weight_group[mc.name] == mc.name]
        self._leader_kv = {
            mc.name: (mc.num_quantizers, mc.codebook_size)
            for mc in modality_configs
            if self._weight_group[mc.name] == mc.name
        }

        # Share embedding weights between temporal and depth (per group).
        if not (tie_depth_io or share_trunk_depth_emb):
            # Original behavior — baseline state-dict identical.
            if embed_dim == depth_dim:
                for leader in leaders:
                    K_g, _ = self._leader_kv[leader]
                    for k in range(K_g):
                        self.depth_transformer.per_group_embeddings[leader][k] = self.token_embeddings[leader][k]

        # Per-group trick-2 projections, created pre-init so _init_weights sets them.
        if share_trunk_depth_emb:
            self.trunk_to_depth_proj = nn.ModuleDict(
                {leader: nn.Linear(embed_dim, depth_dim, bias=False) for leader in leaders}
            )
            if tie_depth_io:
                self.depth_to_trunk_proj = nn.ModuleDict(
                    {leader: nn.Linear(depth_dim, embed_dim, bias=False) for leader in leaders}
                )

        self.apply(_init_weights)

        # Codebook tying — post-init so weight ties aren't clobbered. No-op when
        # both flags are False (see above).
        self._wire_codebook_sharing(leaders)

    def _wire_codebook_sharing(self, leaders: list[str]) -> None:
        """Per-group trick 1 / trick 2 / combined for the shared depth transformer.

        Iterates weight-group leaders: each group's depth tables collapse onto that group's
        trunk codebook via per-group projections (independent trunk→depth maps per
        signal type — negligible params vs the per-group tables removed).
        """
        dt = self.depth_transformer
        tie, share = self.tie_depth_io, self.share_trunk_depth_emb
        if not (tie or share):
            return
        for leader in leaders:
            K_g, V_g = self._leader_kv[leader]
            if share:
                for k in range(K_g):
                    dt.per_group_embeddings[leader][k] = _ProjectedDepthEmbedding(
                        self.token_embeddings[leader][k], self.trunk_to_depth_proj[leader]
                    )
            if tie and share:
                for k in range(K_g):
                    dt.per_group_output_heads[leader][k] = _TiedTrunkHead(
                        self.token_embeddings[leader][k], self.depth_to_trunk_proj[leader], V_g
                    )
            elif tie:
                for k in range(K_g):
                    dt.per_group_output_heads[leader][k].weight = dt.per_group_embeddings[leader][k].weight

    def _aggregate_embeddings(self, tokens: Tensor) -> Tensor:
        """Sum per-quantizer embeddings per modality, stack across modalities.

        Args:
            tokens: (B, S, total_K) token indices.

        Returns:
            aggregated: (B, M, S, D) per-modality aggregated embeddings.
        """
        per_modality = []
        for mc in self.modality_configs:
            start, end = self._modality_offsets[mc.name]
            leader = self._weight_group[mc.name]
            embeds = self.token_embeddings[leader]
            agg = embeds[0](tokens[:, :, start])
            for k in range(1, mc.num_quantizers):
                agg = agg + embeds[k](tokens[:, :, start + k])
            per_modality.append(agg)  # (B, S, D)
        return torch.stack(per_modality, dim=1)  # (B, M, S, D)

    def _build_temporal_input(
        self,
        tokens: Tensor,
        channel_ids: Tensor | None,
        modality_mask: Tensor | None,
        drop_last: bool,
    ) -> Tensor:
        """Build the per-modality BOS-prefixed temporal transformer input.

        ``drop_last=True`` reproduces the teacher-forced training layout (length S);
        ``drop_last=False`` keeps every token so the final position predicts the
        next timestep (length S+1) — used by :meth:`generate`.
        """
        B = tokens.size(0)
        src = tokens[:, :-1] if drop_last else tokens
        agg_embeds = self._aggregate_embeddings(src)  # (B, M, L, D)
        bos_list = [
            self.bos_embeds[self._weight_group[mc.name]].unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, 1, 1, -1)
            for mc in self.modality_configs
        ]
        bos = torch.cat(bos_list, dim=1)  # (B, M, 1, D)
        temporal_input = torch.cat([bos, agg_embeds], dim=2)  # (B, M, L+1, D)

        if modality_mask is not None:
            presence = modality_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, M, 1, 1)
            temporal_input = temporal_input * presence

        if self.channel_embed is not None and channel_ids is not None:
            safe_ids = channel_ids.clamp(min=0)
            ch_embeds = self.channel_embed(safe_ids).unsqueeze(2)  # (B, M, 1, D)
            if modality_mask is not None:
                ch_embeds = ch_embeds * presence
            temporal_input = temporal_input + ch_embeds
        return temporal_input

    def forward(
        self,
        tokens: Tensor,
        channel_ids: Tensor | None = None,
        modality_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass (teacher-forced).

        Args:
            tokens: (B, S, total_K) token indices for all modalities concatenated.
            channel_ids: (B, M) channel index per sample per modality.
            modality_mask: (B, M) bool where True = modality present. None = all present.

        Returns:
            Dict with loss, per_level_loss, per_modality_loss, embeddings, etc.
        """
        B, S, total_K = tokens.shape
        M = len(self.modality_configs)

        # Per-modality aggregation + per-modality BOS (teacher-forced shift).
        # Absent modalities are zeroed; channel embeddings are added per modality.
        temporal_input = self._build_temporal_input(tokens, channel_ids, modality_mask, drop_last=True)  # (B, M, S, D)

        # Pre-compute attention masks OUTSIDE compiled region.
        # Backbone's CRP group sample drives the modality-attention mask.
        backbone_group_ids = self.temporal_transformer.sample_group_ids(B, tokens.device)
        mod_attn_mask = self.temporal_transformer.build_modality_attn_mask(
            modality_mask,
            B,
            group_ids=backbone_group_ids,
        )
        # CLS head: by default reuses the backbone's partition (so CLS summarises
        # the same group depth already consumes). With cls_independent_crp=True,
        # the CLS head samples an independent partition so cls_context[i] carries
        # modality information mod_ctx[i] didn't see.
        if self.use_cls:
            if self.cls_independent_crp:
                cls_group_ids = self.temporal_transformer.sample_group_ids(B, tokens.device)
            else:
                cls_group_ids = backbone_group_ids
            cross_attn_mask = self.temporal_transformer.build_cross_attn_mask(
                modality_mask,
                B,
                group_ids=cls_group_ids,
            )
        else:
            cross_attn_mask = None

        # Backbone returns (B, M, S, D); CLS head returns (B, M, S, D) per-modality.
        temporal_context, cls_per_modality = self.temporal_transformer(
            temporal_input,
            modality_attn_mask=mod_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

        # Per-modality depth transformers + loss computation
        all_per_level_loss = []
        all_per_level_accuracy = []
        all_per_level_logit_max = []
        all_per_level_logz = []
        per_modality_loss: dict[str, Tensor] = {}
        per_modality_accuracy: dict[str, Tensor] = {}
        per_modality_per_sample_loss: dict[str, Tensor] = {}
        per_modality_per_sample_correct: dict[str, Tensor] = {}
        per_modality_logit_max: dict[str, Tensor] = {}
        per_modality_logz: dict[str, Tensor] = {}

        for i, mc in enumerate(self.modality_configs):
            start, end = self._modality_offsets[mc.name]
            modality_context_i = temporal_context[:, i]  # (B, S, D)
            modality_tokens = tokens[:, :, start:end]  # (B, S, K_m)

            # Per-modality CLS: each modality gets its own group-specific CLS context.
            cls_context_i = cls_per_modality[:, i] if cls_per_modality is not None else None

            # Run depth on ALL samples (fixed shapes for compile compatibility).
            # Absent modalities produce garbage outputs — masked out in loss below.
            hidden, output_heads = self.depth_transformer(
                modality_idx=i,
                temporal_context=modality_context_i,
                target_tokens=modality_tokens,
                cls_context=cls_context_i,
            )

            # Compute per-level loss, masking absent samples
            present = modality_mask[:, i] if modality_mask is not None else None

            pl_loss, pl_acc, ps_loss, ps_correct, pl_logit_max, pl_logz = _compute_per_level_loss(
                hidden,
                modality_tokens,
                output_heads,
                mc.num_quantizers,
                mc.codebook_size,
                tokens.device,
                use_activation_checkpointing=self.use_activation_checkpointing,
                sample_mask=present,
            )
            m_loss = pl_loss.mean()

            per_modality_loss[mc.name] = m_loss
            per_modality_accuracy[mc.name] = pl_acc.mean()
            per_modality_per_sample_loss[mc.name] = ps_loss
            per_modality_per_sample_correct[mc.name] = ps_correct
            per_modality_logit_max[mc.name] = pl_logit_max.max()
            per_modality_logz[mc.name] = pl_logz.mean()
            all_per_level_loss.append(pl_loss)
            all_per_level_accuracy.append(pl_acc)
            all_per_level_logit_max.append(pl_logit_max)
            all_per_level_logz.append(pl_logz)

        # Combine modality losses — average over modalities that have any present samples
        present = [i for i in range(M) if modality_mask is None or modality_mask[:, i].any()]
        if not present:
            loss = torch.tensor(0.0, device=tokens.device)
        elif self.modality_loss_weights is not None:
            ws = [self.modality_loss_weights[self.modality_configs[i].name] for i in present]
            ls = [ws[j] * per_modality_loss[self.modality_configs[i].name] for j, i in enumerate(present)]
            loss = sum(ls) / sum(ws)
        else:
            loss = sum(per_modality_loss[self.modality_configs[i].name] for i in present) / len(present)

        per_level_loss = torch.cat(all_per_level_loss)  # (total_K,)
        per_level_accuracy = torch.cat(all_per_level_accuracy)  # (total_K,)
        per_level_logit_max = torch.cat(all_per_level_logit_max)  # (total_K,)
        per_level_logz = torch.cat(all_per_level_logz)  # (total_K,)

        # cls_per_modality (when use_cls=True) flows into each depth transformer as
        # `cls_context_i` earlier in this forward; it is not surfaced here because
        # probes consume the masked mean of the per-modality '1s' outputs. CLS is a
        # depth-conditioning signal only — in the additive fusion inside depth it
        # learns to occupy an orthogonal complement of mod_ctx's residual subspace,
        # so probing CLS alone sees a structurally partial view of the signal.
        embeddings: dict[str, Tensor] = {"1s": temporal_context}  # (B, M, S, D)

        return {
            "loss": loss,
            "per_level_loss": per_level_loss.detach(),
            "per_level_accuracy": per_level_accuracy.detach(),
            "per_level_logit_max": per_level_logit_max.detach(),
            "per_level_logz": per_level_logz.detach(),
            "per_modality_loss": {k: v.detach() for k, v in per_modality_loss.items()},
            "per_modality_accuracy": {k: v.detach() for k, v in per_modality_accuracy.items()},
            "per_modality_per_sample_loss": {k: v.detach() for k, v in per_modality_per_sample_loss.items()},
            "per_modality_per_sample_correct": {k: v.detach() for k, v in per_modality_per_sample_correct.items()},
            "per_modality_logit_max": {k: v.detach() for k, v in per_modality_logit_max.items()},
            "per_modality_logz": {k: v.detach() for k, v in per_modality_logz.items()},
            "temporal_context": temporal_context,
            "embeddings": embeddings,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: Tensor,
        channel_ids: Tensor | None = None,
        modality_mask: Tensor | None = None,
        num_steps: int = 30,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively roll out ``num_steps`` timesteps for all modalities.

        Recomputes the temporal transformer from scratch each step (cheap for
        short clips — no KV cache). At eval the CRP grouping collapses to
        attend-all, so the per-step temporal pass matches a teacher-forced pass
        over the same prefix.

        Args:
            prompt_tokens: (B, S0, total_K) seed tokens, all modalities concatenated
                in ``modality_configs`` order.
            channel_ids: (B, M) channel index per modality, or None.
            modality_mask: (B, M) bool where True = present. None = all present.

        Returns:
            (B, S0 + num_steps, total_K) prompt + generated tokens.
        """
        tokens = prompt_tokens
        B = tokens.size(0)
        for _ in range(num_steps):
            temporal_input = self._build_temporal_input(
                tokens, channel_ids, modality_mask, drop_last=False
            )  # (B, M, S+1, D)

            backbone_group_ids = self.temporal_transformer.sample_group_ids(B, tokens.device)
            mod_attn_mask = self.temporal_transformer.build_modality_attn_mask(
                modality_mask, B, group_ids=backbone_group_ids
            )
            cross_attn_mask = None
            if self.use_cls:
                cross_attn_mask = self.temporal_transformer.build_cross_attn_mask(
                    modality_mask, B, group_ids=backbone_group_ids
                )

            temporal_context, cls_per_modality = self.temporal_transformer(
                temporal_input,
                modality_attn_mask=mod_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )  # (B, M, S+1, D)

            cond_last = temporal_context[:, :, -1]  # (B, M, D)
            new_cols = []
            for i in range(len(self.modality_configs)):
                cls_ctx_i = cls_per_modality[:, i, -1] if cls_per_modality is not None else None
                new_cols.append(
                    self.depth_transformer.sample(
                        i,
                        cond_last[:, i],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                        cls_context=cls_ctx_i,
                    )  # (B, K_i)
                )
            new_row = torch.cat(new_cols, dim=1)  # (B, total_K)
            tokens = torch.cat([tokens, new_row.unsqueeze(1)], dim=1)
        return tokens
