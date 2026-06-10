"""Temporal-context inference for the RQ-Transformer.

We only need the temporal-context output (``embeddings['1s']``); the full ``forward``
additionally runs the depth transformer + per-level cross-entropy (~3x the FLOPs) which is
pure waste at inference, so these helpers run the model up to the temporal transformer only.

The tokenizers emit one token per second, so the temporal context is a 1 Hz sequence, and
embeddings are returned per modality (``z^i_t``) at that native cadence.
"""

from __future__ import annotations

import torch


def require_runnable_device(device: str | torch.device) -> torch.device:
    """Normalise *device* to a ``torch.device``.

    Windowed attention uses ``flex_attention`` on CUDA/CPU; ``flex_attention`` has no MPS
    backend, so on Apple Silicon the model falls back to a dense-mask SDPA path (see
    :func:`hypnos.models.attention.get_block_mask`). CUDA, CPU and MPS are all runnable.
    """
    return torch.device(device)


def _temporal_only_forward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    channel_ids: torch.Tensor | None,
    modality_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run the model's token aggregation + temporal transformer only.

    Returns the per-modality temporal context ``(B, M, S, D)``. Mirrors
    ``MultiModalRQTransformer.forward`` up to (and including) the temporal transformer call,
    skipping the depth transformer + output heads + loss.
    """
    B, S, _ = tokens.shape

    # Per-modality token-embedding aggregation + per-modality BOS prepend.
    agg_embeds = model._aggregate_embeddings(tokens[:, :-1])  # (B, M, S-1, D)
    bos_list = [
        model.bos_embeds[model._weight_group[mc.name]].unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, 1, 1, -1)
        for mc in model.modality_configs
    ]
    bos = torch.cat(bos_list, dim=1)  # (B, M, 1, D)
    temporal_input = torch.cat([bos, agg_embeds], dim=2)  # (B, M, S, D)

    if modality_mask is not None:
        presence = modality_mask.unsqueeze(-1).unsqueeze(-1).float()
        temporal_input = temporal_input * presence

    if model.channel_embed is not None and channel_ids is not None:
        safe_ids = channel_ids.clamp(min=0)
        ch_embeds = model.channel_embed(safe_ids).unsqueeze(2)  # (B, M, 1, D)
        if modality_mask is not None:
            ch_embeds = ch_embeds * presence
        temporal_input = temporal_input + ch_embeds

    # At eval the modality grouping collapses to "all modalities together" (sample_group_ids
    # returns identical ids per sample), so this mask is effectively a no-op — but we build it
    # the same way the model expects.
    backbone_group_ids = model.temporal_transformer.sample_group_ids(B, tokens.device)
    mod_attn_mask = model.temporal_transformer.build_modality_attn_mask(
        modality_mask, B, group_ids=backbone_group_ids,
    )
    # cross_attn_mask=None: this model has use_cls=False, so there is no CLS cross-attention.
    temporal_context, _cls = model.temporal_transformer(
        temporal_input, modality_attn_mask=mod_attn_mask, cross_attn_mask=None,
    )
    return temporal_context  # (B, M, S, D)


@torch.inference_mode()
def temporal_context(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    modality_mask: torch.Tensor,
    channel_ids: torch.Tensor,
    *,
    chunk_tokens: int | None = None,
    device: str | torch.device = 'cpu',
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the temporal model over one record (chunked) -> per-modality 1 Hz context.

    Returns ``(n_tokens, M, D)`` float32 on CPU — the per-modality temporal context at the
    tokenizers' native 1 Hz cadence (one frame per token). Long records are processed in
    non-overlapping chunks and concatenated (window-edge effects are minor — the model is
    trained with a 64-token sliding window and 256-token global window, both << chunk).

    ``chunk_tokens`` defaults to 32768 on CUDA (one fused graph) and 2048 otherwise. On CPU
    flex_attention runs eagerly, and on MPS the windowed attention falls back to a dense-mask
    SDPA path; both materialise a full (chunk, chunk) score matrix per head, so peak memory
    grows ~quadratically with the chunk (≈8 GB at 2048, ≈19 GB at 4096). Lower this on a
    memory-constrained machine (Apple Silicon shares this budget with the rest of the system),
    raise it for speed. Total record length does not affect peak memory (chunks are processed
    sequentially).
    """
    device = require_runnable_device(device)
    _, n_tokens, total_k = tokens.shape
    if n_tokens == 0:
        return torch.zeros((0, len(model.modality_configs), model.embed_dim), dtype=torch.float32)

    if chunk_tokens is None:
        chunk_tokens = 32768 if device.type == 'cuda' else 2048
    chunk = max(1, chunk_tokens)
    tokens_t = tokens[0].to(torch.long)  # (n_tokens, K)
    mod_mask_t = modality_mask.to(device, torch.bool)  # (1, M)
    ch_ids_t = channel_ids.to(device, torch.long)  # (1, M)

    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=autocast_dtype)
        if autocast_dtype is not None and device.type == 'cuda'
        else torch.autocast(device_type='cpu', enabled=False)
    )

    # On CUDA we pad each chunk to a fixed length so torch.compile reuses one graph across
    # records. On CPU/MPS attention runs eagerly over a materialised (S, S) score matrix, so
    # padding short records up to `chunk` would OOM — process actual lengths there.
    pad_to_chunk = device.type == 'cuda'
    pad_template = torch.zeros((chunk, total_k), dtype=tokens_t.dtype) if pad_to_chunk else None

    frames: list[torch.Tensor] = []
    for start in range(0, n_tokens, chunk):
        stop = min(start + chunk, n_tokens)
        actual_len = stop - start
        if pad_to_chunk and actual_len < chunk:
            win = pad_template.clone()
            win[:actual_len] = tokens_t[start:stop]
        else:
            win = tokens_t[start:stop]
        win = win.unsqueeze(0).to(device)  # (1, S, K)
        with autocast_ctx:
            ctx = _temporal_only_forward(model, win, channel_ids=ch_ids_t, modality_mask=mod_mask_t)  # (1, M, S, D)
        ctx = ctx[:, :, :actual_len]  # drop padded tail
        frames.append(ctx.squeeze(0).permute(1, 0, 2).to('cpu', torch.float32))  # (S, M, D)

    return torch.cat(frames, dim=0)  # (n_tokens, M, D)
