"""Hypnos embedding library: EDF -> preprocess -> tokenize -> embeddings.

Minimal, inference-only API around the released Hypnos model. Typical use::

    from hypnos.embedding import embed_edf

    emb = embed_edf("recording.edf")
    # emb: dict {modality_name: np.ndarray [n_seconds, embed_dim] float16}

Step-by-step control (reuse the loaded model across recordings)::

    from hypnos.embedding import load_model, preprocess_edf, tokenize, embed

    model, tokenizers, meta = load_model()
    signals = preprocess_edf("recording.edf", meta)
    tokens, mask, channel_ids = tokenize(tokenizers, meta, signals)
    emb = embed(model, tokens, mask, channel_ids, meta)   # {name: [T, D]}
"""

from __future__ import annotations

import numpy as np
import torch

from .generate import synthesize
from .infer import temporal_context
from .loader import DEFAULT_REPO, load_model
from .manifest import ModalitySpec, ModelMetadata
from .pipeline import preprocess_edf, tokenize

__all__ = [
    'load_model',
    'preprocess_edf',
    'tokenize',
    'embed',
    'embed_edf',
    'synthesize',
    'ModelMetadata',
    'ModalitySpec',
]


def embed(
    model,
    tokens: torch.Tensor,
    modality_mask: torch.Tensor,
    channel_ids: torch.Tensor,
    metadata: ModelMetadata,
    *,
    chunk_tokens: int | None = None,
    device: str | torch.device = 'cpu',
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, np.ndarray]:
    """Generate per-modality 1 Hz embeddings from assembled tokens.

    Returns a ``dict`` mapping each modality present in the recording to its
    ``[n_seconds, embed_dim]`` float16 embedding (``z^i_t``, one vector per second). To get a
    single summary vector, average over modalities; for coarser timescales, mean-pool over
    time (see the README).
    """
    ctx = temporal_context(
        model, tokens, modality_mask, channel_ids,
        chunk_tokens=chunk_tokens, device=device, autocast_dtype=autocast_dtype,
    )  # (T, M, D) float32
    present = modality_mask[0].tolist()
    return {
        spec.name: ctx[:, i].to(torch.float16).numpy()
        for i, spec in enumerate(metadata.modalities)
        if present[i]
    }


def embed_edf(
    edf_path: str,
    model_repo_or_path: str = DEFAULT_REPO,
    *,
    device: str | torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
    notch_freq: float = 50.0,
    causal: bool = True,
    chunk_tokens: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, np.ndarray]:
    """Convenience: load model -> preprocess EDF -> tokenize -> embed, in one call.

    ``model_repo_or_path`` defaults to the released model on the Hub. ``notch_freq`` is the
    powerline frequency to filter out — 50 Hz (default, most of the world) or 60 Hz (Americas).

    Returns a ``{modality_name: [n_seconds, embed_dim]}`` dict of per-modality 1 Hz
    embeddings (only modalities present in the recording). For repeated embedding, call
    :func:`load_model` once and reuse the returned model/tokenizers.
    """
    model, tokenizers, meta = load_model(model_repo_or_path, device=device, dtype=dtype)
    signals = preprocess_edf(edf_path, meta, notch_freq=notch_freq, causal=causal)
    tokens, modality_mask, channel_ids = tokenize(tokenizers, meta, signals, device=device)
    return embed(
        model, tokens, modality_mask, channel_ids, meta,
        chunk_tokens=chunk_tokens, device=device, autocast_dtype=autocast_dtype,
    )
