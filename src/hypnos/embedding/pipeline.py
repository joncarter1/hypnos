"""EDF -> preprocessed per-modality signals -> assembled token tensor.

Two steps, mirroring how the released checkpoints were trained:

1. ``preprocess_edf`` — load each modality's channel from the EDF (with proper EEG/EOG
   contralateral referencing), resample to the modality's sample rate, and apply the
   *causal* preprocessing (forward-only IIR + rolling-EMA z-score) the causal tokenizers
   were trained on.
2. ``tokenize`` — run each modality's signal through its SignalTokenizer and assemble the
   per-modality token streams into the single ``(1, n_tokens, total_K)`` tensor the
   MultiModalRQTransformer expects, plus the ``modality_mask`` and ``channel_ids``.
"""

from __future__ import annotations

import logging

from collections.abc import Mapping, Sequence

import numpy as np
import pyedflib
import torch

from hypnos.data.edf import load_psg_channels
from hypnos.data.preprocessing import causal_preprocess_signal, preprocess_signal, resample_signal
from hypnos.settings import CHANNEL_REGISTRY

from .manifest import ModelMetadata

logger = logging.getLogger(__name__)


def preprocess_edf(
    edf_path: str,
    metadata: ModelMetadata,
    *,
    notch_freq: float = 50.0,
    causal: bool = True,
    tau_seconds: float = 60.0,
    channel_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Load and preprocess each modality's signal from an EDF.

    Returns ``{modality_name: signal (T_m,)}`` for modalities whose channel(s) are present
    in the recording; absent modalities are omitted (handled downstream by ``modality_mask``).
    Each modality's length differs because sample rates differ (e.g. 32 Hz respiratory vs
    128 Hz EEG).

    Args:
        edf_path: Path to the EDF/EDF+ recording.
        metadata: Loaded model metadata (drives channels, sample rate, preprocess modality).
        notch_freq: Powerline frequency to notch out — 50 Hz (EU) or 60 Hz (US). Must match
            the recording's region. Ignored for modalities whose config has no notch.
        causal: Use the causal preprocessing path (matches the released causal tokenizers).
            Set False only to experiment with the zero-phase ``preprocess_signal``.
        tau_seconds: Rolling-normaliser timescale for the causal path.
        channel_aliases: Optional ``{canonical_name: [extra EDF labels]}`` mapping for
            recordings whose channel labels aren't covered by the built-in ``ALT_COLUMNS``.
            Keys are canonical channel names (e.g. ``"ECG"``, ``"C3"``); caller aliases take
            precedence over the built-ins.
    """
    # One channel per modality for the released model (in_channels=1 tokenizers); we take
    # the first channel of each modality spec.
    all_channels = sorted({ch for m in metadata.modalities for ch in m.channels})

    with pyedflib.EdfReader(edf_path) as f:
        resolved = load_psg_channels(
            f, all_channels, drop_unreferenced=True, channel_aliases=channel_aliases
        )

    signals: dict[str, np.ndarray] = {}
    for m in metadata.modalities:
        ch = m.channels[0]
        rc = resolved.get(ch)
        if rc is None:
            logger.info("Channel %r for modality %r not present in %s; skipping.", ch, m.name, edf_path)
            continue
        sig = resample_signal(np.asarray(rc.signal), int(rc.sampling_rate), m.sample_rate)
        if causal:
            processed, _, _ = causal_preprocess_signal(
                sig,
                fs=m.sample_rate,
                modality=m.preprocess_modality,
                notch_freq=notch_freq,
                tau_seconds=tau_seconds,
            )
        else:
            processed, _, _ = preprocess_signal(
                sig,
                fs=m.sample_rate,
                modality=m.preprocess_modality,
                notch_freq=notch_freq,
            )
        signals[m.name] = np.asarray(processed, dtype=np.float32)
    return signals


@torch.inference_mode()
def tokenize(
    tokenizers: dict,
    metadata: ModelMetadata,
    signals: dict[str, np.ndarray],
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize per-modality signals and assemble the token tensor.

    Returns ``(tokens, modality_mask, channel_ids)``:
      - ``tokens``: ``(1, n_tokens, total_K)`` int64, modalities concatenated along the K
        axis in manifest order (== the model's ``_modality_offsets`` layout). Absent
        modalities contribute an all-zero block.
      - ``modality_mask``: ``(1, M)`` bool — True where the modality is present.
      - ``channel_ids``: ``(1, M)`` int64 — ``CHANNEL_REGISTRY`` id of each modality's
        channel (clamped to 0 for absent modalities; the mask zeroes them anyway).

    All present modalities are truncated to the common minimum token count (matches the
    train-time join).
    """
    # Tokenize present modalities to (n_m, K_m) int64.
    per_modality_tokens: dict[str, torch.Tensor] = {}
    for m in metadata.modalities:
        sig = signals.get(m.name)
        if sig is None:
            continue
        x = torch.from_numpy(np.asarray(sig, dtype=np.float32)).view(1, 1, -1).to(device)
        tok = tokenizers[m.name].tokenize(x)  # (1, n_m, K_m)
        per_modality_tokens[m.name] = tok[0].to("cpu", torch.long)

    if not per_modality_tokens:
        raise ValueError("No modalities present in the recording; cannot tokenize.")

    n_tokens = min(t.shape[0] for t in per_modality_tokens.values())
    if n_tokens == 0:
        raise ValueError("Recording too short: produced 0 tokens for at least one present modality.")

    blocks: list[torch.Tensor] = []
    mask: list[bool] = []
    channel_ids: list[int] = []
    for m in metadata.modalities:
        tok = per_modality_tokens.get(m.name)
        if tok is None:
            blocks.append(torch.zeros((n_tokens, m.num_quantizers), dtype=torch.long))
            mask.append(False)
            channel_ids.append(0)
        else:
            blocks.append(tok[:n_tokens])
            mask.append(True)
            channel_ids.append(CHANNEL_REGISTRY[m.channels[0]])

    tokens = torch.cat(blocks, dim=1).unsqueeze(0)  # (1, n_tokens, total_K)
    modality_mask = torch.tensor([mask], dtype=torch.bool)  # (1, M)
    channel_ids_t = torch.tensor([channel_ids], dtype=torch.long)  # (1, M)
    return tokens, modality_mask, channel_ids_t
