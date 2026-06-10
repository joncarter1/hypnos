"""Autoregressive signal generation from the Hypnos RQ-Transformer.

Wraps ``model.generate`` (which emits tokens) together with per-modality decoding, so
callers get waveforms directly without assembling channel ids, modality masks, and token
spans by hand. This is the generative counterpart to :func:`hypnos.embedding.embed`.
"""

from __future__ import annotations

import numpy as np
import torch

from hypnos.settings import CHANNEL_REGISTRY

from .manifest import ModelMetadata


@torch.inference_mode()
def synthesize(
    model,
    tokenizers: dict,
    metadata: ModelMetadata,
    *,
    modalities: list[str] | None = None,
    num_steps: int = 30,
    prompt_tokens: torch.Tensor | None = None,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Autoregressively generate signals for the chosen modalities.

    Cold-starts from the model's learned per-modality beginning-of-sequence embedding
    (or continues ``prompt_tokens`` when given), rolls out ``num_steps`` tokens at the 1 Hz
    token cadence, and decodes each requested modality back to a waveform at its native
    sample rate. Channel conditioning is set automatically from each modality's registered
    channel — passing the wrong channel id produces unphysiological traces, so this is not
    left to the caller.

    Args:
        model, tokenizers, metadata: as returned by :func:`load_model`.
        modalities: modality names to generate; the rest are masked out. ``None`` = all.
        num_steps: number of 1 Hz steps to roll out.
        prompt_tokens: optional ``(1, S0, total_K)`` context to condition on; ``None`` =
            cold start. Decoded output covers the prompt plus the generated steps.
        temperature: sampling temperature (``0`` → greedy). Lower values give smoother,
            more regular traces; the same value applies to every modality.
        top_k, top_p: nucleus / top-k sampling controls.
        generator: optional ``torch.Generator`` for reproducible sampling (must be on the
            model's device).

    Returns:
        ``{modality_name: np.ndarray}`` — one decoded waveform per requested modality, at
        the tokenizer's native sample rate (1-D for the single-channel released model).
    """
    mods = metadata.modalities
    names = [m.name for m in mods]
    targets = list(names if modalities is None else modalities)
    unknown = [t for t in targets if t not in names]
    if unknown:
        raise ValueError(f'unknown modalities {unknown}; choose from {names}')

    device = next(model.parameters()).device
    total_K = sum(m.num_quantizers for m in mods)

    # Per-modality column spans in the concatenated token tensor.
    spans, offset = {}, 0
    for m in mods:
        spans[m.name] = (offset, offset + m.num_quantizers)
        offset += m.num_quantizers

    modality_mask = torch.tensor(
        [[m.name in targets for m in mods]], dtype=torch.bool, device=device
    )
    channel_ids = torch.tensor(
        [[CHANNEL_REGISTRY[m.channels[0]] for m in mods]], dtype=torch.long, device=device
    )

    if prompt_tokens is None:
        prompt_tokens = torch.zeros(1, 0, total_K, dtype=torch.long, device=device)
    else:
        prompt_tokens = prompt_tokens.to(device=device, dtype=torch.long)

    rollout = model.generate(
        prompt_tokens, channel_ids, modality_mask,
        num_steps=num_steps, temperature=temperature, top_k=top_k, top_p=top_p,
        generator=generator,
    )

    signals: dict[str, np.ndarray] = {}
    for name in targets:
        start, end = spans[name]
        wav = tokenizers[name].decode_tokens(rollout[:, :, start:end])  # (1, C, samples)
        arr = wav[0].float().cpu().numpy()  # (C, samples)
        signals[name] = arr[0] if arr.shape[0] == 1 else arr
    return signals
