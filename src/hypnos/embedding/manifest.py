"""Config schema for the single-file model bundle.

The model + all tokenizers ship as one ``.safetensors`` file. Weights live under namespaced
keys (``model/<param>``, ``tok/<stem>/<param>``); the ``config`` block below is stored as a
JSON string in the file metadata::

    config = {
      "model_target": "...MultiModalRQTransformer",
      "model_kwargs": {...},                 # ctor kwargs (no modality_configs)
      "modalities": [ {name, signal_type, channels, tokenizer, num_quantizers,
                       codebook_size, token_duration_sec, sample_rate, preprocess_modality}, ... ],
      "tokenizers": { <stem>: {signal_type, num_quantizers, codebook_size,
                               token_duration_sec, sample_rate, tokenizer_kwargs} },
    }

This module defines the dataclasses and parses/validates the ``config`` block. It does no
file IO and loads no weights — see ``loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

FORMAT_VERSION = 1


@dataclass
class ModalitySpec:
    """One modality of the model."""

    name: str  # unique modality id, e.g. 'eeg_c3'
    signal_type: str  # broad class for weight-grouping, e.g. 'eeg'
    channels: list[str]  # canonical channel names, e.g. ['C3']
    tokenizer: str  # key into the tokenizers map, e.g. 'eeg-q8-causal'
    num_quantizers: int
    codebook_size: int
    token_duration_sec: float
    sample_rate: int  # Hz the signal must be resampled to before tokenization
    preprocess_modality: str  # key into MODALITY_CONFIGS (note: EOG -> 'eeg')


@dataclass
class TokenizerSpec:
    """Construction kwargs for one SignalTokenizer."""

    signal_type: str
    num_quantizers: int
    codebook_size: int
    token_duration_sec: float
    sample_rate: int
    tokenizer_kwargs: dict  # full SignalTokenizer ctor kwargs


@dataclass
class ModelMetadata:
    """Resolved view of a bundle's ``config`` block."""

    model_target: str
    model_kwargs: dict  # ctor kwargs for MultiModalRQTransformer (EXCLUDING modality_configs)
    modalities: list[ModalitySpec]
    tokenizers: dict[str, TokenizerSpec]

    @property
    def unique_tokenizers(self) -> list[str]:
        """Distinct tokenizer stems, in first-seen modality order."""
        seen: list[str] = []
        for m in self.modalities:
            if m.tokenizer not in seen:
                seen.append(m.tokenizer)
        return seen


def parse_config(config: dict) -> ModelMetadata:
    """Validate a bundle ``config`` dict and return :class:`ModelMetadata`."""
    modalities = [
        ModalitySpec(
            name=m['name'],
            signal_type=m['signal_type'],
            channels=list(m['channels']),
            tokenizer=m['tokenizer'],
            num_quantizers=int(m['num_quantizers']),
            codebook_size=int(m['codebook_size']),
            token_duration_sec=float(m['token_duration_sec']),
            sample_rate=int(m['sample_rate']),
            preprocess_modality=m['preprocess_modality'],
        )
        for m in config['modalities']
    ]
    if not modalities:
        raise ValueError('bundle config has no modalities.')

    # All modalities must share a single token cadence — the temporal model concatenates them
    # along time and averages across modalities, valid only at one token-per-interval cadence.
    durations = {round(m.token_duration_sec, 6) for m in modalities}
    if len(durations) != 1:
        raise ValueError(f'All modalities must share token_duration_sec; got {sorted(durations)}.')

    tokenizers = {
        stem: TokenizerSpec(
            signal_type=t['signal_type'],
            num_quantizers=int(t['num_quantizers']),
            codebook_size=int(t['codebook_size']),
            token_duration_sec=float(t['token_duration_sec']),
            sample_rate=int(t['sample_rate']),
            tokenizer_kwargs=dict(t['tokenizer_kwargs']),
        )
        for stem, t in config['tokenizers'].items()
    }

    model_kwargs = dict(config['model_kwargs'])
    if 'modality_configs' in model_kwargs:
        raise ValueError('model_kwargs must NOT contain modality_configs (built from `modalities`).')

    return ModelMetadata(
        model_target=config['model_target'],
        model_kwargs=model_kwargs,
        modalities=modalities,
        tokenizers=tokenizers,
    )
