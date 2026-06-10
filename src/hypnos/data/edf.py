"""Read PSG channels from an EDF file, with EEG/EOG referencing and EMG bipolar derivation.

Scoped to the channels the released model consumes: C3, C4 (EEG), E1, E2 (EOG),
Chin (EMG), ECG, ABD, THX (respiratory). Signals are returned raw; downstream preprocessing
z-score normalises them, so absolute units don't matter and no unit conversion is applied.
"""

import logging
from dataclasses import dataclass, field
from math import gcd

import numpy as np
import pyedflib
from scipy.signal import resample_poly

from ..settings import ABD, ECG, EEG_C3, EEG_C4, EMG_CHIN, EOG_E1, EOG_E2, THX

_logger = logging.getLogger(__name__)

# Alternative EDF labels for each canonical channel, across datasets/conventions.
ALT_COLUMNS = {
    ECG: (
        'EKG', 'ECG1', 'ECG L', 'ECGL', 'ECG L-ECG R',
        'ECG EKG2-EKG', 'EKG2-EKG', 'ECG LA-RA', 'LA-RA',
    ),
    ABD: (
        'Abdo', 'ABDO RES', 'ABDO EFFORT', 'Abdominal', 'ABDOMINAL', 'Abdomen', 'abdomen',
        'Resp Abdominal', 'Resp Abdomen',
    ),
    THX: (
        'Thor', 'THOR RES', 'THOR EFFORT', 'Thoracic', 'Thorax', 'Chest', 'thorax', 'CHEST',
        'Resp Thoracic', 'Resp Chest',
    ),
    # EEG channels (handle referencing conventions)
    EEG_C3: ('C3-M2', 'C3-A2', 'C3_M2', 'C3_A2', 'EEG C3-M2', 'EEG C3-A2', 'EEG(sec) C3', 'EEG(sec)'),
    EEG_C4: ('C4-M1', 'C4-A1', 'C4_M1', 'C4_A1', 'EEG C4-M1', 'EEG C4-A1', 'EEG(sec) C4', 'EEG', 'EEG3'),
    # EOG channels (E1/E2 AASM standard)
    EOG_E1: ('E1-M2', 'E1-A2', 'EOG E1-M2', 'EOG(L)', 'LOC', 'EOG-L', 'EOGl', 'EOG LOC-M2', 'LOC-M2'),
    EOG_E2: (
        'E2-M1', 'E2-M2', 'E2-A1', 'EOG E2-M1', 'EOG(R)', 'ROC', 'EOG-R', 'EOGr',
        'EOG ROC-M1', 'ROC-M1', 'EEG ROC-M1',
    ),
    # EMG chin
    EMG_CHIN: (
        'Chin1-Chin2', 'CHIN1-CHIN2', 'CHIN', 'ChinA', 'Cchin', 'Chin EMG', 'EMG', 'EMG Chin', 'chin',
        'EMG Chin1-Chin2', 'EMG CHIN1-CHIN2', 'EMG Chin2-Chin1', 'Chin2-Chin1', 'EEG Chin1-Chin2',
    ),
}

# Contralateral mastoid referencing (AASM): canonical channel -> required reference electrode.
CONTRALATERAL_REF: dict[str, str] = {'C3': 'M2', 'C4': 'M1', 'E1': 'M2', 'E2': 'M1'}

# A1/A2 are legacy equivalents of M1/M2.
REFERENCE_ALTS: dict[str, list[str]] = {'M1': ['A1'], 'M2': ['A2']}
_ALL_REF_NAMES: set[str] = set(REFERENCE_ALTS) | {a for alts in REFERENCE_ALTS.values() for a in alts}

# Pre-computed bipolar labels (already derived in the EDF), checked before components.
BIPOLAR_LABELS: dict[str, tuple[str, ...]] = {'Chin': ('Chin1-Chin2', 'CHIN1-CHIN2')}

# Component electrode pairs for computing bipolar derivations: (positive, negative).
BIPOLAR_COMPONENTS: dict[str, list[tuple[str, str]]] = {
    'Chin': [
        ('ChinR', 'ChinL'), ('Chin1', 'Chin2'), ('cchin_r', 'cchin_l'),
        ('R Chin', 'L Chin'), ('Rchin', 'Lchin'), ('EMG3', 'EMG2'), ('EMG2', 'EMG1'),
    ],
}


def get_column_match(target_col: str, available_cols: list[str]) -> str | None:
    """Return the EDF label matching ``target_col`` (exact or via ``ALT_COLUMNS``), else None."""
    if target_col in available_cols:
        return target_col
    for alt_col in ALT_COLUMNS.get(target_col, ()):
        if alt_col in available_cols:
            return alt_col
    return None


@dataclass
class ResolvedChannel:
    """A channel resolved from an EDF file with referencing/derivation applied."""

    signal: np.ndarray
    sampling_rate: int
    unit: str
    physical_min: float
    physical_max: float
    method: str  # 'pre_referenced' | 're_referenced' | 'bipolar_derived' | 'bipolar_pre_computed' | 'direct'
    edf_labels: list[str] = field(default_factory=list)


def _find_reference_label(ref_name: str, available_labels: list[str]) -> str | None:
    """Find a reference channel (M1/M2 or A1/A2 equivalent) in available EDF labels."""
    if ref_name in available_labels:
        return ref_name
    for alt in REFERENCE_ALTS.get(ref_name, []):
        if alt in available_labels:
            return alt
    return None


def _load_reference_signals(
    f: pyedflib.EdfReader, label_to_idx: dict[str, int],
) -> dict[str, tuple[np.ndarray, int]]:
    """Load M1/M2 reference signals if available in the EDF."""
    available = list(label_to_idx.keys())
    refs: dict[str, tuple[np.ndarray, int]] = {}
    for ref_name in ('M1', 'M2'):
        label = _find_reference_label(ref_name, available)
        if label is not None:
            idx = label_to_idx[label]
            refs[ref_name] = (f.readSignal(idx), int(f.getSampleFrequency(idx)))
    return refs


def _resample_reference(ref_signal: np.ndarray, ref_fs: int, target_fs: int) -> np.ndarray:
    """Resample a reference signal to match the electrode sampling rate."""
    if ref_fs == target_fs:
        return ref_signal
    g = gcd(target_fs, ref_fs)
    return resample_poly(ref_signal, target_fs // g, ref_fs // g).astype(ref_signal.dtype)


def _is_pre_referenced(ch_name: str, actual_label: str) -> bool:
    """Whether the matched EDF label is already referenced (e.g. 'C3-M2', 'E2-M1')."""
    if ch_name not in CONTRALATERAL_REF:
        return False
    return any(ref in actual_label for ref in _ALL_REF_NAMES)


def _read_signal_metadata(f: pyedflib.EdfReader, idx: int) -> tuple[int, str, float, float]:
    """Read sampling rate, unit, physical min/max for a signal index."""
    return (
        int(f.getSampleFrequency(idx)),
        f.getPhysicalDimension(idx).strip(),
        f.getPhysicalMinimum(idx),
        f.getPhysicalMaximum(idx),
    )


def load_psg_channels(
    f: pyedflib.EdfReader,
    channels: list[str],
    drop_unreferenced: bool = False,
) -> dict[str, ResolvedChannel]:
    """Load PSG channels from an open EDF with proper referencing and derivations.

    Handles three channel types:
    1. **Contralateral-referenced** (EEG/EOG): detects pre-referenced labels (C3-M2),
       re-references bare electrodes using M1/M2, or drops if reference unavailable.
    2. **Bipolar-derived** (Chin EMG): computes derivation from component electrodes
       (e.g. ChinR - ChinL) when available, falls back to pre-computed or single labels.
    3. **Direct** (ECG, ABD, THX): loaded as-is.

    Args:
        f: Open pyedflib.EdfReader.
        channels: Canonical channel names (e.g. ['C3', 'E1', 'Chin', 'ECG']).
        drop_unreferenced: If True, skip channels that need contralateral referencing
            but have no reference electrode available (e.g. bare E1 without M2).

    Returns:
        Dict mapping canonical channel name to ResolvedChannel (channels not found are omitted).
    """
    signal_labels = f.getSignalLabels()
    label_to_idx = {label: i for i, label in enumerate(signal_labels)}
    available = list(label_to_idx.keys())

    ref_signals = _load_reference_signals(f, label_to_idx)
    result: dict[str, ResolvedChannel] = {}
    for ch_name in channels:
        resolved = _resolve_one_channel(f, ch_name, available, label_to_idx, ref_signals, drop_unreferenced)
        if resolved is not None:
            result[ch_name] = resolved
    return result


def _resolve_one_channel(
    f: pyedflib.EdfReader,
    ch_name: str,
    available: list[str],
    label_to_idx: dict[str, int],
    ref_signals: dict[str, tuple[np.ndarray, int]],
    drop_unreferenced: bool,
) -> ResolvedChannel | None:
    """Resolve a single canonical channel from an EDF.

    Tries bipolar derivation first (if applicable), then standard name resolution
    with contralateral re-referencing.
    """
    # --- Step 1: Bipolar channels (e.g. Chin EMG) ---
    if ch_name in BIPOLAR_LABELS or ch_name in BIPOLAR_COMPONENTS:
        # Try pre-computed bipolar labels (e.g. 'Chin1-Chin2' already in EDF)
        for label in BIPOLAR_LABELS.get(ch_name, ()):
            if label in available:
                idx = label_to_idx[label]
                signal = f.readSignal(idx)
                fs, unit, pmin, pmax = _read_signal_metadata(f, idx)
                return ResolvedChannel(
                    signal=signal, sampling_rate=fs, unit=unit,
                    physical_min=pmin, physical_max=pmax,
                    method='bipolar_pre_computed', edf_labels=[label],
                )

        # Try computing from component electrode pairs (e.g. ChinR - ChinL)
        for pos_label, neg_label in BIPOLAR_COMPONENTS.get(ch_name, []):
            if pos_label in available and neg_label in available:
                pos_idx = label_to_idx[pos_label]
                neg_idx = label_to_idx[neg_label]
                pos_signal = f.readSignal(pos_idx)
                neg_signal = f.readSignal(neg_idx)
                pos_fs, pos_unit, pos_pmin, pos_pmax = _read_signal_metadata(f, pos_idx)
                neg_fs = int(f.getSampleFrequency(neg_idx))

                if neg_fs != pos_fs:
                    neg_signal = _resample_reference(neg_signal, neg_fs, pos_fs)

                min_len = min(len(pos_signal), len(neg_signal))
                bipolar = (pos_signal[:min_len] - neg_signal[:min_len]).astype(np.float64)

                _logger.info(f'Computed bipolar derivation {pos_label}-{neg_label} for {ch_name}')
                return ResolvedChannel(
                    signal=bipolar, sampling_rate=pos_fs, unit=pos_unit,
                    physical_min=pos_pmin, physical_max=pos_pmax,
                    method='bipolar_derived', edf_labels=[pos_label, neg_label],
                )
        # Fall through to standard resolution (single electrode fallback)

    # --- Step 2: Standard name resolution ---
    actual_name = get_column_match(ch_name, available)
    if actual_name is None:
        _logger.info(f'Channel {ch_name} not found in EDF')
        return None

    idx = label_to_idx[actual_name]
    signal = f.readSignal(idx)
    fs, unit, pmin, pmax = _read_signal_metadata(f, idx)

    # --- Step 3: Check if already pre-referenced (e.g. 'C3-M2') ---
    if _is_pre_referenced(ch_name, actual_name):
        return ResolvedChannel(
            signal=signal, sampling_rate=fs, unit=unit,
            physical_min=pmin, physical_max=pmax,
            method='pre_referenced', edf_labels=[actual_name],
        )

    # --- Step 4: Bare electrode needing contralateral reference ---
    ref_name = CONTRALATERAL_REF.get(ch_name)
    if ref_name:
        if ref_name in ref_signals:
            ref_signal, ref_fs = ref_signals[ref_name]
            ref_resampled = _resample_reference(ref_signal, ref_fs, fs)
            min_len = min(len(signal), len(ref_resampled))
            signal = (signal[:min_len] - ref_resampled[:min_len]).astype(np.float64)
            return ResolvedChannel(
                signal=signal, sampling_rate=fs, unit=unit,
                physical_min=pmin, physical_max=pmax,
                method='re_referenced', edf_labels=[actual_name],
            )
        # No reference electrodes in EDF at all → hardware applied referencing
        if not ref_signals:
            _logger.info(f'{ch_name}: no reference electrodes in EDF, assuming acquisition-referenced')
            return ResolvedChannel(
                signal=signal, sampling_rate=fs, unit=unit,
                physical_min=pmin, physical_max=pmax,
                method='pre_referenced', edf_labels=[actual_name],
            )
        # Some refs exist but not the one we need → genuinely unreferenced
        if drop_unreferenced:
            _logger.warning(f'{ch_name}: needs {ref_name} but only {set(ref_signals)} available, dropping')
            return None
        _logger.warning(f'{ch_name}: bare electrode without {ref_name} reference, using unreferenced')

    # --- Step 5: No referencing needed (ECG, ABD, etc.) or unreferenced fallback ---
    return ResolvedChannel(
        signal=signal, sampling_rate=fs, unit=unit,
        physical_min=pmin, physical_max=pmax,
        method='direct', edf_labels=[actual_name],
    )
