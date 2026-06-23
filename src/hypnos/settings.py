"""Channel constants for the pretrained model.

The model uses 8 channels (the 8 modalities below). Channel ids index into the model's
learned channel-embedding table; the published checkpoint's table contains exactly these 8
trained rows. ``CHANNEL_REGISTRY`` and ``NUM_KNOWN_CHANNELS`` must stay in sync with it.

Modality names (``eeg_c3``, ``ecg``, …) live in :mod:`hypnos.modalities` — distinct from the
channel names below (``C3``, ``ECG``, …), which is why they are kept in a separate module.
"""

# Canonical channel names used by the 8 modalities.
EEG_C3 = "C3"
EEG_C4 = "C4"
EOG_E1 = "E1"
EOG_E2 = "E2"
EMG_CHIN = "Chin"
ECG = "ECG"
ABD = "ABD"
THX = "THX"

# Canonical channel name -> row in the (8-row) channel-embedding table.
CHANNEL_REGISTRY: dict[str, int] = {
    EEG_C3: 0,
    EEG_C4: 1,
    EOG_E1: 2,
    EOG_E2: 3,
    EMG_CHIN: 4,
    ECG: 5,
    ABD: 6,
    THX: 7,
}

NUM_KNOWN_CHANNELS = len(CHANNEL_REGISTRY)
