"""Offline end-to-end verification with a tiny in-process model bundle.

Part A: assert the ported `_temporal_only_forward` matches the model's real `forward`
        ['embeddings']['1s'] exactly (the core correctness gate).
Part B: build a tiny single-file bundle (config + model + tokenizer weights) + a synthetic
        EDF, then run load_model -> preprocess_edf -> tokenize -> embed.
"""

import tempfile
from pathlib import Path

import numpy as np
import pyedflib
import torch

from hypnos.models.rq_transformer import ModalityConfig, MultiModalRQTransformer
from hypnos.models.tokenizer import SignalTokenizer

torch.manual_seed(0)

EMBED_DIM = 32
CB = 16  # codebook size
K_EEG, K_ECG, K_RESP = 2, 2, 2

MODALITIES = [
    # name, signal_type, channels, tokenizer_stem, num_quantizers, sample_rate, ratios, preprocess
    ("eeg_c3", "eeg", ["C3"], "eeg", K_EEG, 128, [4, 4, 4, 2], "eeg"),
    ("eeg_c4", "eeg", ["C4"], "eeg", K_EEG, 128, [4, 4, 4, 2], "eeg"),
    ("ecg", "ecg", ["ECG"], "ecg", K_ECG, 128, [4, 4, 4, 2], "ecg"),
    ("resp_abd", "resp", ["ABD"], "resp", K_RESP, 32, [4, 4, 2], "respiratory"),
]

MODEL_KWARGS = dict(
    embed_dim=EMBED_DIM,
    temporal_depth=2,
    temporal_heads=2,
    depth_depth=1,
    depth_heads=1,
    max_seq_len=4096,
    dropout=0.0,
    sliding_window=8,
    channel_embeddings=True,
    use_cls=False,
    qk_norm=True,
    causal=True,
)


def build_model():
    cfgs = [
        ModalityConfig(name=n, num_quantizers=k, codebook_size=CB, signal_type=st)
        for (n, st, _ch, _tk, k, _sr, _r, _pp) in MODALITIES
    ]
    m = MultiModalRQTransformer(modality_configs=cfgs, **MODEL_KWARGS).eval()
    return m


def tokenizer_kwargs(num_q, sample_rate, ratios):
    return dict(
        in_channels=1,
        sample_rate=sample_rate,
        token_duration_sec=1.0,
        embed_dim=16,
        n_filters=4,
        ratios=ratios,
        n_residual_layers=1,
        mode="discrete",
        codebook_size=CB,
        codebook_dim=8,
        num_quantizers=num_q,
        attention_depth=0,
        causal=True,
        norm="weight",
        activation="elu",
    )


def build_bundle(path):
    """Build a tiny single-file .safetensors bundle (config metadata + namespaced weights)."""
    import json

    from safetensors.torch import save_file

    tensors, tokenizers = {}, {}
    for k_, v in build_model().state_dict().items():
        tensors[f"model/{k_}"] = v.detach().contiguous().clone()
    for _n, _st, _ch, stem, k, sr, ratios, _pp in MODALITIES:
        if stem in tokenizers:
            continue
        kw = tokenizer_kwargs(k, sr, ratios)
        tokenizers[stem] = {
            "signal_type": _st, "num_quantizers": k, "codebook_size": CB,
            "token_duration_sec": 1.0, "sample_rate": sr, "tokenizer_kwargs": kw,
        }
        for k_, v in SignalTokenizer(**kw).state_dict().items():
            tensors[f"tok/{stem}/{k_}"] = v.detach().contiguous().clone()
    modalities = [
        {"name": n, "signal_type": st, "channels": ch, "tokenizer": stem, "num_quantizers": k,
         "codebook_size": CB, "token_duration_sec": 1.0, "sample_rate": sr, "preprocess_modality": pp}
        for (n, st, ch, stem, k, sr, _r, pp) in MODALITIES
    ]
    config = {
        "model_target": "hypnos.models.rq_transformer.MultiModalRQTransformer",
        "model_kwargs": MODEL_KWARGS, "modalities": modalities, "tokenizers": tokenizers,
    }
    save_file(tensors, str(path), metadata={"format_version": "1", "config": json.dumps(config)})


def test_temporal_only_matches_forward():
    from hypnos.embedding.infer import _temporal_only_forward

    model = build_model()
    M = len(MODALITIES)
    total_K = sum(mm[4] for mm in MODALITIES)
    B, S = 2, 90
    tokens = torch.randint(0, CB, (B, S, total_K))
    from hypnos.settings import NUM_KNOWN_CHANNELS

    channel_ids = torch.randint(0, NUM_KNOWN_CHANNELS, (B, M))
    modality_mask = torch.tensor([[True, True, True, False], [True, False, True, True]])

    with torch.inference_mode():
        ref = model.forward(tokens, channel_ids=channel_ids, modality_mask=modality_mask)["embeddings"]["1s"]
        got = _temporal_only_forward(model, tokens, channel_ids=channel_ids, modality_mask=modality_mask)
    assert ref.shape == got.shape == (B, M, S, EMBED_DIM), (ref.shape, got.shape)
    max_diff = (ref - got).abs().max().item()
    assert torch.allclose(ref, got, atol=1e-5), f"temporal-only mismatch, max_diff={max_diff}"

    print(f"[A] _temporal_only_forward matches forward (max_diff={max_diff:.2e}) OK")


def make_synthetic_edf(path, duration_sec=120):
    labels = {"C3-M2": 128, "C4-M1": 128, "ECG": 128, "ABD": 32}
    w = pyedflib.EdfWriter(str(path), len(labels))
    headers, data = [], []
    for i, (lab, fs) in enumerate(labels.items()):
        headers.append(
            {
                "label": lab,
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": -500.0,
                "physical_max": 500.0,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        )
        rng = np.random.default_rng(i)
        data.append((rng.standard_normal(fs * duration_sec) * 50).astype(np.float64))
    w.setSignalHeaders(headers)
    w.writeSamples(data)
    w.close()


def test_end_to_end_pipeline():
    from hypnos.embedding import embed, load_model, preprocess_edf, tokenize

    with tempfile.TemporaryDirectory() as d:
        bundle_path = Path(d) / "bundle.safetensors"
        build_bundle(bundle_path)

        edf = Path(d) / "rec.edf"
        make_synthetic_edf(edf)

        lm, toks, meta = load_model(bundle_path, device="cpu")
        assert list(meta.unique_tokenizers) == ["eeg", "ecg", "resp"]
        assert lm._modality_offsets == {"eeg_c3": (0, 2), "eeg_c4": (2, 4), "ecg": (4, 6), "resp_abd": (6, 8)}, (
            lm._modality_offsets
        )
        print("[B] load_model OK; offsets", lm._modality_offsets)

        signals = preprocess_edf(str(edf), meta, notch_freq=60.0, causal=True)
        assert set(signals) == {"eeg_c3", "eeg_c4", "ecg", "resp_abd"}, set(signals)
        # 128 Hz vs 32 Hz length ratio
        assert abs(len(signals["eeg_c3"]) / len(signals["resp_abd"]) - 4.0) < 0.01
        print("[B] preprocess_edf OK; lengths", {k: len(v) for k, v in signals.items()})

        tokens, mask, ch_ids = tokenize(toks, meta, signals, device="cpu")
        assert tokens.shape[0] == 1 and tokens.shape[2] == 8, tokens.shape
        assert mask.shape == (1, 4) and bool(mask.all())
        assert ch_ids.tolist() == [[0, 1, 5, 6]], ch_ids.tolist()  # C3=0,C4=1,ECG=5,ABD=6
        print("[B] tokenize OK; tokens", tuple(tokens.shape), "channel_ids", ch_ids.tolist())

        n_tok = tokens.shape[1]
        # Per-modality 1 Hz dict (one frame per token).
        per = embed(lm, tokens, mask, ch_ids, meta, device="cpu")
        assert set(per) == {"eeg_c3", "eeg_c4", "ecg", "resp_abd"}, set(per)
        assert all(v.shape == (n_tok, EMBED_DIM) and v.dtype == np.float16 for v in per.values())
        assert all(np.isfinite(v).all() for v in per.values())
        print(f"[B] embed 1Hz per-modality dict OK; keys {sorted(per)}")

        # README recipe: pool over modalities + 30-s epochs.
        fused = np.mean(list(per.values()), axis=0)  # [T, D]
        n_ep = fused.shape[0] // 30
        epochs = fused[: n_ep * 30].reshape(n_ep, 30, -1).mean(1)  # [T//30, D]
        assert epochs.shape == (n_tok // 30, EMBED_DIM), epochs.shape
        print(f"[B] modality+epoch pooling recipe OK; {epochs.shape}")

        # absent-modality path: drop ABD; dict must exclude it.
        partial = {k: v for k, v in signals.items() if k != "resp_abd"}
        t2, m2, c2 = tokenize(toks, meta, partial, device="cpu")
        assert m2.tolist() == [[True, True, True, False]]
        per2 = embed(lm, t2, m2, c2, meta, device="cpu")
        assert set(per2) == {"eeg_c3", "eeg_c4", "ecg"}, set(per2)
        print("[B] absent-modality path OK; per-modality keys", sorted(per2))


if __name__ == "__main__":
    test_temporal_only_matches_forward()
    test_end_to_end_pipeline()
    print("\nALL CHECKS PASSED")
