"""Load the single-file model bundle (local path or HuggingFace repo).

The bundle is a ``.safetensors`` file: all weights live under namespaced keys
(``model/<param>`` for the RQ-Transformer, ``tok/<stem>/<param>`` for each tokenizer) and the
config (model + tokenizer construction kwargs, modality layout) is a JSON string in the
file's metadata. safetensors is a pure-tensor format — no arbitrary-code unpickling.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from hypnos.models.rq_transformer import ModalityConfig, MultiModalRQTransformer
from hypnos.models.tokenizer import SignalTokenizer

from .infer import require_runnable_device
from .manifest import ModelMetadata, parse_config

logger = logging.getLogger(__name__)

# Default HuggingFace repo and bundle filename.
DEFAULT_REPO = "joncarter/hypnos"
BUNDLE_FILENAME = "hypnos.safetensors"


def resolve_bundle(path_or_repo: str | Path, filename: str = BUNDLE_FILENAME) -> Path:
    """Return a local path to the bundle file.

    Accepts a local ``.safetensors`` file, a local directory containing ``filename``, an
    ``hf://<repo_id>`` URI, or a bare HuggingFace repo id (``owner/name``). Remote repos are
    fetched with ``hf_hub_download``: the bundle file, plus a best-effort ``config.json`` so
    the Hub registers the download (its stats counter keys on ``config.json``, the default
    query file, not on the ``.safetensors`` bundle).
    """
    s = str(path_or_repo)
    if os.path.isfile(s):
        return Path(s)
    if os.path.isdir(s):
        return Path(s) / filename

    repo_id = s[len("hf://") :] if s.startswith("hf://") else s
    if "/" not in repo_id:
        raise FileNotFoundError(f"{s!r} is not a local bundle/dir and not a valid HuggingFace repo id (owner/name).")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - dependency declared in pyproject
        raise ImportError("huggingface-hub is required to load a bundle from the Hub.") from e

    # Touch config.json so the Hub counts this as a download — its stats key on config.json,
    # not on the .safetensors bundle. Best-effort: a GET (or cached HEAD revalidation) is
    # enough to register, and loading must never fail if the file is absent or the request
    # errors (a HEAD/GET to a missing entry raises, which we swallow).
    try:
        hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception:
        logger.debug("config.json not fetched from %s; download count may not register", repo_id)

    logger.info("Downloading %s from HuggingFace repo %s...", filename, repo_id)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _read_bundle(path: Path) -> tuple[dict, dict, dict[str, dict]]:
    """Read a ``.safetensors`` bundle -> ``(config, model_state_dict, {stem: tokenizer_sd})``."""
    from safetensors import safe_open

    model_sd: dict = {}
    tokenizer_sds: dict[str, dict] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        if "config" not in md:
            raise RuntimeError(f"{path}: no 'config' in safetensors metadata; not a Hypnos bundle.")
        config = json.loads(md["config"])
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.startswith("model/"):
                model_sd[key[len("model/") :]] = tensor
            elif key.startswith("tok/"):
                _, stem, param = key.split("/", 2)
                tokenizer_sds.setdefault(stem, {})[param] = tensor
            else:
                raise RuntimeError(f"unexpected tensor key {key!r} in bundle")
    return config, model_sd, tokenizer_sds


def load_model(
    path_or_repo: str | Path = DEFAULT_REPO,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[MultiModalRQTransformer, dict[str, SignalTokenizer], ModelMetadata]:
    """Build the model + per-modality tokenizers from a bundle.

    ``path_or_repo`` defaults to the released model on the Hub (:data:`DEFAULT_REPO`); pass a
    local ``.safetensors`` path or another repo id to override.

    Returns ``(model, tokenizers_by_modality, metadata)``. Tokenizer instances are shared
    across modalities that reference the same tokenizer (keyed by modality name in the
    returned dict). Model and tokenizers are returned in eval mode on ``device``.
    """
    device = require_runnable_device(device)  # fail fast before loading weights (e.g. on mps)
    config, model_sd, tokenizer_sds = _read_bundle(resolve_bundle(path_or_repo))
    meta = parse_config(config)

    # modality_configs MUST be built in config order — this order defines the model's
    # _modality_offsets (the column layout of the token tensor) and the averaging order.
    modality_configs = [
        ModalityConfig(
            name=m.name, num_quantizers=m.num_quantizers, codebook_size=m.codebook_size, signal_type=m.signal_type
        )
        for m in meta.modalities
    ]
    model = MultiModalRQTransformer(modality_configs=modality_configs, **meta.model_kwargs)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading RQ-Transformer weights: {unexpected[:10]}...")
    if missing:
        logger.warning("Missing keys when loading RQ-Transformer (likely tied weights): %s", missing[:10])
    model.to(device=device, dtype=dtype).eval()

    # Build each unique tokenizer once, then fan out to every modality that uses it.
    tokenizer_instances: dict[str, SignalTokenizer] = {}
    for stem in meta.unique_tokenizers:
        spec = meta.tokenizers[stem]
        tok = SignalTokenizer(**spec.tokenizer_kwargs)
        tmissing, tunexpected = tok.load_state_dict(tokenizer_sds[stem], strict=False)
        if tunexpected:
            raise RuntimeError(f"Unexpected keys loading tokenizer {stem!r}: {tunexpected[:10]}...")
        if tmissing:
            logger.warning("Missing keys loading tokenizer %r: %s", stem, tmissing[:10])
        tokenizer_instances[stem] = tok.to(device=device, dtype=dtype).eval()

    tokenizers_by_modality = {m.name: tokenizer_instances[m.tokenizer] for m in meta.modalities}
    return model, tokenizers_by_modality, meta
