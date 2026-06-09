"""Artifact metadata validation — the tokenizer-hash gate (design.md §3).

A KenLM ``.binary`` is built over Whisper BPE token-ids by the pipeline at
build-time. If the tokenizer that produced the corpus differs from the one the
wrapper decodes with, token-ids silently mismatch and quietly degrade quality.
So the wrapper refuses to load the LM unless the artifact metadata's
``tokenizer_hash`` / ``ct2_model_hash`` match the current model.

The metadata file is expected next to the ``.binary`` as ``<binary>.meta.json``
(or an explicit path). Absence is treated per ``strict``.
"""

from __future__ import annotations

import json
from pathlib import Path


class MetadataMismatchError(RuntimeError):
    """Raised when artifact metadata does not match the loaded model."""


def metadata_path_for(lm_path: Path) -> Path:
    return lm_path.with_suffix(lm_path.suffix + ".meta.json")


def load_metadata(lm_path: Path, explicit: Path | None = None) -> dict | None:
    path = explicit or metadata_path_for(lm_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_metadata(
    metadata: dict | None,
    *,
    tokenizer_hash: str | None,
    ct2_model_hash: str | None = None,
    strict: bool = True,
) -> None:
    """Raise ``MetadataMismatchError`` if hashes disagree.

    ``strict`` controls missing-metadata handling: when True (default) a missing
    file or missing hash field is a hard error, because an unverifiable LM is the
    exact silent-failure case this gate exists to prevent.
    """
    if metadata is None:
        if strict:
            raise MetadataMismatchError(
                "LM artifact metadata not found; cannot verify tokenizer_hash. "
                "Provide <binary>.meta.json or set verify_metadata=False to bypass."
            )
        return

    _check("tokenizer_hash", metadata.get("tokenizer_hash"), tokenizer_hash, strict)
    if ct2_model_hash is not None:
        _check("ct2_model_hash", metadata.get("ct2_model_hash"), ct2_model_hash, strict)


def _check(name: str, expected: str | None, actual: str | None, strict: bool) -> None:
    if expected is None:
        if strict:
            raise MetadataMismatchError(f"artifact metadata missing '{name}'")
        return
    if actual is not None and expected != actual:
        raise MetadataMismatchError(
            f"{name} mismatch: artifact={expected!r} model={actual!r}. "
            "The LM was built with a different tokenizer/model; refusing to load."
        )
