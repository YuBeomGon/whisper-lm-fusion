"""Tests for the tokenizer-hash gate (no CTranslate2 required)."""

from __future__ import annotations

import json

import pytest

from stt_wrapper.metadata import (
    MetadataMismatchError,
    load_metadata,
    metadata_path_for,
    verify_metadata,
)


def test_matching_hashes_pass():
    md = {"tokenizer_hash": "abc", "ct2_model_hash": "xyz"}
    verify_metadata(md, tokenizer_hash="abc", ct2_model_hash="xyz", strict=True)


def test_tokenizer_hash_mismatch_raises():
    md = {"tokenizer_hash": "abc"}
    with pytest.raises(MetadataMismatchError):
        verify_metadata(md, tokenizer_hash="different", strict=True)


def test_missing_metadata_strict_raises():
    with pytest.raises(MetadataMismatchError):
        verify_metadata(None, tokenizer_hash="abc", strict=True)


def test_missing_metadata_non_strict_passes():
    verify_metadata(None, tokenizer_hash="abc", strict=False)


def test_missing_field_strict_raises():
    with pytest.raises(MetadataMismatchError):
        verify_metadata({}, tokenizer_hash="abc", strict=True)


def test_load_metadata_sidecar(tmp_path):
    lm = tmp_path / "lm.binary"
    lm.write_bytes(b"\x00")
    meta = metadata_path_for(lm)
    assert meta.name == "lm.binary.meta.json"
    meta.write_text(json.dumps({"tokenizer_hash": "h"}), encoding="utf-8")
    loaded = load_metadata(lm)
    assert loaded == {"tokenizer_hash": "h"}


def test_load_metadata_absent_returns_none(tmp_path):
    lm = tmp_path / "lm.binary"
    assert load_metadata(lm) is None
