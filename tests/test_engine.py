"""Engine loop tests via a fake backend (no CTranslate2 required).

The backend abstraction lets us exercise the full long-form policy — prompt
build, N-best selection, context carry, seek — with a deterministic stub.
"""

from __future__ import annotations

import numpy as np

from whisper_lm_fusion.backends.base import Backend, RawResult
from whisper_lm_fusion.config import LoadConfig
from whisper_lm_fusion.engine import Engine

# minimal Whisper-like special token map
_SPECIAL = {
    "<|startoftranscript|>": 50258,
    "<|ko|>": 50264,
    "<|en|>": 50259,
    "<|transcribe|>": 50359,
    "<|0.00|>": 50365,
    "<|notimestamps|>": 50364,
    "<|startofprev|>": 50361,
    "<|endoftext|>": 50257,
}


class FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return _SPECIAL[token]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{i}" for i in ids if i < _SPECIAL["<|endoftext|>"])


class FakeBackend(Backend):
    supports_fusion = True

    def __init__(self, sequences):
        self._tok = FakeTokenizer()
        self._sequences = sequences
        self.calls: list[dict] = []

    @property
    def tokenizer(self):
        return self._tok

    def extract_features(self, chunk, sr):
        return ("feat", len(chunk))

    def generate(self, features, prompt, *, options, fusion):
        self.calls.append({"prompt": prompt, "fusion": fusion})
        idx = min(len(self.calls) - 1, len(self._sequences) - 1)
        seqs = self._sequences[idx]
        return RawResult(sequences_ids=seqs, scores=[-0.2] * len(seqs), no_speech_prob=0.0)


def _engine(backend: FakeBackend) -> Engine:
    cfg = LoadConfig(model_path="unused", lm_path=None)
    return Engine(backend, cfg)


def test_empty_audio_returns_empty():
    eng = _engine(FakeBackend([[[100, 101]]]))
    assert eng.transcribe(np.zeros(0, dtype=np.float32), 16000).text == ""


def test_single_chunk_decodes_text():
    backend = FakeBackend([[[100, 101]]])
    eng = _engine(backend)
    audio = np.ones(16000, dtype=np.float32)  # 1s < window => single iteration
    result = eng.transcribe(audio, 16000)
    assert result.text == "w100 w101"
    assert len(backend.calls) == 1


def test_short_tail_uses_notimestamps_prompt():
    backend = FakeBackend([[[100]]])
    eng = _engine(backend)
    eng.transcribe(np.ones(16000, dtype=np.float32), 16000)
    # cut == n (final chunk) => notimestamps token appended to SOT prompt
    assert _SPECIAL["<|notimestamps|>"] in backend.calls[0]["prompt"]


def test_fusion_off_passes_empty_fusion_kwargs():
    backend = FakeBackend([[[100]]])
    eng = _engine(backend)
    eng.transcribe(np.ones(16000, dtype=np.float32), 16000, lm_enabled=False)
    assert backend.calls[0]["fusion"].to_generate_kwargs() == {}


def test_language_override_changes_prompt_token():
    backend = FakeBackend([[[100]]])
    eng = _engine(backend)
    eng.transcribe(np.ones(16000, dtype=np.float32), 16000, language="en")
    assert _SPECIAL["<|en|>"] in backend.calls[0]["prompt"]


def test_keyword_overrides_reach_options():
    backend = FakeBackend([[[100]]])
    eng = _engine(backend)
    captured = {}
    orig = backend.generate

    def spy(features, prompt, *, options, fusion):
        captured["beam_size"] = options.beam_size
        return orig(features, prompt, options=options, fusion=fusion)

    backend.generate = spy
    eng.transcribe(np.ones(16000, dtype=np.float32), 16000, beam_size=8)
    assert captured["beam_size"] == 8


def test_fusion_requires_supporting_backend():
    backend = FakeBackend([[[100]]])
    backend.supports_fusion = False
    cfg = LoadConfig(model_path="unused", lm_path="lm.binary", verify_metadata=False)
    eng = Engine(backend, cfg)
    try:
        eng.transcribe(np.ones(16000, dtype=np.float32), 16000, lm_enabled=True)
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported fusion")
