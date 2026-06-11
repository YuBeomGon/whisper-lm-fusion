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


def test_fusion_on_emits_lm_fusion_kwargs():
    backend = FakeBackend([[[100]]])
    cfg = LoadConfig(
        model_path="unused", lm_path="lm.binary", verify_metadata=False, topk_default=50
    )
    eng = Engine(backend, cfg)
    eng.transcribe(
        np.ones(16000, dtype=np.float32), 16000, lm_enabled=True, alpha=0.2
    )
    kwargs = backend.calls[0]["fusion"].to_generate_kwargs()
    assert kwargs == {
        "lm_fusion_model_path": "lm.binary",
        "lm_fusion_alpha": 0.2,
        "lm_fusion_asr_topk": 50,
        "lm_fusion_debug": False,
    }
    # lm_fusion_beta must never be passed (SSOT §4)
    assert "lm_fusion_beta" not in kwargs


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


def test_temperature_fallback_adds_candidates_and_can_win():
    class FallbackBackend(FakeBackend):
        def generate(self, features, prompt, *, options, fusion):
            self.calls.append({"prompt": prompt, "fusion": fusion, "temp": options.sampling_temperature})
            if len(self.calls) == 1:
                return RawResult(sequences_ids=[[10]], scores=[-1.5], no_speech_prob=0.0)
            return RawResult(sequences_ids=[[20, 21]], scores=[-0.2], no_speech_prob=0.0)

    backend = FallbackBackend([[[10]], [[20, 21]]])
    eng = _engine(backend)
    result = eng.transcribe(
        np.ones(16000, dtype=np.float32),
        16000,
        fallback_policy="low_logprob",
        temperature_fallback=(0.2,),
        return_nbest=True,
    )
    assert result.text == "w20 w21"
    assert len(backend.calls) == 2
    assert backend.calls[1]["temp"] == 0.2
    assert result.nbest is not None
    assert result.nbest[0][1]["source"] == "temperature_fallback"


def test_per_window_language_override_uses_detected_language():
    from whisper_lm_fusion.backends.base import LanguageProb

    class LanguageBackend(FakeBackend):
        def detect_language(self, features):
            return [LanguageProb("en", 0.95)]

    backend = LanguageBackend([[[100]]])
    eng = _engine(backend)
    eng.transcribe(
        np.ones(16000, dtype=np.float32),
        16000,
        language="ko",
        language_policy="per_window_confident",
        language_override_prob=0.7,
    )
    assert _SPECIAL["<|en|>"] in backend.calls[0]["prompt"]


def test_runtime_cjk_kana_suppress_tokens_are_appended():
    class ScriptTokenizer(FakeTokenizer):
        def get_vocab(self):
            return {"한": 10, "中": 11, "カ": 12, "A": 13}

        def decode(self, ids, skip_special_tokens: bool = True):
            table = {10: "한", 11: "中", 12: "カ", 13: "A"}
            return "".join(table.get(i, f"w{i}") for i in ids)

    class ScriptBackend(FakeBackend):
        def __init__(self):
            super().__init__([[[100]]])
            self._tok = ScriptTokenizer()
            self.seen_suppress = None

        def generate(self, features, prompt, *, options, fusion):
            self.seen_suppress = options.suppress_tokens
            return super().generate(features, prompt, options=options, fusion=fusion)

    backend = ScriptBackend()
    eng = _engine(backend)
    eng.transcribe(np.ones(16000, dtype=np.float32), 16000)
    assert -1 in backend.seen_suppress
    assert 11 in backend.seen_suppress
    assert 12 in backend.seen_suppress
    assert 10 not in backend.seen_suppress
    assert 13 not in backend.seen_suppress
