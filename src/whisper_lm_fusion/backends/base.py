"""Backend abstraction.

A ``Backend`` is the *model execution provider* — the thing that turns an audio
chunk + a Whisper prompt into decoded hypotheses. The long-form decoding policy
(window loop, silence cut, timestamp seek, language branch, fallback, N-best
selection) lives in the engine and is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from whisper_lm_fusion.config import DecodeOptions, FusionOptions


@dataclass
class RawResult:
    """Normalized output of a single ``generate`` call (one audio chunk)."""

    sequences_ids: list[list[int]]
    scores: list[float]
    no_speech_prob: float = 0.0


@dataclass
class LanguageProb:
    """Normalized language detection result."""

    language: str
    probability: float


class Backend(ABC):
    """Interface every backend implements."""

    supports_fusion: bool = False

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        ...

    @abstractmethod
    def extract_features(self, chunk: np.ndarray, sr: int) -> Any:
        """Turn a mono float waveform chunk into backend-native features."""

    @abstractmethod
    def generate(
        self,
        features: Any,
        prompt: list[int],
        *,
        options: DecodeOptions,
        fusion: FusionOptions,
    ) -> RawResult:
        """Decode one chunk and return normalized hypotheses."""

    def detect_language(self, features: Any) -> list[LanguageProb]:
        """Optional Whisper language detection hook.

        Backends that do not expose it return an empty list.  The engine keeps
        language-policy logic generic and will simply fall back to the configured
        base language when this returns no result.
        """
        return []

    def align_tail_trim(
        self,
        features: Any,
        token_ids: list[int],
        *,
        options: DecodeOptions,
    ) -> list[int] | None:
        """Optional alignment-posterior tail trim hook.

        Return a trimmed token list, or ``None`` to leave the hypothesis unchanged.
        This is deliberately a hook because CT2/HF/other backends expose alignment
        differently.  The public config still lets a sweep toggle the algorithm;
        unsupported backends are safe no-ops.
        """
        return None
