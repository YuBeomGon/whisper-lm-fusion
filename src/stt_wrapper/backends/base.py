"""Backend abstraction.

A ``Backend`` is the *model execution provider* — the thing that turns an audio
chunk + a Whisper prompt into decoded hypotheses. The long-form decoding policy
(window loop, silence cut, timestamp seek, N-best gate) lives in the engine and
is backend-agnostic; only the actual feature extraction and ``generate`` differ
between CTranslate2, TensorRT-LLM, HuggingFace Whisper, OpenAI Whisper, etc.

Implement a new backend by subclassing ``Backend`` and registering it (see
``stt_wrapper.backends.register_backend``). Backends are constructed lazily by
the factory so importing the wrapper never forces a heavy runtime import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from stt_wrapper.config import DecodeOptions, FusionOptions


@dataclass
class RawResult:
    """Normalized output of a single ``generate`` call (one audio chunk)."""

    sequences_ids: list[list[int]]
    scores: list[float]
    no_speech_prob: float = 0.0


class Backend(ABC):
    """Interface every backend implements.

    ``tokenizer`` must be a Whisper tokenizer exposing ``decode`` and
    ``convert_tokens_to_ids`` (the engine uses it to build prompts and decode
    token ids). ``supports_fusion`` advertises whether KenLM fusion kwargs are
    honored; non-fusion backends simply ignore the passed ``FusionOptions``.
    """

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
