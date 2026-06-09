"""whisper-lm-fusion: a generic faster-whisper-style STT wrapper.

Wraps a (patched) CTranslate2 Whisper backend, hides the long-form decoding
pipeline, and exposes ``load()`` + ``transcribe()`` with tunable parameters.
Domain logic (glossary, corpus, KenLM build) is NOT here — see docs/design.md.

    import whisper_lm_fusion
    engine = whisper_lm_fusion.load("path/to/ct2-model")
    text = engine.transcribe(audio, sr).text
"""

from __future__ import annotations

from whisper_lm_fusion.backends import (
    Backend,
    RawResult,
    available_backends,
    create_backend,
    register_backend,
)
from whisper_lm_fusion.config import DecodeOptions, FusionOptions, LoadConfig
from whisper_lm_fusion.engine import Engine, load
from whisper_lm_fusion.metadata import MetadataMismatchError
from whisper_lm_fusion.results import Segment, TranscriptionResult

__version__ = "0.1.0"

__all__ = [
    "load",
    "Engine",
    "DecodeOptions",
    "LoadConfig",
    "FusionOptions",
    "TranscriptionResult",
    "Segment",
    "MetadataMismatchError",
    # backend factory
    "Backend",
    "RawResult",
    "create_backend",
    "register_backend",
    "available_backends",
    "__version__",
]
