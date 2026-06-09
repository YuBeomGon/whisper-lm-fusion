"""stt-wrapper: a generic faster-whisper-style STT wrapper.

Wraps a (patched) CTranslate2 Whisper backend, hides the long-form decoding
pipeline, and exposes ``load()`` + ``transcribe()`` with tunable parameters.
Domain logic (glossary, corpus, KenLM build) is NOT here — see docs/design.md.

    import stt_wrapper
    engine = stt_wrapper.load("path/to/ct2-model")
    text = engine.transcribe(audio, sr).text
"""

from __future__ import annotations

from stt_wrapper.backends import (
    Backend,
    RawResult,
    available_backends,
    create_backend,
    register_backend,
)
from stt_wrapper.config import DecodeOptions, FusionOptions, LoadConfig
from stt_wrapper.engine import Engine, load
from stt_wrapper.metadata import MetadataMismatchError
from stt_wrapper.results import Segment, TranscriptionResult

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
