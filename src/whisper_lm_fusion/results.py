"""Transcription result type.

``text`` is always present; ``segments`` / ``scores`` / ``nbest`` are populated
only when the corresponding ``return_*`` flag is set on ``DecodeOptions``. The
object stringifies to its text so simple callers can treat it like a string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Segment:
    text: str
    start: float
    end: float
    logprob: float
    no_speech_prob: float


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment] | None = None
    scores: list[float] | None = None
    nbest: list[list[Any]] | None = None

    def __str__(self) -> str:  # let `str(result)` == result.text
        return self.text
