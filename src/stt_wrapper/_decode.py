"""Pure decoding helpers — the replaceable base-logic units.

These implement the verified backbone ported from ct2-lm-fusion-runner
(window loop / RMS silence cut / timestamp seek / N-best gate), but as small,
side-effect-free functions so future decoding_strategy work (language override,
align trim, MBR rerank, ...) can swap or wrap them without touching the engine.

Nothing here imports CTranslate2; everything is testable on NumPy alone.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stt_wrapper.config import DecodeOptions


def compression_ratio(text: str) -> float:
    """gzip compression ratio; high values flag degenerate/repetitive output."""
    payload = text.encode("utf-8")
    if not payload:
        return 0.0
    compressed = gzip.compress(payload)
    if not compressed:
        return 0.0
    return len(payload) / len(compressed)


# --- N-best gate -----------------------------------------------------------

@dataclass
class Hypothesis:
    """One decoded candidate after selection."""

    token_ids: list[int]
    text: str
    logprob: float
    degenerate: bool
    no_speech_prob: float


def _is_better(
    cand_logprob: float,
    cand_degenerate: bool,
    best_logprob: float,
    best_degenerate: bool,
    have_best: bool,
) -> bool:
    """Prefer non-degenerate, then higher logprob (ported selection rule)."""
    if not have_best:
        return True
    if best_degenerate and not cand_degenerate:
        return True
    return best_degenerate == cand_degenerate and cand_logprob > best_logprob


def select_hypothesis(
    scores: Sequence[float],
    sequences_ids: Sequence[Sequence[int]],
    decode_text,
    options: DecodeOptions,
    no_speech_prob: float = 0.0,
) -> Hypothesis:
    """Pick the best N-best candidate.

    ``decode_text`` maps token ids -> text (tokenizer.decode); injected so this
    stays free of any tokenizer/CT2 dependency and is unit-testable.
    """
    best = Hypothesis([], "", float("-inf"), True, float(no_speech_prob or 0.0))
    have_best = False
    for logprob, token_ids in zip(scores, sequences_ids):
        text = decode_text(token_ids).strip()
        degenerate = compression_ratio(text) > options.compression_ratio_threshold
        if _is_better(logprob, degenerate, best.logprob, best.degenerate, have_best):
            best = Hypothesis(list(token_ids), text, float(logprob), degenerate, best.no_speech_prob)
            have_best = True
    return best


# --- RMS silence cut -------------------------------------------------------

@dataclass
class FrameEnergy:
    rms: np.ndarray
    hop: int
    silence_floor: float


def frame_energy(audio: np.ndarray, sr: int, options: DecodeOptions) -> FrameEnergy:
    """Per-frame RMS energy and a silence floor percentile."""
    hop = max(int(options.timestamp_resolution * sr), 1)
    n_frames = len(audio) // hop
    if n_frames <= 0:
        return FrameEnergy(np.zeros(0, dtype=np.float64), hop, 0.0)
    frames = audio[: n_frames * hop].astype(np.float64).reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    floor = float(np.percentile(rms, options.silence_percentile))
    return FrameEnergy(rms, hop, floor)


def decide_cut(
    seek: int,
    window: int,
    n: int,
    energy: FrameEnergy,
    sr: int,
    options: DecodeOptions,
) -> tuple[int, bool]:
    """Decide where to cut the current chunk.

    Returns ``(cut_sample, silence_cut)``. A full window prefers an RMS trough
    at/under the silence floor so long utterances aren't split mid-speech.
    """
    hard_end = min(seek + window, n)
    if hard_end != seek + window or energy.rms.size == 0:
        return hard_end, False

    f0 = (seek + int(options.min_advance_seconds * sr)) // energy.hop
    f1 = hard_end // energy.hop
    region = energy.rms[f0:f1]
    if region.size == 0:
        return hard_end, False
    local = int(np.argmin(region))
    if region[local] <= energy.silence_floor:
        cut = max((f0 + local) * energy.hop, seek + 1)
        return cut, True
    return hard_end, False


# --- timestamp seek --------------------------------------------------------

def advance_seek(
    seek: int,
    cut: int,
    chunk_len: int,
    window: int,
    token_ids: Sequence[int],
    timestamp_begin: int,
    sr: int,
    options: DecodeOptions,
    silence_cut: bool,
) -> int:
    """Next seek position. On a silence cut, advance to the cut; otherwise use
    the last emitted timestamp token (bounded) for adaptive advance."""
    if silence_cut:
        return cut

    timestamps = [t - timestamp_begin for t in token_ids if t >= timestamp_begin]
    advance_seconds = options.window_seconds
    if chunk_len >= window and timestamps and timestamps[-1] > 0:
        last_ts = timestamps[-1] * options.timestamp_resolution
        if options.min_advance_seconds <= last_ts <= options.window_seconds:
            advance_seconds = last_ts
    return seek + max(int(advance_seconds * sr), 1)
