"""Pure decoding helpers — replaceable self-evolve decoding units.

These helpers implement the *compressed control surface* of the phase3 strategy
inventory: RMS/silence long-form cut, timestamp seek, N-best selection,
conditional fallback gates, no-speech emission gating, and optional MBR-style
selection.  They intentionally stay free of CTranslate2 imports.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from whisper_lm_fusion.config import DecodeOptions


def compression_ratio(text: str) -> float:
    """gzip compression ratio; high values flag degenerate/repetitive output."""
    payload = text.encode("utf-8")
    if not payload:
        return 0.0
    compressed = gzip.compress(payload)
    if not compressed:
        return 0.0
    return len(payload) / len(compressed)


# --- N-best / branch selection --------------------------------------------

@dataclass
class Candidate:
    """One raw candidate from a decode branch."""

    token_ids: list[int]
    text: str
    logprob: float
    no_speech_prob: float = 0.0
    source: str = "base"
    language: str | None = None
    temperature: float | None = None
    compression: float = 0.0
    degenerate: bool = False


@dataclass
class Hypothesis:
    """Selected hypothesis and diagnostics for one window."""

    token_ids: list[int]
    text: str
    logprob: float
    degenerate: bool
    no_speech_prob: float
    candidates: list[Candidate] = field(default_factory=list)
    dropped_no_speech: bool = False
    selected_source: str = "base"
    selected_language: str | None = None
    selected_temperature: float | None = None


DecodeText = Callable[[Sequence[int]], str]


def make_candidates(
    scores: Sequence[float],
    sequences_ids: Sequence[Sequence[int]],
    decode_text: DecodeText,
    options: DecodeOptions,
    *,
    no_speech_prob: float = 0.0,
    source: str = "base",
    language: str | None = None,
    temperature: float | None = None,
) -> list[Candidate]:
    """Normalize backend N-best output into selector-ready candidates."""
    candidates: list[Candidate] = []
    for logprob, token_ids in zip(scores, sequences_ids):
        text = decode_text(token_ids).strip()
        cr = compression_ratio(text)
        candidates.append(
            Candidate(
                token_ids=list(token_ids),
                text=text,
                logprob=float(logprob),
                no_speech_prob=float(no_speech_prob or 0.0),
                source=source,
                language=language,
                temperature=temperature,
                compression=cr,
                degenerate=cr > options.compression_ratio_threshold,
            )
        )
    return candidates


def _token_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Small Levenshtein distance for N-best MBR.  Used only on short candidate sets."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i]
        for j, bj in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj)))
        prev = cur
    return prev[-1]


def _mbr_rank(c: Candidate, pool: Sequence[Candidate]) -> tuple[float, float]:
    denom = max(len(pool) - 1, 1)
    dist = 0.0
    for other in pool:
        if other is c:
            continue
        norm = max(len(c.token_ids), len(other.token_ids), 1)
        dist += _token_distance(c.token_ids, other.token_ids) / norm
    return (dist / denom, -c.logprob)


def _length_bonus_allowed(cand: Candidate, best: Candidate, options: DecodeOptions) -> bool:
    if cand.degenerate or best.degenerate:
        return False
    if cand.logprob < best.logprob - options.score_margin:
        return False
    cand_len = max(len(cand.text), len(cand.token_ids))
    best_len = max(len(best.text), len(best.token_ids), 1)
    return cand_len >= best_len * options.min_length_ratio_for_longer


def select_candidate(candidates: Sequence[Candidate], options: DecodeOptions) -> Candidate | None:
    """Select a candidate using one of the self-evolve selection strategies.

    Policies:
    - ``axis_aware``: suppress repetitive candidates first, then maximize logprob.
    - ``longer_within_margin``: recover likely deletions by accepting a longer clean
      candidate within ``score_margin`` of the current best.
    - ``token_mbr``: choose the token medoid of clean N-best candidates.
    - ``logprob``: pure score baseline.
    """
    pool = [c for c in candidates if c.text]
    if not pool:
        return None

    if options.selection_policy == "logprob":
        return max(pool, key=lambda c: c.logprob)

    clean = [c for c in pool if not c.degenerate]
    search_pool = clean or pool

    if options.selection_policy == "token_mbr" and len(search_pool) > 1:
        return min(search_pool, key=lambda c: _mbr_rank(c, search_pool))

    best = max(search_pool, key=lambda c: c.logprob)

    use_longer = (
        options.selection_policy == "longer_within_margin"
        or options.prefer_longer_within_margin
    )
    if use_longer:
        for cand in sorted(search_pool, key=lambda c: (len(c.text), c.logprob), reverse=True):
            if _length_bonus_allowed(cand, best, options):
                return cand
    return best


def select_hypothesis(
    scores: Sequence[float],
    sequences_ids: Sequence[Sequence[int]],
    decode_text: DecodeText,
    options: DecodeOptions,
    no_speech_prob: float = 0.0,
) -> Hypothesis:
    """Compatibility wrapper around candidate selection for a single branch."""
    candidates = make_candidates(
        scores,
        sequences_ids,
        decode_text,
        options,
        no_speech_prob=no_speech_prob,
        source="base",
    )
    return hypothesis_from_candidates(candidates, options)


def hypothesis_from_candidates(
    candidates: Sequence[Candidate],
    options: DecodeOptions,
) -> Hypothesis:
    selected = select_candidate(candidates, options)
    if selected is None:
        return Hypothesis([], "", float("-inf"), True, 0.0, list(candidates))
    dropped = should_drop_for_no_speech(selected, options)
    return Hypothesis(
        token_ids=list(selected.token_ids),
        text="" if dropped else selected.text,
        logprob=selected.logprob,
        degenerate=selected.degenerate,
        no_speech_prob=selected.no_speech_prob,
        candidates=list(candidates),
        dropped_no_speech=dropped,
        selected_source=selected.source,
        selected_language=selected.language,
        selected_temperature=selected.temperature,
    )


def should_fallback(hyp: Hypothesis, options: DecodeOptions) -> bool:
    """Whether to run conditional temperature fallback for this window."""
    if options.fallback_policy == "off" or not options.temperature_fallback:
        return False
    low_logprob = hyp.logprob < options.logprob_threshold
    degenerate = hyp.degenerate
    if options.fallback_policy == "always":
        return True
    if options.fallback_policy == "low_logprob":
        return low_logprob
    if options.fallback_policy == "degenerate":
        return degenerate
    return low_logprob or degenerate


def should_drop_for_no_speech(candidate: Candidate, options: DecodeOptions) -> bool:
    """Joint no-speech emission gate.

    This is intentionally not a hard ``no_speech_prob > threshold`` rule.  It only
    drops text when no_speech is high *and* the text is already low-confidence or
    degenerate, which matches the phase3 hallucination guard lesson.
    """
    if candidate.no_speech_prob <= options.no_speech_threshold:
        return False
    return candidate.logprob < options.no_speech_logprob_threshold or candidate.degenerate


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
