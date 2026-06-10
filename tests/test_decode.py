"""Unit tests for the pure decoding helpers (no CTranslate2 required)."""

from __future__ import annotations

import numpy as np

from whisper_lm_fusion._decode import (
    advance_seek,
    compression_ratio,
    decide_cut,
    frame_energy,
    select_hypothesis,
)
from whisper_lm_fusion.config import DecodeOptions


def test_compression_ratio_flags_repetition():
    normal = compression_ratio("the quick brown fox jumps over the lazy dog")
    repeated = compression_ratio("ha " * 200)
    assert repeated > normal
    assert compression_ratio("") == 0.0


def test_select_hypothesis_prefers_non_degenerate():
    opts = DecodeOptions(compression_ratio_threshold=2.4)
    # candidate 0: high logprob but degenerate (repetitive); candidate 1: clean
    scores = [-0.1, -0.5]
    seqs = [[1, 1, 1], [2, 3, 4]]
    texts = {(1, 1, 1): "na " * 100, (2, 3, 4): "hello world"}
    hyp = select_hypothesis(scores, seqs, lambda ids: texts[tuple(ids)], opts)
    assert hyp.text == "hello world"
    assert not hyp.degenerate


def test_select_hypothesis_logprob_tiebreak_when_both_clean():
    opts = DecodeOptions()
    scores = [-0.9, -0.2]
    seqs = [[1], [2]]
    texts = {(1,): "alpha", (2,): "beta"}
    hyp = select_hypothesis(scores, seqs, lambda ids: texts[tuple(ids)], opts)
    assert hyp.text == "beta"  # higher logprob wins among non-degenerate


def test_frame_energy_silence_floor():
    sr = 16000
    audio = np.concatenate([np.ones(sr) * 0.5, np.zeros(sr)]).astype(np.float32)
    opts = DecodeOptions()
    e = frame_energy(audio, sr, opts)
    assert e.rms.size > 0
    assert e.silence_floor >= 0.0


def test_decide_cut_short_tail_returns_hard_end():
    sr = 16000
    opts = DecodeOptions(window_seconds=30.0)
    audio = np.zeros(sr, dtype=np.float32)  # 1s < window
    e = frame_energy(audio, sr, opts)
    window = int(opts.window_seconds * sr)
    cut, silence = decide_cut(0, window, len(audio), e, sr, opts)
    assert cut == len(audio)
    assert silence is False


def test_decide_cut_finds_silence_trough_in_full_window():
    sr = 16000
    opts = DecodeOptions(window_seconds=30.0, min_advance_seconds=20.0)
    # 30s window: loud, a silent trough near 25s, then loud again
    loud = np.ones(int(25 * sr), dtype=np.float32) * 0.5
    silent = np.zeros(int(1 * sr), dtype=np.float32)
    loud2 = np.ones(int(10 * sr), dtype=np.float32) * 0.5
    audio = np.concatenate([loud, silent, loud2])
    e = frame_energy(audio, sr, opts)
    window = int(opts.window_seconds * sr)
    cut, silence = decide_cut(0, window, len(audio), e, sr, opts)
    assert silence is True
    # cut should land inside the silent region (~25-26s)
    assert 24 * sr <= cut <= 27 * sr


def test_advance_seek_silence_cut_uses_cut():
    opts = DecodeOptions()
    sr = 16000
    seek = advance_seek(0, 12345, 12345, 30 * sr, [], 1000, sr, opts, silence_cut=True)
    assert seek == 12345


def test_advance_seek_default_full_window_when_no_timestamps():
    opts = DecodeOptions(window_seconds=30.0)
    sr = 16000
    window = int(opts.window_seconds * sr)
    seek = advance_seek(0, window, window, window, [], 1000, sr, opts, silence_cut=False)
    assert seek == window  # full window advance


def test_select_hypothesis_can_prefer_longer_within_margin():
    opts = DecodeOptions(
        selection_policy="longer_within_margin",
        score_margin=0.15,
        min_length_ratio_for_longer=1.2,
    )
    scores = [-0.20, -0.30]
    seqs = [[1], [2, 3, 4]]
    texts = {(1,): "짧음", (2, 3, 4): "더 긴 정상 문장"}
    hyp = select_hypothesis(scores, seqs, lambda ids: texts[tuple(ids)], opts)
    assert hyp.text == "더 긴 정상 문장"


def test_no_speech_joint_gate_drops_bad_hypothesis():
    opts = DecodeOptions(no_speech_threshold=0.5, no_speech_logprob_threshold=-0.8)
    hyp = select_hypothesis(
        [-1.2],
        [[1, 2]],
        lambda ids: "hallucinated text",
        opts,
        no_speech_prob=0.9,
    )
    assert hyp.text == ""
    assert hyp.dropped_no_speech is True


def test_token_mbr_chooses_medoid_candidate():
    opts = DecodeOptions(selection_policy="token_mbr")
    scores = [-0.1, -0.2, -0.3]
    seqs = [[1, 2, 3], [1, 2, 4], [9, 9, 9]]
    texts = {
        (1, 2, 3): "가 나 다",
        (1, 2, 4): "가 나 라",
        (9, 9, 9): "전혀 다름",
    }
    hyp = select_hypothesis(scores, seqs, lambda ids: texts[tuple(ids)], opts)
    assert hyp.token_ids in ([1, 2, 3], [1, 2, 4])
