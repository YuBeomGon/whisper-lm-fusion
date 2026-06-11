"""Combining the self-evolve decoding policies via DecodeOptions.

Run:  python examples/tune_policies.py path/to/ct2-model path/to/speech.wav

Each policy switches internal pipeline logic (see docs/guide/pipeline.md). This
is the kind of option set an external sweep runner would vary; here it is just a
"strong" preset assembled by hand.
"""

from __future__ import annotations

import sys

import soundfile as sf

import whisper_lm_fusion
from whisper_lm_fusion import DecodeOptions


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    model_path, audio_path = sys.argv[1], sys.argv[2]

    options = DecodeOptions(
        # N-best selection: recover likely deletions on fast/dense speech
        selection_policy="longer_within_margin",
        score_margin=0.10,
        min_length_ratio_for_longer=1.05,
        # retry only gate-failing windows up a temperature ladder
        fallback_policy="gate_fail",
        temperature_fallback=(0.2, 0.4, 0.6),
        # let confident windows override the fixed language token
        language_policy="per_window_confident",
        language_override_prob=0.7,
        # Korean call-center: drop stray kana/CJK, keep Hangul/Latin/digits
        suppress_cjk_kana=True,
        # verify fluent tails against acoustics (no-op if backend lacks the hook)
        align_tail_trim=True,
    )

    engine = whisper_lm_fusion.load(model_path, device="cuda", compute_type="float16")
    audio, sr = sf.read(audio_path)

    result = engine.transcribe(audio, sr, options=options)
    print(result.text)


if __name__ == "__main__":
    main()
