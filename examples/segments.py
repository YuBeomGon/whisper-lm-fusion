"""Structured output — segments, per-window scores, and N-best candidates.

Run:  python examples/segments.py path/to/ct2-model path/to/speech.wav

`text` is always present; segments/scores/nbest are populated only when the
matching return_* flag is set. N-best entries carry their `source`
(base / dual_language / temperature_fallback) and `language`.
"""

from __future__ import annotations

import sys

import soundfile as sf

import whisper_lm_fusion


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    model_path, audio_path = sys.argv[1], sys.argv[2]

    engine = whisper_lm_fusion.load(model_path, device="cuda", compute_type="float16")
    audio, sr = sf.read(audio_path)

    result = engine.transcribe(
        audio,
        sr,
        return_segments=True,
        return_scores=True,
        return_nbest=True,
    )

    print("TEXT:", result.text)

    print("\nSEGMENTS:")
    for seg in result.segments or []:
        print(f"  [{seg.start:6.2f}-{seg.end:6.2f}] logp={seg.logprob:6.2f} {seg.text}")

    print("\nPER-WINDOW N-BEST (first window):")
    for window in (result.nbest or [])[:1]:
        for cand in window:
            print(f"  logp={cand['logprob']:6.2f} src={cand['source']:<18} {cand['text']}")


if __name__ == "__main__":
    main()
