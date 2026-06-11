"""Plain STT (no LM fusion) — works with stock PyPI ctranslate2.

Run:  python examples/plain_stt.py path/to/ct2-model path/to/speech.wav

The default DecodeOptions are a conservative long-form backbone; this example
shows a few common request-time overrides without turning on any costly policy.
See docs/guide/parameters.md for the full surface.
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
    audio, sr = sf.read(audio_path)  # caller owns decode/resample (mono float waveform)

    result = engine.transcribe(
        audio,
        sr,
        language="ko",
        beam_size=5,
        suppress_cjk_kana=True,  # keep Hangul/Latin/digits, drop stray CJK/Kana
    )
    print(result.text)


if __name__ == "__main__":
    main()
