"""Minimal stt-wrapper usage.

Run:  python examples/basic.py path/to/ct2-model path/to/speech.wav
Requires a working CTranslate2 install and a CT2-converted Whisper model.
"""

from __future__ import annotations

import sys

import soundfile as sf

import stt_wrapper


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    model_path, audio_path = sys.argv[1], sys.argv[2]

    engine = stt_wrapper.load(model_path, device="cuda", compute_type="float16")
    audio, sr = sf.read(audio_path)

    result = engine.transcribe(audio, sr, return_segments=True)
    print(result.text)
    for seg in result.segments or []:
        print(f"  [{seg.start:6.2f}-{seg.end:6.2f}] {seg.text}")


if __name__ == "__main__":
    main()
