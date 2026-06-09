"""Dev smoke: run the wrapper on one real long-form file and print the text.

NOT an evaluation (no CER, no domain rules) — just confirms the package's
long-form loop runs end-to-end on real audio. CER measurement lives outside
this package.

Usage (needs patched CT2 env):
    source scripts/ct2_env.sh
    PYTHONPATH=src python scripts/smoke_transcribe.py <model_dir> <audio.wav> [processor_id]
"""

from __future__ import annotations

import sys
import time

import librosa
import numpy as np
import soundfile as sf

import whisper_lm_fusion

TARGET_SR = 16000


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    model_dir, audio_path = sys.argv[1], sys.argv[2]
    processor_id = sys.argv[3] if len(sys.argv) > 3 else None

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:  # safety: collapse to mono
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    dur = len(audio) / TARGET_SR
    print(f"audio: {audio_path}")
    print(f"  duration: {dur:.1f}s  (orig sr={sr})")

    engine = whisper_lm_fusion.load(
        model_dir,
        backend="ct2",
        device="cuda",
        compute_type="float16",
        processor_path=processor_id,
    )

    t0 = time.time()
    result = engine.transcribe(audio, TARGET_SR, return_segments=True)
    elapsed = time.time() - t0

    print(f"  decode: {elapsed:.1f}s  RTF={elapsed / dur:.4f}")
    print(f"  windows/segments: {len(result.segments or [])}")
    print(f"  chars: {len(result.text)}")
    print("--- text (head 500) ---")
    print(result.text[:500])


if __name__ == "__main__":
    main()
