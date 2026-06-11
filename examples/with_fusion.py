"""KenLM shallow fusion — requires the patched CTranslate2 fork (WITH_KENLM=ON).

Run:  python examples/with_fusion.py path/to/ct2-model path/to/domain.binary path/to/speech.wav

Fusion turns on only when lm_path is set AND lm_enabled=True AND alpha > 0;
otherwise this decodes identically to plain STT. The KenLM .binary must have a
`<binary>.meta.json` sidecar (tokenizer/model hash) unless verify_lm_metadata=False.
See docs/guide/fusion.md.
"""

from __future__ import annotations

import sys

import soundfile as sf

import whisper_lm_fusion


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    model_path, lm_path, audio_path = sys.argv[1], sys.argv[2], sys.argv[3]

    engine = whisper_lm_fusion.load(
        model_path,
        lm_path=lm_path,
        alpha_default=0.2,        # load-time default; overridable per request
        verify_lm_metadata=True,  # refuse on tokenizer/model hash mismatch
        # tokenizer_hash="<hash>",  # provide to actually compare against the sidecar
    )
    audio, sr = sf.read(audio_path)

    fused = engine.transcribe(audio, sr, lm_enabled=True, alpha=0.2).text
    baseline = engine.transcribe(audio, sr, lm_enabled=False).text

    print("fusion  :", fused)
    print("baseline:", baseline)


if __name__ == "__main__":
    main()
