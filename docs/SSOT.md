# stt-wrapper SSOT

**Date**: 2026-06-09

This document is the canonical project reference for `stt-wrapper`. Other docs
should summarize and link here instead of duplicating policy.

## 1. Project Identity

`stt-wrapper` is a generic, faster-whisper-style STT wrapper.

It exposes two main calls:

- `stt_wrapper.load(...)`: load a backend model once.
- `engine.transcribe(audio, sr, ...)`: transcribe one already-decoded waveform.

The wrapper owns long-form decoding mechanics such as windowing, silence-aware
cuts, timestamp-adaptive seek, prompt construction, and N-best selection. It
does not own domain data or customer-specific policy.

## 2. Responsibility Boundary

The wrapper knows only:

- audio waveform and sample rate
- model path
- optional KenLM binary path
- decoding parameters
- backend selection
- tokenizer/model metadata used for compatibility checks

The caller owns:

- file I/O and codec decoding
- resampling and channel handling
- VAD and diarization
- glossary/domain corpus construction
- KenLM training and artifact packaging
- post-correction and evaluation reports

The stable boundary is:

```python
audio: np.ndarray  # mono float waveform, preferably float32
sr: int            # preferably 16000
```

The wrapper should not silently grow into a pipeline framework.

## 3. Current Public API

```python
import stt_wrapper

engine = stt_wrapper.load(
    "path/to/ct2-whisper-model",
    backend="ct2",
    device="cuda",
    compute_type="float16",
)

result = engine.transcribe(audio, sr, return_segments=True)
print(result.text)
```

Implemented today:

- `load(model_path, ..., backend="ct2")`
- `Engine.transcribe(audio, sr, options=None, **overrides)`
- `DecodeOptions` keyword overrides
- `TranscriptionResult.text`
- `TranscriptionResult.segments` when `return_segments=True`
- backend registry via `register_backend(...)`
- KenLM metadata sidecar loading and verification

Reserved but not fully implemented:

- `return_scores`
- `return_nbest`
- `fusion_mode`

These names may stay in the config surface, but public docs should not promise
populated scores/N-best output until the implementation fills them.

## 4. Backend And Fusion Status

The default backend is `ct2`, implemented with CTranslate2 Whisper.

Plain STT should work with a normal CTranslate2 installation. KenLM BPE shallow
fusion requires a patched CTranslate2 build that exposes these Python generate
kwargs:

- `lm_fusion_model_path`
- `lm_fusion_alpha`
- `lm_fusion_asr_topk`
- `lm_fusion_debug`

`lm_fusion_beta` is not supported by the current binding and must not be passed.

### Patched CTranslate2 Install

The patched CTranslate2 (KenLM BPE fusion) lives at the fork:

- `https://github.com/YuBeomGon/CTranslate2` — branch `feature/kenlm-bpe-fusion`

It is not on PyPI; it is built from source with KenLM enabled (`WITH_KENLM=ON`),
then the Python binding is installed (`pip install ./python`). This is the
canonical install path for fusion. `scripts/ct2_env.sh` remains a convenience for
pointing Python at an already-built local checkout, not the documented install.

## 5. KenLM Artifact Contract

The wrapper receives an already-built KenLM `.binary`. It does not build KenLM
models.

Expected sidecar:

```json
{
  "model_name": "large-v3-turbo",
  "tokenizer_hash": "...",
  "ct2_model_hash": "...",
  "kenlm_order": 5,
  "asr_topk": 50,
  "fusion_mode": "topk",
  "corpus_version": "domain_20260607",
  "alpha_default": 0.2,
  "topk_default": 50
}
```

Current behavior:

- `<lm>.binary.meta.json` is loaded next to the KenLM binary.
- `tokenizer_hash` is required in strict mode.
- `ct2_model_hash` is checked when the caller provides a current model hash.
- missing metadata is rejected by default when `lm_path` is provided.

Important gap:

- the wrapper currently verifies against hashes passed by the caller; it does
  not yet compute tokenizer/model hashes by itself. Before OSS release, either
  add hash-generation utilities or document exactly how callers must produce the
  values.

## 6. Open Source Readiness Checklist

Required before GitHub release:

- Add a real `LICENSE` file and align `pyproject.toml` metadata.
- Finalize package name, repository URL, author/maintainer metadata.
- Add CI for lint and unit tests.
- Keep model files, audio files, and KenLM artifacts out of git.
- Remove or clearly mark internal-only experiment logs under `docs/assets/`.
- Replace internal absolute paths and private repo references in public docs.
- Document patched CTranslate2 install from the fork (`YuBeomGon/CTranslate2`,
  branch `feature/kenlm-bpe-fusion`); link the fork's build flags.
- Keep `README.md` focused on install, quick start, API, limitations, and docs map.
- Make examples match the input contract, especially mono/resampling behavior.
- Add friendly error messages for bad language/task tokens and unsupported fusion.
- Add or document tokenizer/model hash generation for KenLM metadata.
- Do not advertise `return_scores`, `return_nbest`, or `fusion_mode` as supported
  behavior until the implementation is complete.

## 7. Documentation Map

- `README.md`: public quick start and install guide.
- `docs/SSOT.md`: canonical project contract and release readiness.
- `docs/design.md`: design details for the wrapper boundary and backend contract.
- `docs/decoding_strategy.md`: internal decoding-strategy research notes.
- `docs/implementation_plan.md`: implementation history and planning notes.
- `docs/principles.md`: public-facing design principles and OSS checklist notes.

