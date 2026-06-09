# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic
versioning.

## [0.1.0] - 2026-06-09

Initial release.

### Added
- `load()` + `transcribe()` thin API over a CTranslate2 Whisper backend.
- Long-form decoding: 30s window loop, RMS silence-aware cut, timestamp-adaptive
  seek, N-best selection gate.
- Optional KenLM BPE shallow fusion via the patched CTranslate2 fork
  (`lm_enabled` / `alpha` / `topk`, per-request override).
- Pluggable backend factory (`create_backend` / `register_backend`); `ct2`
  backend ships, others (TensorRT-LLM / HuggingFace / OpenAI Whisper) can be added
  without engine changes.
- Tokenizer-hash gate: refuses an LM whose artifact metadata does not match the
  loaded model.
- `TranscriptionResult` with `text` plus opt-in `segments`.
