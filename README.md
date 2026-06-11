# whisper-lm-fusion

A generic, faster-whisper-style **STT wrapper** over a (patched) CTranslate2 Whisper
backend with **optional KenLM BPE shallow fusion**.

It hides the long-form decoding pipeline (window loop, silence-aware cut,
timestamp-adaptive seek, N-best selection) behind two calls — `load()` and
`transcribe()` — and exposes the knobs as parameters.

> **Generic by design.** The wrapper only knows _audio + parameters -> text_.
> Domain logic (glossary, corpus, KenLM building, post-correction rules) lives in
> your pipeline, never here. See [`docs/SSOT.md`](docs/SSOT.md).

## Why

Tuned for three things generic STT handles poorly:

- **Domain adaptation** — when the target vocabulary is known, plug in a
  pipeline-built KenLM (`.binary`) for shallow fusion. No retraining; toggle it
  per request with `alpha` / `lm_enabled`.
- **Fast / rapid speech** — the long-form decoding policy (silence-aware cut,
  adaptive seek, N-best gating) is tuned to resist deletion and degeneracy on
  quick, dense speech.
- **Pluggable backends** — model execution is chosen by a factory, so the same
  decoding policy runs over CTranslate2 today and TensorRT-LLM / HuggingFace /
  OpenAI Whisper later (see [Backends](#backends)).

Everything is exposed as parameters; the domain content itself stays in your
pipeline.

## Install

Not published on PyPI — install from source (the wrapper is pure Python, no
compilation):

```bash
git clone https://github.com/YuBeomGon/whisper-lm-fusion.git
cd whisper-lm-fusion
pip install -e .            # core (backend-agnostic)
pip install -e ".[ct2]"     # + CTranslate2 backend (plain STT, PyPI ctranslate2)
```

`[ct2]` pulls the **stock PyPI `ctranslate2`** (no source build) and is all you
need for plain STT plus the full decoding control surface. Only KenLM fusion
needs a source-built fork — see below.

### KenLM fusion (patched CTranslate2)

Fusion needs a CTranslate2 build with the KenLM BPE patch, which exposes the
`lm_fusion_*` generate kwargs. It is **not** on PyPI — build it from the fork
[YuBeomGon/CTranslate2 @ `feature/kenlm-bpe-fusion`](https://github.com/YuBeomGon/CTranslate2/tree/feature/kenlm-bpe-fusion):

```bash
git clone -b feature/kenlm-bpe-fusion \
  https://github.com/YuBeomGon/CTranslate2.git
cd CTranslate2
# build the C++ lib with KenLM enabled (WITH_KENLM=ON), then the Python binding
# full build flags: see the fork README
pip install ./python
```

Build with `WITH_KENLM=ON` (one build supports **both** baseline and fusion: with
no `lm_fusion_*` args it decodes exactly like baseline). Without this patched
build the wrapper still runs as a plain Whisper STT (`lm_enabled=False`).

> **Do not install both.** The PyPI `ctranslate2` and the source-built patched
> `libctranslate2` are ABI-incompatible. For fusion, skip `[ct2]` and use the
> fork only; `scripts/ct2_env.sh` points Python at an already-built local
> checkout without installing.

## Backends

The model execution layer is pluggable via a factory — the long-form decoding
policy is backend-agnostic:

```python
engine = whisper_lm_fusion.load("path/to/model", backend="ct2")  # default
whisper_lm_fusion.available_backends()                           # -> ['ct2']
```

`ct2` (CTranslate2, with optional KenLM fusion) ships today. TensorRT-LLM /
HuggingFace Whisper / OpenAI Whisper backends can be added by subclassing
`Backend` and calling `register_backend("name", "module:Class")` — no engine
changes needed.

## Quick start

```python
import soundfile as sf
import whisper_lm_fusion

engine = whisper_lm_fusion.load("path/to/ct2-whisper-model")   # boot once
audio, sr = sf.read("speech.wav")                        # mono float waveform
text = engine.transcribe(audio, sr).text
print(text)
```

### With KenLM fusion (optional)

```python
engine = whisper_lm_fusion.load(
    "path/to/ct2-whisper-model",
    lm_path="path/to/domain.binary",   # pipeline-built KenLM artifact
    tokenizer_hash="<hash>",           # verified against artifact metadata
)
text = engine.transcribe(audio, sr, lm_enabled=True, alpha=0.2).text
```

`alpha=0` or `lm_enabled=False` turns fusion off and decodes identically to the
baseline backend.

### Input contract

`transcribe()` takes an **already-decoded waveform** — `audio: np.ndarray`
(float32, mono, ideally 16 kHz) + `sr`. File I/O, codec decoding, resampling, and
l/r channel split are the **caller's** responsibility (the pipeline owns them), so
the library stays generic and free of heavy audio-decoding deps. See
[`docs/design.md` §5.1](docs/design.md).

## Parameters

`transcribe()` accepts a `DecodeOptions` or keyword overrides:

| group | params (defaults) |
|---|---|
| search | `beam_size=5`, `num_hypotheses=5`, `patience=2.0`, `sampling_temperature=0.0`, `sampling_topk=1`, `length_penalty=1.0`, `repetition_penalty=1.0`, `no_repeat_ngram_size=0`, `max_length=448` |
| gates | `logprob_threshold=-1.0`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4` |
| constraints | `suppress_blank=True`, `suppress_tokens=(-1,)`, `max_initial_timestamp_index=50` |
| segmentation | `window_seconds=30.0`, `timestamp_resolution=0.02`, `min_advance_seconds=20.0`, `silence_percentile=20.0`, `max_context_tokens=200` |
| language | `language="ko"`, `task="transcribe"` |
| policy | `selection_policy="axis_aware"`, `fallback_policy="off"`, `language_policy="fixed"`, `context_policy="confidence_gated"` — the self-evolve control surface; each switches internal logic. Full values/sub-knobs in [`pipeline_control_surface.md`](docs/research/pipeline_control_surface.md) |
| fusion | `lm_enabled=False`, `alpha=None`, `topk=None` (None → engine defaults), `fusion_debug=False` |
| output | `return_segments=False` |

Defaults follow the verified backbone in
[`docs/research/decoding_strategy.md`](docs/research/decoding_strategy.md). The
`search` and `constraints` knobs map 1:1 to CTranslate2's `Whisper.generate`; other
backends map or ignore them. `sampling_temperature` is a single deterministic
value; a faster-whisper-style temperature fallback ladder is implemented via
`fallback_policy` + `temperature_fallback`, which re-decodes only gate-failing
windows (see the `policy` row and
[`docs/research/pipeline_control_surface.md`](docs/research/pipeline_control_surface.md)).

`return_scores`, `return_nbest`, and `fusion_mode` exist in the internal config
surface but are not yet documented as completed public behavior. See
[`docs/SSOT.md`](docs/SSOT.md).

## Scope

**v1 (thin):** `load()` + `transcribe()`, long-form decoding, tokenizer-hash gate.

**Out of scope (later):** batching, VAD, segment merge, word timestamps,
diarization, and the advanced strategies catalogued in `docs/research/decoding_strategy.md`.

## Docs

**User / operations** (`docs/guide/`):

- [`docs/guide/pipeline.md`](docs/guide/pipeline.md) — how `transcribe()` decodes, step by step
- [`docs/guide/parameters.md`](docs/guide/parameters.md) — full `load()` / `DecodeOptions` reference
- [`docs/guide/fusion.md`](docs/guide/fusion.md) — KenLM LM fusion operations guide
- runnable [`examples/`](examples/): `plain_stt.py`, `with_fusion.py`, `tune_policies.py`, `segments.py`

**Contract / research**:

- [`docs/SSOT.md`](docs/SSOT.md) — canonical project contract & OSS readiness
- [`docs/design.md`](docs/design.md) — interface & responsibility boundary
- [`docs/research/pipeline_control_surface.md`](docs/research/pipeline_control_surface.md) — sweepable control surface
- [`docs/research/decoding_strategy.md`](docs/research/decoding_strategy.md) — internal strategy catalog
- [`docs/dev/implementation_plan.md`](docs/dev/implementation_plan.md) — build plan · [`docs/archive/principles.md`](docs/archive/principles.md) — design principles

## License

MIT — see [`LICENSE`](LICENSE).

> This wrapper only routes a KenLM `.binary` path to the backend; it does not link
> or bundle KenLM. KenLM's own license applies to the (separate) patched
> CTranslate2 build that links it, not to this repo.
