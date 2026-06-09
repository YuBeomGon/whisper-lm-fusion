"""Engine: ``load()`` once, ``transcribe()`` per request.

The long-form decoding policy lives here (window loop + the four base-logic
units in ``whisper_lm_fusion._decode``) and is backend-agnostic. Model execution is
delegated to a :class:`~whisper_lm_fusion.backends.base.Backend` chosen by the factory,
so CTranslate2, TensorRT-LLM, HuggingFace/OpenAI Whisper, etc. are swappable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from whisper_lm_fusion import _decode
from whisper_lm_fusion.backends import Backend, create_backend
from whisper_lm_fusion.config import DecodeOptions, FusionOptions, LoadConfig
from whisper_lm_fusion.metadata import load_metadata, verify_metadata
from whisper_lm_fusion.results import Segment, TranscriptionResult


class Engine:
    """A loaded STT engine. Create via :func:`load`, not directly."""

    def __init__(self, backend: Backend, config: LoadConfig) -> None:
        self._backend = backend
        self._tokenizer = backend.tokenizer
        self._config = config

        # model-level special tokens (language/task are resolved per request)
        tok = self._tokenizer
        self._timestamp_begin = tok.convert_tokens_to_ids("<|0.00|>")
        self._notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")
        self._startofprev = tok.convert_tokens_to_ids("<|startofprev|>")
        self._eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self._sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    # -- fusion resolution --------------------------------------------------

    def _resolve_fusion(self, options: DecodeOptions) -> FusionOptions:
        if not options.lm_enabled or self._config.lm_path is None:
            return FusionOptions(enabled=False)
        if not self._backend.supports_fusion:
            raise ValueError(
                f"backend {self._config.backend!r} does not support KenLM fusion"
            )
        alpha = options.alpha if options.alpha is not None else self._config.alpha_default
        topk = options.topk if options.topk is not None else self._config.topk_default
        return FusionOptions(
            enabled=True,
            model_path=self._config.lm_path,
            alpha=alpha,
            asr_topk=topk,
            debug=options.fusion_debug,
        )

    # -- prompt construction ------------------------------------------------

    def _build_prompt(
        self, options: DecodeOptions, context_ids: list[int], use_timestamps: bool
    ) -> list[int]:
        tok = self._tokenizer
        sot = [
            self._sot,
            tok.convert_tokens_to_ids(f"<|{options.language}|>"),
            tok.convert_tokens_to_ids(f"<|{options.task}|>"),
        ]
        if not use_timestamps:
            sot = [*sot, self._notimestamps]
        if context_ids:
            return [self._startofprev, *context_ids[-options.max_context_tokens :], *sot]
        return list(sot)

    # -- main loop ----------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        options: DecodeOptions | None = None,
        **overrides: Any,
    ) -> TranscriptionResult:
        """Transcribe ``audio`` (mono float waveform) at sample rate ``sr``.

        Pass a :class:`DecodeOptions` or keyword overrides, e.g.
        ``engine.transcribe(audio, sr, beam_size=8, lm_enabled=True, alpha=0.2)``.
        """
        options = _merge_options(options, overrides)
        fusion = self._resolve_fusion(options)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="")

        window = max(int(options.window_seconds * sr), 1)
        n = len(audio)
        energy = _decode.frame_energy(audio, sr, options)

        texts: list[str] = []
        segments: list[Segment] = []
        context_ids: list[int] = []
        seek = 0

        while seek < n:
            cut, silence_cut = _decode.decide_cut(seek, window, n, energy, sr, options)
            chunk = audio[seek:cut]

            features = self._backend.extract_features(chunk, sr)
            use_timestamps = (not silence_cut) and (cut < n)
            prompt = self._build_prompt(options, context_ids, use_timestamps)

            result = self._backend.generate(features, prompt, options=options, fusion=fusion)
            hyp = _decode.select_hypothesis(
                result.scores,
                result.sequences_ids,
                lambda ids: self._tokenizer.decode(ids, skip_special_tokens=True),
                options,
                no_speech_prob=result.no_speech_prob,
            )

            if hyp.text:
                texts.append(hyp.text)
                if options.return_segments:
                    segments.append(
                        Segment(
                            text=hyp.text,
                            start=seek / sr,
                            end=cut / sr,
                            logprob=hyp.logprob,
                            no_speech_prob=hyp.no_speech_prob,
                        )
                    )

            # confidence-gated context carry
            if hyp.logprob >= options.logprob_threshold and not hyp.degenerate:
                context_ids = [t for t in hyp.token_ids if t < self._eot]
            elif hyp.no_speech_prob < options.no_speech_threshold:
                context_ids = []

            seek = _decode.advance_seek(
                seek, cut, len(chunk), window, hyp.token_ids,
                self._timestamp_begin, sr, options, silence_cut,
            )

        return TranscriptionResult(
            text=" ".join(texts),
            segments=segments if options.return_segments else None,
        )


def _merge_options(options: DecodeOptions | None, overrides: dict[str, Any]) -> DecodeOptions:
    base = options or DecodeOptions()
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)


def load(
    model_path: str | Path,
    lm_path: str | Path | None = None,
    *,
    backend: str = "ct2",
    device: str = "cuda",
    compute_type: str = "float16",
    alpha_default: float = 0.0,
    topk_default: int = 50,
    fusion_mode: str = "topk",
    processor_path: str | Path | None = None,
    verify_lm_metadata: bool = True,
    tokenizer_hash: str | None = None,
    ct2_model_hash: str | None = None,
) -> Engine:
    """Boot an :class:`Engine`.

    ``model_path`` is an already-converted Whisper model for the chosen
    ``backend``. ``lm_path`` is a pipeline-built KenLM ``.binary``; when given,
    its sidecar metadata is checked against ``tokenizer_hash`` / ``ct2_model_hash``
    and the LM is refused on mismatch (design.md §3). Pass ``lm_path=None`` for
    plain STT.
    """
    config = LoadConfig(
        model_path=Path(model_path),
        backend=backend,
        lm_path=Path(lm_path) if lm_path is not None else None,
        device=device,
        compute_type=compute_type,
        alpha_default=alpha_default,
        topk_default=topk_default,
        fusion_mode=fusion_mode,
        processor_path=processor_path,
        verify_metadata=verify_lm_metadata,
    )

    if config.lm_path is not None and config.verify_metadata:
        metadata = load_metadata(config.lm_path)
        verify_metadata(
            metadata,
            tokenizer_hash=tokenizer_hash,
            ct2_model_hash=ct2_model_hash,
            strict=True,
        )

    backend_impl = create_backend(config.backend, config)
    return Engine(backend_impl, config)
