"""Engine: ``load()`` once, ``transcribe()`` per request.

The engine owns the self-evolved long-form pipeline: windowing, RMS silence cut,
timestamp seek, language branch, conditional fallback, N-best selection, optional
script suppression, optional align hook, and confidence-gated context carry.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from whisper_lm_fusion import _decode
from whisper_lm_fusion.backends import Backend, create_backend
from whisper_lm_fusion.backends.base import LanguageProb, RawResult
from whisper_lm_fusion.config import DecodeOptions, FusionOptions, LoadConfig
from whisper_lm_fusion.metadata import load_metadata, verify_metadata
from whisper_lm_fusion.results import Segment, TranscriptionResult


class Engine:
    """A loaded STT engine. Create via :func:`load`, not directly."""

    def __init__(self, backend: Backend, config: LoadConfig) -> None:
        self._backend = backend
        self._tokenizer = backend.tokenizer
        self._config = config
        self._cjk_kana_suppress_cache: tuple[int, ...] | None = None

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

    # -- prompt / token constraints ----------------------------------------

    def _build_prompt(
        self,
        options: DecodeOptions,
        context_ids: list[int],
        use_timestamps: bool,
        *,
        language: str | None = None,
    ) -> list[int]:
        tok = self._tokenizer
        lang = language or options.language
        sot = [
            self._sot,
            tok.convert_tokens_to_ids(f"<|{lang}|>"),
            tok.convert_tokens_to_ids(f"<|{options.task}|>"),
        ]
        if not use_timestamps:
            sot = [*sot, self._notimestamps]
        if context_ids and options.context_policy != "off" and options.max_context_tokens > 0:
            return [self._startofprev, *context_ids[-options.max_context_tokens :], *sot]
        return list(sot)

    def _with_runtime_suppress_tokens(self, options: DecodeOptions) -> DecodeOptions:
        if not options.suppress_cjk_kana:
            return options
        extra = self._cjk_kana_suppress_tokens()
        merged = tuple(dict.fromkeys([*options.suppress_tokens, *extra]))
        if merged == options.suppress_tokens:
            return options
        return replace(options, suppress_tokens=merged)

    def _cjk_kana_suppress_tokens(self) -> tuple[int, ...]:
        if self._cjk_kana_suppress_cache is not None:
            return self._cjk_kana_suppress_cache
        tok = self._tokenizer
        vocab = getattr(tok, "get_vocab", lambda: {})()
        ids: list[int] = []
        for token_id in vocab.values():
            if not isinstance(token_id, int) or token_id < 0:
                continue
            try:
                text = tok.decode([token_id], skip_special_tokens=True)
            except Exception:
                continue
            if _contains_cjk_kana(text):
                ids.append(token_id)
        self._cjk_kana_suppress_cache = tuple(sorted(set(ids)))
        return self._cjk_kana_suppress_cache

    # -- language policy ----------------------------------------------------

    def _language_decision(
        self,
        features: Any,
        options: DecodeOptions,
    ) -> tuple[str, str | None, list[LanguageProb]]:
        """Return ``(primary_language, secondary_language, probs)``.

        ``secondary_language`` is only used for ``dual_band`` decode.  The base
        language remains the safe fallback if detection is unsupported/uncertain.
        """
        if options.language_policy == "fixed":
            return options.language, None, []
        probs = sorted(
            self._backend.detect_language(features),
            key=lambda p: p.probability,
            reverse=True,
        )
        if not probs:
            return options.language, None, []
        top = probs[0]
        if top.language == options.language:
            return options.language, None, probs
        if options.language_policy == "per_window_confident":
            if top.probability >= options.language_override_prob:
                return top.language, None, probs
            return options.language, None, probs
        if (
            options.dual_language_low_prob
            <= top.probability
            < options.dual_language_high_prob
        ):
            return options.language, top.language, probs
        if top.probability >= options.language_override_prob:
            return top.language, None, probs
        return options.language, None, probs

    # -- one-window decoding ------------------------------------------------

    def _decode_once(
        self,
        features: Any,
        prompt: list[int],
        options: DecodeOptions,
        fusion: FusionOptions,
        *,
        source: str,
        language: str,
        temperature: float | None = None,
    ) -> tuple[RawResult, list[_decode.Candidate]]:
        result = self._backend.generate(features, prompt, options=options, fusion=fusion)
        candidates = _decode.make_candidates(
            result.scores,
            result.sequences_ids,
            lambda ids: self._tokenizer.decode(ids, skip_special_tokens=True),
            options,
            no_speech_prob=result.no_speech_prob,
            source=source,
            language=language,
            temperature=temperature,
        )
        return result, candidates

    def _decode_window(
        self,
        features: Any,
        context_ids: list[int],
        use_timestamps: bool,
        options: DecodeOptions,
        fusion: FusionOptions,
    ) -> _decode.Hypothesis:
        primary_lang, secondary_lang, _lang_probs = self._language_decision(features, options)
        all_candidates: list[_decode.Candidate] = []

        for idx, lang in enumerate([primary_lang, secondary_lang]):
            if lang is None:
                continue
            prompt = self._build_prompt(options, context_ids, use_timestamps, language=lang)
            _result, candidates = self._decode_once(
                features,
                prompt,
                options,
                fusion,
                source="base" if idx == 0 else "dual_language",
                language=lang,
                temperature=options.sampling_temperature,
            )
            all_candidates.extend(candidates)

        hyp = _decode.hypothesis_from_candidates(all_candidates, options)
        if _decode.should_fallback(hyp, options):
            fallback_candidates: list[_decode.Candidate] = []
            for temp in options.temperature_fallback:
                if temp == options.sampling_temperature:
                    continue
                topk = options.fallback_sampling_topk or options.sampling_topk
                fallback_options = replace(
                    options,
                    sampling_temperature=temp,
                    sampling_topk=topk,
                )
                prompt = self._build_prompt(
                    fallback_options,
                    context_ids,
                    use_timestamps,
                    language=primary_lang,
                )
                _result, candidates = self._decode_once(
                    features,
                    prompt,
                    fallback_options,
                    fusion,
                    source="temperature_fallback",
                    language=primary_lang,
                    temperature=temp,
                )
                fallback_candidates.extend(candidates)
            if fallback_candidates:
                hyp = _decode.hypothesis_from_candidates(
                    [*all_candidates, *fallback_candidates],
                    options,
                )

        return self._maybe_align_trim(features, hyp, options)

    def _maybe_align_trim(
        self,
        features: Any,
        hyp: _decode.Hypothesis,
        options: DecodeOptions,
    ) -> _decode.Hypothesis:
        if not options.align_tail_trim or not hyp.token_ids or hyp.dropped_no_speech:
            return hyp
        trigger = False
        if options.align_trigger_degenerate and hyp.degenerate:
            trigger = True
        if options.align_trigger_low_logprob and hyp.logprob < options.logprob_threshold:
            trigger = True
        if not trigger:
            return hyp
        trimmed = self._backend.align_tail_trim(features, hyp.token_ids, options=options)
        if not trimmed or trimmed == hyp.token_ids:
            return hyp
        text = self._tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        if not text:
            return hyp
        cr = _decode.compression_ratio(text)
        return replace(
            hyp,
            token_ids=list(trimmed),
            text=text,
            degenerate=cr > options.compression_ratio_threshold,
            selected_source=f"{hyp.selected_source}+align_trim",
        )

    # -- context policy -----------------------------------------------------

    def _update_context(
        self,
        hyp: _decode.Hypothesis,
        options: DecodeOptions,
        current: list[int],
    ) -> list[int]:
        if options.context_policy == "off":
            return []
        if not hyp.token_ids or hyp.dropped_no_speech:
            return current if hyp.no_speech_prob >= options.no_speech_threshold else []
        clean_ids = [t for t in hyp.token_ids if t < self._eot]
        if options.context_policy == "always":
            return clean_ids
        if hyp.logprob >= options.logprob_threshold and not hyp.degenerate:
            return clean_ids
        if hyp.no_speech_prob < options.no_speech_threshold:
            return []
        return current

    # -- main loop ----------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        options: DecodeOptions | None = None,
        **overrides: Any,
    ) -> TranscriptionResult:
        """Transcribe ``audio`` (mono float waveform) at sample rate ``sr``."""
        options = _merge_options(options, overrides)
        options = self._with_runtime_suppress_tokens(options)
        fusion = self._resolve_fusion(options)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="")

        window = max(int(options.window_seconds * sr), 1)
        n = len(audio)
        energy = _decode.frame_energy(audio, sr, options)

        texts: list[str] = []
        segments: list[Segment] = []
        scores: list[float] = []
        nbest: list[list[dict[str, object]]] = []
        context_ids: list[int] = []
        seek = 0

        while seek < n:
            cut, silence_cut = _decode.decide_cut(seek, window, n, energy, sr, options)
            chunk = audio[seek:cut]

            features = self._backend.extract_features(chunk, sr)
            use_timestamps = (not silence_cut) and (cut < n)
            hyp = self._decode_window(features, context_ids, use_timestamps, options, fusion)

            if hyp.text:
                texts.append(hyp.text)
            if options.return_scores:
                scores.append(hyp.logprob)
            if options.return_nbest:
                nbest.append([_candidate_to_dict(c) for c in hyp.candidates])
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

            context_ids = self._update_context(hyp, options, context_ids)

            seek = _decode.advance_seek(
                seek,
                cut,
                len(chunk),
                window,
                hyp.token_ids,
                self._timestamp_begin,
                sr,
                options,
                silence_cut,
            )

        return TranscriptionResult(
            text=" ".join(texts),
            segments=segments if options.return_segments else None,
            scores=scores if options.return_scores else None,
            nbest=nbest if options.return_nbest else None,
        )


def _candidate_to_dict(candidate: _decode.Candidate) -> dict[str, object]:
    return {
        "text": candidate.text,
        "logprob": candidate.logprob,
        "no_speech_prob": candidate.no_speech_prob,
        "compression": candidate.compression,
        "degenerate": candidate.degenerate,
        "source": candidate.source,
        "language": candidate.language,
        "temperature": candidate.temperature,
        "num_tokens": len(candidate.token_ids),
    }


def _contains_cjk_kana(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:  # Hiragana + Katakana
            return True
        if 0x3400 <= cp <= 0x4DBF:  # CJK Ext A
            return True
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
            return True
        if 0xF900 <= cp <= 0xFAFF:  # CJK Compatibility Ideographs
            return True
    return False


def _merge_options(options: DecodeOptions | None, overrides: dict[str, Any]) -> DecodeOptions:
    base = options or DecodeOptions()
    if not overrides:
        return base
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
    """Boot an :class:`Engine`."""
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
