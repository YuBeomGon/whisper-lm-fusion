"""CTranslate2 Whisper backend (supports patched KenLM BPE fusion)."""

from __future__ import annotations

from typing import Any

import numpy as np

from whisper_lm_fusion.backends.base import Backend, LanguageProb, RawResult
from whisper_lm_fusion.config import DecodeOptions, FusionOptions, LoadConfig


class Ct2Backend(Backend):
    """Runs an already-CT2-converted Whisper model via ``ctranslate2``."""

    supports_fusion = True

    def __init__(self, config: LoadConfig) -> None:
        import ctranslate2
        from transformers import WhisperProcessor

        processor_src = config.processor_path or config.model_path
        self._processor = WhisperProcessor.from_pretrained(
            str(processor_src),
            local_files_only=config.processor_local_files_only,
        )
        self._model = ctranslate2.models.Whisper(
            str(config.model_path),
            device=config.device,
            compute_type=config.compute_type,
        )

    @property
    def tokenizer(self) -> Any:
        return self._processor.tokenizer

    def extract_features(self, chunk: np.ndarray, sr: int) -> Any:
        import ctranslate2

        inputs = self._processor(chunk, sampling_rate=sr, return_tensors="np")
        return ctranslate2.StorageView.from_array(inputs.input_features)

    def generate(
        self,
        features: Any,
        prompt: list[int],
        *,
        options: DecodeOptions,
        fusion: FusionOptions,
    ) -> RawResult:
        suppress_tokens = list(options.suppress_tokens)
        kwargs = {
            "beam_size": options.beam_size,
            "num_hypotheses": min(options.num_hypotheses, max(options.beam_size, 1)),
            "patience": options.patience,
            "length_penalty": options.length_penalty,
            "repetition_penalty": options.repetition_penalty,
            "no_repeat_ngram_size": options.no_repeat_ngram_size,
            "max_length": options.max_length,
            "sampling_topk": options.sampling_topk,
            "sampling_temperature": options.sampling_temperature,
            "suppress_blank": options.suppress_blank,
            "suppress_tokens": suppress_tokens,
            "max_initial_timestamp_index": options.max_initial_timestamp_index,
            "return_scores": True,
            "return_no_speech_prob": True,
            **fusion.to_generate_kwargs(),
        }
        result = self._model.generate(features, [prompt], **kwargs)[0]
        return RawResult(
            sequences_ids=[list(s) for s in result.sequences_ids],
            scores=[float(s) for s in result.scores],
            no_speech_prob=float(getattr(result, "no_speech_prob", 0.0) or 0.0),
        )

    def detect_language(self, features: Any) -> list[LanguageProb]:
        detector = getattr(self._model, "detect_language", None)
        if detector is None:
            return []
        try:
            raw = detector(features)
        except TypeError:
            return []
        if not raw:
            return []
        # CT2 commonly returns: [[("<|ko|>", 0.99), ...]] for batch size 1.
        first = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        out: list[LanguageProb] = []
        for item in first:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            token, prob = item[0], item[1]
            if isinstance(token, int):
                token = self.tokenizer.convert_ids_to_tokens(token)
            lang = str(token).replace("<|", "").replace("|>", "")
            try:
                out.append(LanguageProb(language=lang, probability=float(prob)))
            except (TypeError, ValueError):
                continue
        return out

    def align_tail_trim(
        self,
        features: Any,
        token_ids: list[int],
        *,
        options: DecodeOptions,
    ) -> list[int] | None:
        """Best-effort CT2 alignment hook.

        CT2 alignment APIs differ by version.  Rather than guess incorrectly and
        break decoding, this method only activates when a compatible low-level
        method is present and returns a recognized posterior structure.  In most
        environments it will safely no-op; the algorithm surface remains exposed
        for runners/backends that implement alignment.
        """
        align = getattr(self._model, "align", None)
        if align is None or not token_ids:
            return None
        try:
            raw = align(features, [token_ids])
        except TypeError:
            return None
        except Exception:
            return None
        # Expected-ish forms are backend/version dependent.  Support a simple
        # trailing probability list if provided; otherwise leave unchanged.
        probs = getattr(raw, "probs", None) or getattr(raw, "probabilities", None)
        if isinstance(raw, list) and raw:
            probs = getattr(raw[0], "probs", None) or getattr(raw[0], "probabilities", None)
        if not probs:
            return None
        try:
            prob_list = [float(p) for p in probs]
        except (TypeError, ValueError):
            return None
        trim_at = _low_prob_tail_start(prob_list, options.align_prob_floor, options.align_min_run)
        if trim_at is None or trim_at <= 0 or trim_at >= len(token_ids):
            return None
        return token_ids[:trim_at]


def _low_prob_tail_start(probs: list[float], floor: float, min_run: int) -> int | None:
    run = 0
    start: int | None = None
    for i in range(len(probs) - 1, -1, -1):
        if probs[i] < floor:
            run += 1
            start = i
        else:
            break
    if run >= min_run:
        return start
    return None
