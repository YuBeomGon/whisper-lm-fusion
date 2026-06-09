"""CTranslate2 Whisper backend (supports patched KenLM BPE fusion).

All CT2 / transformers imports are inside methods so the package imports cleanly
without a working CTranslate2; only constructing this backend requires it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from whisper_lm_fusion.backends.base import Backend, RawResult
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
        kwargs = {
            "beam_size": options.beam_size,
            "num_hypotheses": options.num_hypotheses,
            "patience": options.patience,
            "length_penalty": options.length_penalty,
            "repetition_penalty": options.repetition_penalty,
            "no_repeat_ngram_size": options.no_repeat_ngram_size,
            "max_length": options.max_length,
            "sampling_topk": options.sampling_topk,
            "sampling_temperature": options.sampling_temperature,
            "suppress_blank": options.suppress_blank,
            "suppress_tokens": list(options.suppress_tokens),
            "max_initial_timestamp_index": options.max_initial_timestamp_index,
            "return_scores": True,
            "return_no_speech_prob": True,
            **fusion.to_generate_kwargs(),  # CT2-specific lm_fusion_* mapping
        }
        result = self._model.generate(features, [prompt], **kwargs)[0]
        return RawResult(
            sequences_ids=[list(s) for s in result.sequences_ids],
            scores=[float(s) for s in result.scores],
            no_speech_prob=float(getattr(result, "no_speech_prob", 0.0) or 0.0),
        )
