"""Configuration schemas for whisper-lm-fusion.

Two scopes, deliberately separated (see docs/implementation_plan.md §4):

- ``LoadConfig``   : init-time, set once in ``load()`` (model, device, fusion defaults).
- ``DecodeOptions``: request-time, passed to ``transcribe()`` (search, gates, segmentation).

Request-time fusion fields (``alpha``/``topk``/``lm_enabled``) default to ``None`` and
fall back to the engine's load-time defaults when not given, so a caller can toggle fusion
per request without restating every value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadConfig:
    """Init-time configuration. Set once when the engine boots."""

    model_path: Path
    backend: str = "ct2"  # selects the execution provider (see whisper_lm_fusion.backends)
    lm_path: Path | None = None  # patched-CT2 KenLM .binary; None => fusion unavailable
    device: str = "cuda"
    compute_type: str = "float16"

    # processor / tokenizer
    processor_path: str | Path | None = None  # defaults to model_path when None
    processor_local_files_only: bool = True

    # fusion defaults (a request may override these)
    alpha_default: float = 0.0
    topk_default: int = 50
    fusion_mode: str = "topk"

    # metadata gate: refuse LM load on tokenizer/model hash mismatch (design.md §3)
    verify_metadata: bool = True


@dataclass(frozen=True)
class FusionOptions:
    """Resolved per-request fusion settings handed to patched CT2.

    Note: ``lm_fusion_beta`` is intentionally absent — the patched binding does not
    expose it and it must not be passed (design.md §7).
    """

    enabled: bool = False
    model_path: Path | None = None
    alpha: float = 0.0
    asr_topk: int = 50
    debug: bool = False

    def to_generate_kwargs(self) -> dict[str, object]:
        if not self.enabled or self.model_path is None or self.alpha <= 0.0:
            return {}
        return {
            "lm_fusion_model_path": str(self.model_path),
            "lm_fusion_alpha": self.alpha,
            "lm_fusion_asr_topk": self.asr_topk,
            "lm_fusion_debug": self.debug,
        }


@dataclass(frozen=True)
class DecodeOptions:
    """Request-time decoding options. Every field has a sane default."""

    # language / task
    language: str = "ko"  # used to build the <|{language}|> prompt token
    task: str = "transcribe"

    # search (decoding_strategy defaults)
    beam_size: int = 5
    num_hypotheses: int = 5
    patience: float = 2.0
    sampling_temperature: float = 0.0

    # acceptance / fallback gates
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4

    # segmentation / seek
    window_seconds: float = 30.0
    timestamp_resolution: float = 0.02
    min_advance_seconds: float = 20.0
    silence_percentile: float = 20.0
    max_context_tokens: int = 200

    # fusion overrides (None => fall back to engine load-time default)
    lm_enabled: bool = False
    alpha: float | None = None
    topk: int | None = None
    fusion_debug: bool = False

    # output opt-in
    return_segments: bool = False
    return_scores: bool = False
    return_nbest: bool = False
