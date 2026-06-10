"""Configuration schemas for whisper-lm-fusion.

Two scopes, deliberately separated (see docs/implementation_plan.md §4):

- ``LoadConfig``   : init-time, set once in ``load()`` (model, device, fusion defaults).
- ``DecodeOptions``: request-time, passed to ``transcribe()`` (search, gates, segmentation).

The public surface is intentionally *not* a clone of faster-whisper.  The core
knobs expose the self-evolve decoding strategies discovered in the phase3 runs:
N-best selection, conditional fallback, language branching, script masking,
confidence-gated context, and optional alignment hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LanguagePolicy = Literal["fixed", "per_window_confident", "dual_band"]
FallbackPolicy = Literal["off", "gate_fail", "low_logprob", "degenerate", "always"]
SelectionPolicy = Literal[
    "axis_aware",
    "logprob",
    "longer_within_margin",
    "token_mbr",
]
ContextPolicy = Literal["off", "always", "confidence_gated"]


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
    """Request-time decoding options.

    Defaults are conservative and usable as a plain STT wrapper.  The extra knobs
    are the compressed control surface of the self-evolve strategy taxonomy, so a
    sweep runner can turn them on/off without editing engine code.
    """

    # language / task
    language: str = "ko"  # base prompt token: <|ko|>
    task: str = "transcribe"

    # search / N-best
    beam_size: int = 5
    num_hypotheses: int = 5
    patience: float = 2.0
    sampling_temperature: float = 0.0
    sampling_topk: int = 1  # >1 enables top-k sampling (with sampling_temperature)
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0  # >1 penalizes repeats (anti-loop/hallucination)
    no_repeat_ngram_size: int = 0  # 0 = disabled
    max_length: int = 448  # max generated tokens per window

    # hypothesis selection; maps multiple phase3 ideas into one sweepable surface
    selection_policy: SelectionPolicy = "axis_aware"
    prefer_longer_within_margin: bool = False
    score_margin: float = 0.10
    min_length_ratio_for_longer: float = 1.05

    # acceptance / failure gates
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    no_speech_logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

    # conditional fallback: retry only failed windows, not every window
    fallback_policy: FallbackPolicy = "off"
    temperature_fallback: tuple[float, ...] = ()
    fallback_sampling_topk: int = 0  # 0 => use sampling_topk

    # language branch / prompt policy from phase3_013 family
    language_policy: LanguagePolicy = "fixed"
    language_override_prob: float = 0.7
    dual_language_low_prob: float = 0.4
    dual_language_high_prob: float = 0.7

    # token constraints and script mask
    suppress_blank: bool = True
    suppress_tokens: tuple[int, ...] = (-1,)  # -1 = model config's default symbol set
    suppress_cjk_kana: bool = True
    max_initial_timestamp_index: int = 50

    # segmentation / seek backbone
    window_seconds: float = 30.0
    timestamp_resolution: float = 0.02
    min_advance_seconds: float = 20.0
    silence_percentile: float = 20.0

    # context carry policy; confidence_gated is the phase3 lesson, not always-on rolling text
    max_context_tokens: int = 200
    context_policy: ContextPolicy = "confidence_gated"

    # alignment hook.  Backends that cannot provide align posterior leave this as no-op.
    align_tail_trim: bool = False
    align_prob_floor: float = 0.3
    align_min_run: int = 8
    align_trigger_low_logprob: bool = True
    align_trigger_degenerate: bool = True

    # fusion overrides (None => fall back to engine load-time default)
    lm_enabled: bool = False
    alpha: float | None = None
    topk: int | None = None
    fusion_debug: bool = False

    # output opt-in
    return_segments: bool = False
    return_scores: bool = False
    return_nbest: bool = False
