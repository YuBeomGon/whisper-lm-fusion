# Self-Evolve STT Pipeline Control Surface

이 문서는 `whisper-lm-fusion` wrapper가 노출하는 sweep 가능한 파이프라인 노브를 정리한다.
목표는 faster-whisper 옵션을 그대로 복제하는 것이 아니라, phase3 self-evolve에서 발견된
알고리즘을 제품형 파이프라인의 작은 control surface로 압축하는 것이다.

## 1. Core backbone

| 축 | 옵션 | 설명 |
|---|---|---|
| long-form | `window_seconds`, `min_advance_seconds`, `silence_percentile`, `timestamp_resolution` | 30초 window, RMS trough cut, timestamp seek |
| search | `beam_size`, `num_hypotheses`, `patience` | beam/N-best backbone |
| gates | `logprob_threshold`, `compression_ratio_threshold`, `no_speech_threshold` | low-confidence, 반복, no-speech guard |
| context | `context_policy`, `max_context_tokens` | rolling context는 기본 confidence-gated |

## 2. Self-evolve derived algorithms

| 알고리즘 | 옵션 | 기본값 | sweep 예 |
|---|---|---:|---|
| Axis-aware N-best selector | `selection_policy="axis_aware"` | on | `axis_aware`, `logprob` |
| Longer-within-margin selector | `prefer_longer_within_margin`, `score_margin`, `min_length_ratio_for_longer` | off | deletion 보정 후보 |
| Token MBR selector | `selection_policy="token_mbr"` | off | N-best medoid 선택 |
| Conditional temperature fallback | `fallback_policy`, `temperature_fallback`, `fallback_sampling_topk` | off | `gate_fail` + `(0.2,0.4,0.6)` |
| Per-window language override | `language_policy`, `language_override_prob` | fixed | `per_window_confident`, prob `0.7` |
| Ambiguous dual-language branch | `language_policy="dual_band"`, `dual_language_low_prob`, `dual_language_high_prob` | off | `[0.4,0.7)` dual decode |
| CJK/Kana script suppress | `suppress_cjk_kana` | true | usually fixed true |
| Align tail trim hook | `align_tail_trim`, `align_prob_floor`, `align_min_run`, trigger flags | off | backend 지원 시만 동작 |
| Repetition guard | `repetition_penalty`, `no_repeat_ngram_size` | off | loop 완화 |

## 3. Suggested sweep groups

### A. Search / gate baseline

```python
DecodeOptions(
    beam_size=5,
    num_hypotheses=5,
    patience=2.0,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
    context_policy="confidence_gated",
)
```

### B. Self-evolve strong options

```python
DecodeOptions(
    selection_policy="axis_aware",
    prefer_longer_within_margin=True,
    score_margin=0.10,
    fallback_policy="gate_fail",
    temperature_fallback=(0.2, 0.4, 0.6),
    language_policy="per_window_confident",
    language_override_prob=0.7,
    suppress_cjk_kana=True,
)
```

### C. Experimental / costly options

```python
DecodeOptions(
    selection_policy="token_mbr",
    language_policy="dual_band",
    dual_language_low_prob=0.4,
    dual_language_high_prob=0.7,
    align_tail_trim=True,
    align_prob_floor=0.3,
    align_min_run=8,
)
```

## 4. Explicitly excluded from generic wrapper

아래는 wrapper 내부 기본 알고리즘이 아니라 caller/pipeline 또는 별도 stage가 소유한다.

- Jamo correction / recurrence canonicalization
- Domain glossary content / term bank content
- KenLM corpus build and alpha/topk sweep
- Dataset evaluation / CER reporting
- VTLN / ABE / dual-front-end audio witness
- ROVER / multi-run ensemble runner

wrapper는 메커니즘과 파라미터만 제공한다. 무엇을 sweep할지, 어떤 데이터를 평가할지는 외부 sweep runner가 결정한다.
