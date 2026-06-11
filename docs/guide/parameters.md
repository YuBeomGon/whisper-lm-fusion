# Parameters — 전체 레퍼런스

`load()`(init-time)와 `transcribe()` / `DecodeOptions`(request-time)의 모든 노브를 정리한다.
각 파라미터가 파이프라인 어느 단계에 작용하는지는 [`pipeline.md`](pipeline.md) 참고.

> 두 스코프는 의도적으로 분리되어 있다. 모델/디바이스/fusion 기본값은 `load()`에서 한 번,
> 디코딩 동작은 요청마다 `transcribe()`에서 바꾼다.

---

## 1. `load()` — init-time

```python
engine = whisper_lm_fusion.load(model_path, lm_path=None, *, backend="ct2",
    device="cuda", compute_type="float16", alpha_default=0.0, topk_default=50,
    fusion_mode="topk", processor_path=None, verify_lm_metadata=True,
    tokenizer_hash=None, ct2_model_hash=None)
```

| 인자 | 기본 | 설명 |
|---|---|---|
| `model_path` | — | CT2 변환된 Whisper 모델 경로 |
| `lm_path` | `None` | KenLM `.binary` 경로. `None`이면 fusion 비활성 |
| `backend` | `"ct2"` | 실행 백엔드(`available_backends()`) |
| `device` / `compute_type` | `"cuda"` / `"float16"` | CT2 디바이스/정밀도 |
| `alpha_default` / `topk_default` | `0.0` / `50` | fusion의 load-time 기본값(요청에서 override 가능) |
| `fusion_mode` | `"topk"` | fusion 모드 라벨 |
| `processor_path` | `None` | tokenizer/processor 경로(미지정 시 `model_path`) |
| `verify_lm_metadata` | `True` | LM 메타데이터 해시 검증 강제(불일치 시 에러) |
| `tokenizer_hash` / `ct2_model_hash` | `None` | 메타데이터 대조용 현재 모델 해시 |

자세한 fusion/메타데이터는 [`fusion.md`](fusion.md).

---

## 2. `DecodeOptions` — request-time

`transcribe(audio, sr, **overrides)` 또는 `DecodeOptions(...)`로 전달. 기본값은 보수적이며,
전부 off로 두면 안정적 long-form baseline으로 동작한다.

### 2.1 Language / task
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `language` | `"ko"` | SOT 언어 토큰 |
| `task` | `"transcribe"` | `transcribe` / `translate` |

### 2.2 Search / N-best
| 파라미터 | 기본 | 권장범위 | 설명 |
|---|---|---|---|
| `beam_size` | `5` | 1~8 | beam 폭 |
| `num_hypotheses` | `5` | ≤`beam_size` | N-best 수(beam_size로 클램프) |
| `patience` | `2.0` | 1.0~2.5 | beam 후보 유지 |
| `sampling_temperature` | `0.0` | — | 1차 디코드 온도(보통 0) |
| `sampling_topk` | `1` | — | >1이면 top-k 샘플링 |
| `length_penalty` | `1.0` | — | 길이 보정 |
| `repetition_penalty` | `1.0` | 1.0~1.3 | 반복 억제(>1) |
| `no_repeat_ngram_size` | `0` | 0~3 | n-gram 반복 차단(0=off) |
| `max_length` | `448` | — | 윈도우당 최대 토큰 |

### 2.3 Selection (selection_policy)
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `selection_policy` | `"axis_aware"` | `axis_aware` / `logprob` / `longer_within_margin` / `token_mbr` |
| `prefer_longer_within_margin` | `False` | longer-within-margin 강제(정책 무관 적용) |
| `score_margin` | `0.10` | best logprob 대비 허용 margin |
| `min_length_ratio_for_longer` | `1.05` | 더 긴 후보로 볼 길이비 하한 |

### 2.4 Gates
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `logprob_threshold` | `-1.0` | 저신뢰 윈도우 기준 |
| `no_speech_threshold` | `0.6` | no-speech drop 기준 |
| `no_speech_logprob_threshold` | `-1.0` | no-speech drop 시 logprob 보조 기준 |
| `compression_ratio_threshold` | `2.4` | gzip 압축비 degeneracy 기준 |

### 2.5 Fallback (fallback_policy)
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `fallback_policy` | `"off"` | `off` / `gate_fail` / `low_logprob` / `degenerate` / `always` |
| `temperature_fallback` | `()` | 재디코드 온도 사다리(예 `(0.2,0.4,0.6)`) |
| `fallback_sampling_topk` | `0` | 0이면 `sampling_topk` 사용 |

### 2.6 Language policy
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `language_policy` | `"fixed"` | `fixed` / `per_window_confident` / `dual_band` |
| `language_override_prob` | `0.7` | per-window override 채택 임계 |
| `dual_language_low_prob` / `dual_language_high_prob` | `0.4` / `0.7` | dual_band 애매 구간 |

### 2.7 Token / script
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `suppress_blank` | `True` | 시작 blank 억제 |
| `suppress_tokens` | `(-1,)` | -1=model default symbol set |
| `suppress_cjk_kana` | `True` | CJK/Kana 억제(한글/영문/숫자 보존) |
| `max_initial_timestamp_index` | `50` | 첫 timestamp 상한 |

### 2.8 Segmentation / seek
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `window_seconds` | `30.0` | 윈도우 길이 |
| `timestamp_resolution` | `0.02` | timestamp/RMS hop 해상도 |
| `min_advance_seconds` | `20.0` | seek 최소 전진 가드 |
| `silence_percentile` | `20.0` | RMS silence floor 퍼센타일 |

### 2.9 Context (context_policy)
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `context_policy` | `"confidence_gated"` | `off` / `always` / `confidence_gated` |
| `max_context_tokens` | `200` | carry할 context 토큰 상한 |

### 2.10 Align tail trim
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `align_tail_trim` | `False` | 꼬리 trim 활성(백엔드 hook 필요) |
| `align_prob_floor` | `0.3` | 꼬리로 볼 posterior 임계 |
| `align_min_run` | `8` | 연속 저posterior 최소 길이 |
| `align_trigger_low_logprob` / `align_trigger_degenerate` | `True` / `True` | 트리거 조건 |

### 2.11 Fusion (요청 단위 override)
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `lm_enabled` | `False` | fusion on/off |
| `alpha` | `None` | None이면 `alpha_default`. `<=0`이면 fusion 미적용 |
| `topk` | `None` | None이면 `topk_default` |
| `fusion_debug` | `False` | fusion 디버그 출력 |

상세: [`fusion.md`](fusion.md).

### 2.12 Output opt-in
| 파라미터 | 기본 | 채워지는 곳 |
|---|---|---|
| `return_segments` | `False` | `.segments` |
| `return_scores` | `False` | `.scores` |
| `return_nbest` | `False` | `.nbest` (`source`/`language` 포함) |
