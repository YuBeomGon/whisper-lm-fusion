# STT 디코딩 전략 정리

**일자**: 2026-06-09

이 문서는 `whisper-lm-fusion`의 `transcribe()` 내부를 **어떤 디코딩 전략으로 채울지** 정리한 것이다.
과거 `phase3_*` 자동 반복 실험에서 나온 검증된 아이디어를 wrapper 구현 관점으로 재배치했다.

> 이 문서는 내부 연구/전략 노트다. 공개 API 계약은 [`docs/SSOT.md`](../SSOT.md)를 따른다.
> GitHub 공개 시에는 데이터셋명, 내부 run id, 공개 불가 수치를 sanitize하거나 별도 internal
> 문서로 분리한다.

원자료(참고용, 정본):
- `docs/assets/all_experiment_idea_inventory.md` — 아이디어 카탈로그(정성)
- `docs/assets/cer15_method_and_parameter_catalog.md` — 15%대 후보별 정확한 파라미터(정본)

> **경고**: 모든 수치는 0715 eval 기준이다. holdout/다른 corpus에서는 반드시 독립 재평가한다.
> `detect_language`·`align`·glossary는 모두 substitution을 건드리므로 **한 축씩 독립 ablation 후 결합**한다.

---

## 1. 성능 기준점 (best-first)

| 후보 | CER | 핵심 아이디어 |
|---|---:|---|
| `phase3_013_iter_022` | **0.1539** | per-window `detect_language`로 confident non-Korean(prob≥0.7)일 때만 언어 토큰 교체 |
| `phase3_013_iter_016` | 0.1548 | align posterior loop-tail trim (low-logprob window까지 확장) |
| `phase3_004_iter_018` | 0.1574 | conditional temperature fallback (실패 window만 재디코드) |
| `phase3_014_iter_046` | 0.1585 | glossary condition-B를 0.1 logprob margin 안에서 채택 |

---

## 2. 공통 Backbone

15%대 후보 다수는 완전히 다른 알고리즘이 아니라, 아래 backbone 위에 기능이 쌓인 형태다.
이것이 모든 새 실험의 기본 출발점이다.

| 축 | 공통 경향 |
|---|---|
| chunk | 30초 window batched decode |
| timestamp | timestamp token으로 seek advance (blind 30초 stride보다 우월) |
| gate | `avg_logprob < -1.0`, compression ratio `> 2.4`를 실패 신호로 사용 |
| fallback | 실패 window만 temperature fallback |
| search | 상위권은 greedy보다 beam/N-best 사용 |
| context | 일부는 `<|startofprev|>`로 이전 context 또는 glossary 주입 |

---

## 3. 디코딩 전략 카테고리 (8군)

각 항목: 아이디어 → 다시 테스트할 형태(권장 파라미터).

### 3.1 Long-form / seek / segmentation
- **30초 window batched decode** — 30초 이후를 버리는 stub 구조의 삭제 오류 제거. *모든 새 실험의 기본 backbone.*
- **timestamp 기반 adaptive seek** — blind stride보다 우월. 15%대 decode stack과 결합.
- **silence-aware cut / RMS trough** — 긴 발화 중간을 자르지 않도록 경계 탐색. forced fallback cut에만 RMS trough 적용.
- **dual grid / phase-shift witness** — seam 인근 짧은 replace span에만 제한 적용.
- **trailing segment re-decode** — 마지막 segment를 다음 window interior로 재해석, difflib reconcile.
- **hierarchical segment re-decode** — low-confidence(`lp < -0.5`) band window 내부 segment만 isolate decode.

### 3.2 Decode search / fallback
- **beam search + N-best** — substitution 축에서 일관되게 강함. `beam=5`, `num_hypotheses=5` 기본값.
- **patience 확대** — early pruning되는 correct path 구제. `patience=2.0`(013계열) / `1.0`(014계열).
- **beam width 8** — 검색 폭 확대. runtime budget 안에서 `beam=5` 대비 isolated A/B.
- **conditional temperature fallback** — *15%대 검증 핵심.* gate `avg_logprob<-1.0`, `compression>2.4` 통과 실패 window만 재디코드.
- **fallback ladder trim** — high-temp rung은 noise 생성. `(0.0,0.2,0.4,0.6)` vs `(0.0,0.4,0.8)` 비교.
- **axis-aware selector** — 반복 후보는 compression 우선, 정상 후보는 logprob 우선 tie-break.
- **no-speech single beam gate** — hallucination 방어. fallback 뒤 최종 emission gate로 결합.

### 3.3 Language / prompt / glossary
- **per-window language detection** — *전체 최고 성능.* top language prob ≥ 0.7일 때만 `<|ko|>` override.
- **ambiguous band dual-language decode** — `[0.4,0.7)` band에서만 양쪽 decode 후 선택.
- **static glossary via `<|startofprev|>`** — rolling feedback은 악화, static prompt는 domain prior로 유효. rolling tail 제거.
- **glossary logprob margin** — strict score 비교가 domain correction을 버리는 문제 해결. `lp_b ≥ lp_a - 0.1`.
- **glossary trigger decouple** — cond-B를 `lp_a < -0.5` band까지 허용(acceptance gate와 분리).
- **confidence-gated context carry** — compression/logprob 통과 window만 context로 carry (무조건 carry는 오류 전파).
- **bidirectional context** — low-confidence window에만 2-pass future context.
- **term bank / recurrence glossary** — 안정적으로 반복된 단어를 bank화, re-emission penalty와 함께 drift 방지.

### 3.4 Alignment / acoustic verification
- **align posterior tail trim** — loop/hallucinated tail 제거. floor `0.3`, min run `8`, trigger=logprob fail 또는 repeated_text.
- **frame advance trim** — frame 진행 정체를 loop 신호로 사용. `prob floor` OR `frame advance`.
- **align-based N-best rerank** — score가 LM-biased일 때 acoustic posterior로 선택. top-2/top-5 disagreement window에만.
- **align-verified lexicon rewrite** — jamo rewrite 후 changed span posterior 비교.
- **token prob excision** — isolated unsupported token만 제거, 연속 저확률 run은 보수 처리.

### 3.5 N-best / consensus / ensemble
- **beam medoid / MBR** — margin이 충분할 때만 top beam override.
- **token/word majority vote** — equal-length replace span 한정.
- **sampling MBR / independent draws** — `temp=0.2`, `topk=8`, `N=5` 주변. beam과 다른 오류 분포 witness.
- **ROVER gated override** — anchor support gate + vote min 조합.
- **lexicon rerank inside epsilon** — logprob 근접 후보 중 domain term coverage로 선택, strict superset gate.
- **longest beam within score margin** — under-generation/deletion 보정. compression-pass + margin `0.10~0.15`.

### 3.6 Token constraints / repetition / hallucination guards
- **CJK/Kana/foreign script suppression** — Korean + allowed Latin 외 script mask. cross-script substitution 방지.
- **default suppress_tokens 해제** — deletion-heavy backbone에서만 isolated test.
- **suppress_blank=False** — clipped onset에서 첫 content token 강제 방지. `max_initial_timestamp`와 onset A/B.
- **repetition_penalty=1.1** — 반복 loop 완화. `1.05/1.1/1.15`를 guard metric과 함께 측정.
- **no_repeat_ngram** — exact loop만 막는 targeted guard. ngram 4/5/6 비교.
- **seam exact dedup** — exact consecutive duplicate segment만 제거.
- **no_speech threshold tuning** — `0.5/0.6` + avg_logprob joint gate.

### 3.7 Audio frontend / input conditioning
- **high-pass / low-band removal** — 전화 음성 저역 rumble 제거. boxcar high-pass width `sr/80` 전후.
- **pre-emphasis** — obstruent/high-freq cue 복원. raw trusted floor + margin 있는 dual front-end.
- **loudness normalization / gain cap** — target RMS `0.05`, max gain `2.5`부터. (조용한 파일 noise 증폭 주의)
- **VTLN / formant warp witness** — whole pipeline이 아니라 correction witness로만 사용.
- **dual front-end score arbitration** — raw score가 낮을 때만 pre-emphasis second pass.

### 3.8 Post-decode lexical correction
- **jamo consonant correction** — 동일 음절수/모음, 첫 음절/조사 guard 유지.
- **recurrence canonicalization** — corpus 내부 반복 count threshold 필요.
- **align-verified glossary rewrite** — changed span만 posterior 비교.
- **logit glossary snap** — `return_logits_vocab` 가능성 확인 후 별도 test.

---

## 4. 파라미터 정본값 요약

| 그룹 | 파라미터 | 값 | 비고 |
|---|---|---|---|
| Search | `beam_size` | `5` | 15%대 backbone 핵심값 |
| Search | `num_hypotheses` | `5` (또는 `beam_size`) | selector/consensus 기반 |
| Search | `patience` | `2.0`(013) / `1.0`(014) | 013계열에서 강함 |
| Search | `length_penalty` | `1.2` | deletion/short hypothesis 대응, 단독 best는 아님 |
| Gate | avg logprob | `-1.0` | 표준 confidence gate |
| Gate | compression ratio | `2.4` | repetitive decode 감지 |
| Gate | no_speech | `0.6` (또는 `>0.5`) | hallucination guard |
| Fallback | temperature ladder | `(0.0,0.4,0.8)`(004) / `(0.2,0.4,0.6)`(013) / `(0.0..1.0)`(014) | 넓으면 high-temp 부작용 |
| Language | override prob | `0.7` | confident non-Korean only가 안전 |
| Language | dual decode band | `0.4..0.7` | 022보다 약간 나쁨, 비용 증가 |
| Align | prob floor | `0.3` | loop-tail trim |
| Align | min run | `8` tokens | 과도한 clipping 방지 |
| Align | trigger | degenerate or low-logprob | degenerate-only보다 좋음 |
| Glossary | cond-B margin | `score_B ≥ score_A - 0.1` | 014 best |
| Glossary | cond-B trigger | gate-fail (또는 `lp_a < -0.5`) | 안전 vs reach 확대 |

---

## 5. 기록된 부정적 교훈

- rolling transcript를 무조건 `<|startofprev|>`로 주입하면 오류가 다음 window로 전파된다 → static glossary / confidence-gated carry.
- 단순 loudness/pre-emphasis는 조용한 파일에서 noise도 키워 substitution을 악화 → gain cap / score arbitration.
- 큰 repetition penalty는 정상 반복 한국어 토큰까지 눌러 deletion 증가.
- 높은 temperature ladder는 low-confidence window에서 rare-token tail/hallucination 생성 → ladder trim / best-of·MBR gate.
- strict glossary/post-correction은 false positive 위험 → logprob margin, align posterior, jamo guard, superset gate.
- `suppress_tokens` script mask는 Korean-only엔 유리하나 code-switch/Latin acronym을 막지 않도록 범위 명확화.
- whole-hypothesis MBR/ROVER는 평균적 이상 문장을 만들 수 있음 → anchor-support / margin gate.

---

## 6. 권장 재실험 순서

한 번에 모두 합치면 원인 attribution이 어렵다. 한 축씩 켜며 측정한다.

| step | base | change | 이유 |
|---|---|---|---|
| 1 | timestamp-seek + fallback | `beam=5`, `num_hypotheses=5`, `patience=2.0` | 013계열 안정 backbone |
| 2 | step 1 | `detect_language` confident override `≥0.7` | 전체 best의 핵심 |
| 3 | step 1 | align trim `prob<0.3`, min run `8`, trigger low-logprob 포함 | second-best 계열 |
| 4 | step 1 | conditional fallback temps `(0.0,0.4,0.8)` 또는 `(0.2,0.4,0.6)` | low-confidence rescue 비교 |
| 5 | step 1 | glossary condition-B, margin `0.1` | domain substitution dataset 확인 |
| 6 | best of 2-5 | `length_penalty=1.2`, `max_initial_timestamp_index=0`, `suppress_blank=False` 각각 ablation | deletion/cold-onset 개선 여부 |

---

## 7. wrapper 구현 범위 매핑

design.md의 1차 thin 범위와 위 전략의 대응.

| 범위 | 대상 전략 |
|---|---|
| **1차 (thin)** | 3.1 30초 window + timestamp seek, 3.2 beam=5/N-best=5 + conditional temperature fallback, no_speech gate |
| **후속(코어 확장)** | 3.3 language detect override, 3.4 align trim, 3.5 N-best rerank, 3.6 script suppress |
| **pipeline 소유(wrapper 비범위)** | glossary corpus/빌드, jamo/recurrence post-correction 규칙, domain term bank, alpha/topk "좋은 값" |

> wrapper는 generic이므로 glossary "내용"·domain term·corpus는 모른다. 위 glossary/post-correction
> 전략은 **파라미터·토큰을 받아 적용하는 메커니즘**만 wrapper에 두고, "무엇을" 넣을지는 pipeline이 결정한다.
