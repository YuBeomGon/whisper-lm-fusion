# Self-Evolving STT Pipeline Technique Taxonomy

# Self-Evolving STT Pipeline: Autoresearch로 발견된 Long-form Whisper/CT2 개선 기법 Taxonomy

> 기준 자료: `restored.zip` 내부의 `MANIFEST.tsv`와 각 후보의 `transcribe.py` 구현.  
> 제외 범위: harness/evolve 본체, 검증 코드, 외부 score report. 사용자가 제공한 압축에는 후보 코드와 CER manifest만 있으므로, harness 구조는 별도 문서에서 다루는 것이 정확하다.

------------------------------------------------------------------------

## 0. 요약

이 문서는 STT 파이프라인을 self-evolve/autoresearch 방식으로 고도화하면서 생성된 후보 코드들을 기술적으로 분류한 것이다. 핵심은 “하나의 수동 설계”가 아니라, 장문 Whisper/CT2 디코딩에서 사용할 수 있는 다양한 control surface가 실제 코드 후보로 탐색되었다는 점이다.

제공된 `MANIFEST.tsv` 기준으로 후보 수와 CER 분포는 다음과 같다.

| Baseline / target 기준 | 내용                                                                                       |
|------------------------|--------------------------------------------------------------------------------------------|
| 평가 데이터            | `AIG_녹취반출_20250715`, scored 11개 파일, 약 5시간 분량                                   |
| 초기 앵커              | `0.4685` — beam5 bare default 기준                                                         |
| 재측정 baseline        | **`0.1714`** — faster-whisper prod-left 풀 파이프라인, 2026-06-01 re-anchor                |
| target CER             | `0.10` — 성공 기준                                                                         |
| 역대 best              | `phase3_013_iter_022`, **`0.153887`** ≈ `0.1539`                                           |
| 해석                   | 역대 best는 재측정 baseline `0.1714`는 넘어섰지만, target `0.10`에는 아직 도달하지 못했다. |

| 항목                 |           값 |
|----------------------|-------------:|
| manifest row 수      |          693 |
| CER가 기록된 후보 수 |          520 |
| CER 미기록 후보 수   |          173 |
| best CER             | **0.153887** |
| median CER           |     0.180476 |
| mean CER             |     0.213199 |
| max CER              |     0.698360 |
| CER \<= 0.16         |         41개 |
| CER \<= 0.17         |        158개 |
| CER \<= 0.18         |        258개 |
| CER \<= 0.20         |        388개 |

최고 후보는 다음이다.

| 순위 | 후보                  |          CER | 비고                                         |
|-----:|-----------------------|-------------:|----------------------------------------------|
|    1 | `phase3_013_iter_022` | **0.153887** | 여러 강한 기법이 함께 들어간 조합형 champion |
|    2 | `phase3_013_iter_023` |     0.154541 | ambiguous language dual-branch 포함          |
|    3 | `phase3_013_iter_016` |     0.154828 | align tail trim trigger 확장 계열            |
|    4 | `phase3_013_iter_013` |     0.155107 | selective notimestamps 계열                  |
|    5 | `phase3_013_iter_018` |     0.155107 | align/context/beam 계열                      |
|    6 | `phase3_013_iter_014` |     0.155185 | align posterior tail trim 계열               |
|    7 | `phase3_013_iter_019` |     0.155403 | beam/fallback 변형                           |
|    8 | `phase3_013_iter_020` |     0.155586 | beam width 확대 계열                         |
|    9 | `phase3_013_iter_015` |     0.155604 | beam/fallback 변형                           |
|   10 | `phase3_013_iter_021` |     0.156057 | align-based N-best rerank                    |

중요한 해석상 주의점은 다음이다.

- 대부분의 후보는 **단일 변수 ablation**이 아니다.
- 아래 40개 기법의 CER는 “그 기법이 구현된 후보의 CER”이지, “그 기법 하나의 순수 기여도”를 뜻하지 않는다.
- 일부 후보의 docstring에는 이전 iteration 대비 의도와 진단 메모가 남아 있지만, 제공 자료 안에는 체계적인 parent graph나 ablation report가 없다.
- 따라서 각 technique card에는 `단독성/기여도 판정`을 별도 표시한다.

------------------------------------------------------------------------

## 1. 데이터와 해석 기준

### 1.1 확인 가능한 것

- 각 후보의 `transcribe.py` 실제 구현
- `MANIFEST.tsv`의 후보별 CER
- 일부 코드 주석에 남은 진단 메모
  - 예: `sub 57%`, `sub 58%`, `length_ratio 0.94`, `length_ratio 0.95`, 특정 파일 CER 등
- CT2/Whisper 호출 surface
  - `generate(..., return_scores=True)`
  - `generate(..., return_no_speech_prob=True)`
  - `generate(..., beam_size=..., num_hypotheses=...)`
  - `model.align(...)`
  - `model.detect_language(...)`
  - `suppress_tokens`
  - `<|startofprev|>`, `<|notimestamps|>`, timestamp token

### 1.2 확인 불가능하거나 제한적인 것

| 항목                    | 상태                                                                        |
|-------------------------|-----------------------------------------------------------------------------|
| runtime wall-clock      | 별도 로그 없음. 코드 구조 기반 비용 추정만 가능                             |
| sub/del/ins 세부 metric | 별도 metric 파일 없음. 일부 후보 docstring의 진단 메모만 존재               |
| hallucination rate      | 별도 로그 없음. compression/no_speech/repetition 관련 코드로 간접 확인 가능 |
| 순수 ablation delta     | 대체로 없음. 후보 대부분이 조합형                                           |
| harness/evolve 구조     | 제공 파일에 없음. 본 문서 범위에서 제외                                     |

------------------------------------------------------------------------

## 2. 전체 Technique Taxonomy

40개 기법은 다음 그룹으로 묶을 수 있다.

| 그룹                                    |          기법 번호 | 주제                                                                           |
|-----------------------------------------|-------------------:|--------------------------------------------------------------------------------|
| A. Long-form windowing / seek           |        1-3, 38, 40 | 장문 오디오를 30초 window로 나누고 seam을 다루는 방법                          |
| B. Confidence / fallback                |         4-7, 11-12 | score, compression, no-speech, timestamp/no-timestamp decode mode              |
| C. Beam / N-best / selection            | 8-10, 15-17, 27-29 | beam, N-best, align rerank, MBR/ROVER/sampling consensus                       |
| D. Language / script control            |          13-14, 24 | language token override, CJK/Kana suppress                                     |
| E. Domain prior / glossary / context    |              18-23 | glossary, lexicon, self-harvested term memory, previous/future context         |
| F. Suppression / decoding constraints   |              25-26 | rank disagreement suppress, repetition/no-repeat                               |
| G. Post-correction / self-consistency   |              30-32 | majority spelling, jamo correction, logit-distribution snap                    |
| H. Audio / feature frontend             |              33-37 | loudness, pre-emphasis, mel feature 조작, VTLN, artificial bandwidth extension |
| I. 실패했지만 중요한 window commit 계열 |              39-40 | overlap stitch, central band, left-context run-up                              |

------------------------------------------------------------------------

## 3. 40개 기법 전체 목록

> `단독성` 표기 기준:  
> - `분리형에 가까움`: 기존 backbone 위에 특정 축을 비교적 분리해 추가한 후보. 그래도 완전한 ablation은 아님.  
> - `조합형`: 여러 기법이 동시에 포함되어 순수 기여도를 분리하기 어려움.  
> - `실패/탐색형`: CER는 나빴지만 아이디어 자체가 의미 있는 후보.

|  \# | 기법                                                  | 대표 후보                    |                 CER | 단독성/기여도 판정 | 핵심                                                    |
|----:|-------------------------------------------------------|------------------------------|--------------------:|--------------------|---------------------------------------------------------|
|   1 | sequential 30s windowing                              | `phase3_013_iter_002`        |            0.411350 | 분리형에 가까움    | 30초 이후 truncation 문제 복구                          |
|   2 | timestamp-driven adaptive seek                        | `phase3_004_iter_007`        |            0.161110 | 분리형에 가까움    | 마지막 timestamp token으로 seek advance                 |
|   3 | silence/RMS acoustic cut                              | `phase3_013_iter_006`        |            0.175095 | 조합형             | pause 지점에서 window boundary 안정화                   |
|   4 | `return_scores` confidence gate                       | `phase3_004_iter_011`        |            0.161110 | 조합형             | avg logprob로 window 신뢰도 판단                        |
|   5 | temperature fallback ladder                           | `phase3_004_iter_018`        |            0.157355 | 조합형             | 낮은 품질 window만 temperature 재디코드                 |
|   6 | gzip compression degeneracy gate                      | `phase3_013_iter_009`        |            0.170861 | 조합형             | 반복/degenerate text 감지                               |
|   7 | no-speech probability guard                           | `phase3_014_iter_048`        |            0.158976 | 조합형             | silence hallucination 방어                              |
|   8 | beam search + N-best                                  | `phase3_013_iter_011`        |            0.158801 | 조합형             | greedy local commit 완화                                |
|   9 | beam patience 조정                                    | `phase3_013_iter_012`        |            0.156789 | 조합형             | beam 후보를 더 오래 유지                                |
|  10 | length/coverage selection                             | `phase3_004_iter_023`        |            0.158967 | 조합형             | deletion 방지 위해 coverage/길이 고려                   |
|  11 | selective `<|notimestamps|>` clean-text decode        | `phase3_013_iter_013`        |            0.155107 | 조합형             | timestamp는 seek용, text는 clean decode                 |
|  12 | cross-decode-mode fallback                            | `phase3_004_iter_040`        |            0.161564 | 조합형             | timestamp/no-timestamp branch 비교                      |
|  13 | per-window language detection override                | `phase3_013_iter_022`        |        **0.153887** | 조합형             | window별 language posterior로 SOT token 조정            |
|  14 | ambiguous language dual-branch                        | `phase3_013_iter_023`        |            0.154541 | 조합형             | ko branch와 detected-language branch 병렬 비교          |
|  15 | `model.align()` tail trim                             | `phase3_013_iter_014`        |            0.155185 | 조합형             | low posterior tail 제거                                 |
|  16 | align-based N-best rerank                             | `phase3_013_iter_021`        |            0.156057 | 조합형             | decoder score 대신 acoustic posterior로 후보 재평가     |
|  17 | alignment field 기반 seek                             | `phase3_004_iter_044`        |            0.312410 | 실패/탐색형        | timestamp 대신 forced alignment로 seek 시도             |
|  18 | static glossary prompt prefix                         | `phase3_014_iter_039`        |            0.159037 | 조합형             | domain glossary를 soft prompt로 투입                    |
|  19 | glossary branch + logprob margin                      | `phase3_014_iter_046`        |            0.158540 | 조합형             | glossary branch가 margin 안이면 채택                    |
|  20 | lexicon-guided N-best rescoring                       | `phase3_030_iter_068`        |            0.171305 | 조합형             | N-best 후보를 domain lexicon으로 재점수화               |
|  21 | self-harvested glossary / term memory                 | `phase3_014_iter_034`        |            0.178415 | 탐색형             | 한 call 내 반복 term을 prior로 사용                     |
|  22 | confidence-gated `<|startofprev|>` carry              | `phase3_016_iter_005`        |            0.160474 | 조합형             | 이전 문맥을 신뢰도 있을 때만 carry                      |
|  23 | bidirectional / right-context conditioning            | `phase3_004_iter_026`        |            0.162278 | 조합형             | 1차 transcript의 앞/뒤 window를 prompt로 사용           |
|  24 | CJK/Kana/script suppress                              | `phase3_013_iter_017`        |            0.156762 | 조합형             | 히라가나/가타카나/CJK token suppress                    |
|  25 | rank-disagreement suppress-and-redecode               | `phase3_014_iter_029`        |            0.164474 | 탐색형             | rank0/rank1 충돌 token을 금지 후 재디코드               |
|  26 | repetition penalty / no-repeat-ngram                  | `phase3_004_iter_017`        |            0.160204 | 조합형             | 반복 hallucination 억제                                 |
|  27 | MBR over N-best                                       | `phase3_014_iter_077`        |            0.163820 | 탐색형             | 평균적으로 가장 유사한 후보 선택                        |
|  28 | word-level majority / beam consensus                  | `phase3_014_iter_058`        |            0.161860 | 탐색형             | beam 후보들의 단어 단위 다수결                          |
|  29 | independent sampling consensus / ROVER                | `phase3_014_iter_082`        |            0.162417 | 탐색형             | sampling 후보 4개로 replace span vote                   |
|  30 | majority spelling canonicalization                    | `phase3_014_iter_037`        |            0.161895 | 분리형에 가까움    | transcript 내부 frequent spelling으로 rare variant 교정 |
|  31 | Jamo self-consensus / glossary correction             | `phase3_014_iter_041`        |            0.160030 | 분리형에 가까움    | 한글 자모 분해로 자음 혼동 교정                         |
|  32 | per-step logit distribution glossary snap             | `phase3_014_iter_052`        |            0.161799 | 탐색형             | vocab distribution 기반 glossary snap 시도              |
|  33 | loudness/RMS normalization                            | `phase3_030_iter_018`        |            0.163271 | 탐색형             | 조용한 speech frame 보강                                |
|  34 | pre-emphasis / high-pass / spectral shaping           | `phase3_013_iter_026`        |            0.158575 | 조합형             | 고역/자음 cue 강조                                      |
|  35 | feature-domain CMN / mel ramp / unsharp / variance EQ | `phase3_014_iter_063`        |            0.165598 | 탐색형             | waveform이 아닌 log-mel feature 직접 조작               |
|  36 | VTLN / acoustic perturbation ensemble                 | `phase3_014_iter_064`        |            0.162139 | 탐색형             | feature warp view를 witness로 사용                      |
|  37 | artificial bandwidth extension / nonlinear frontend   | `phase3_014_iter_071`        |            0.176864 | 탐색형             | rectification으로 high-band witness 생성                |
|  38 | phase-shifted dual-grid decoding                      | `phase3_014_iter_068`        |            0.161860 | 탐색형             | 0초 grid와 15초 shift grid를 seam-local reconcile       |
|  39 | overlap window + stitch                               | `phase3_030_iter_104`        |            0.290896 | 실패/탐색형        | overlap text/token matching stitch                      |
|  40 | central-band commit / left-context run-up             | `phase3_014_iter_057`, `061` | 0.494249 / 0.410435 | 실패/탐색형        | window edge를 버리거나 run-up을 decode 후 discard       |

------------------------------------------------------------------------

# 4. Technique Cards

## A. Long-form windowing / seek 계열

### 1) Sequential 30s windowing

- 대표 후보: `phase3_013_iter_002`, CER `0.411350`
- 구현 성격: 비교적 분리형
- 핵심: Whisper feature extractor가 30초 window로 pad/truncate하므로, 긴 오디오를 30초 단위로 직접 slice하여 전체 tail을 보게 함.
- 의미: 이후 모든 long-form 후보의 기반. 단독으로는 CER가 높지만, 첫 30초 이후 삭제 문제를 복구한다.
- 한계: blind 30초 cut은 문장 중간을 자르기 때문에 boundary deletion/substitution을 만들 수 있다.

### 2) Timestamp-driven adaptive seek

- 대표 후보: `phase3_004_iter_007`, CER `0.161110`
- 구현 성격: 비교적 분리형
- 핵심: `<|notimestamps|>`를 제거해 timestamp token을 출력하게 하고, 마지막 complete segment timestamp를 읽어 다음 `seek` 위치를 정한다.
- 코드상 개념:

``` sourceCode
last_ts = token_id - timestamp_begin
advance_seconds = last_ts * 0.02
seek += advance_seconds * sr
```

- 의미: blind +30s advance보다 훨씬 안전하다. decoder가 완성했다고 표시한 segment까지만 commit하고, 나머지는 다음 window에서 다시 보게 한다.
- 한계: timestamp token 자체가 틀릴 수 있으므로 minimum advance guard가 필요하다.

### 3) Silence/RMS acoustic cut

- 대표 후보: `phase3_013_iter_006`, CER `0.175095`; `phase3_016_iter_003`, CER `0.162653`
- 구현 성격: 조합형
- 핵심: raw audio의 frame RMS를 계산해 window tail에서 에너지가 낮은 pause 지점을 찾고 그 지점을 boundary로 삼는다.
- 의미: timestamp는 decoder output이고 RMS는 audio signal이므로 서로 다른 evidence이다.
- 한계: 조용한 발화와 silence를 구분하기 어렵고, 전화 잡음이 있으면 RMS floor가 불안정하다.

------------------------------------------------------------------------

## B. Confidence / fallback 계열

### 4) `return_scores` confidence gate

- 대표 후보: `phase3_004_iter_011`, CER `0.161110`
- 구현 성격: 조합형
- 핵심: CT2 `generate(..., return_scores=True)`의 `scores[0]`를 window-level 평균 logprob로 사용해 low-confidence window를 표시한다.
- 의미: 모든 window를 동일하게 취급하지 않고, 재디코드/후처리 대상만 좁힐 수 있다.
- 한계: fluent-but-wrong substitution은 score가 높을 수 있다.

### 5) Temperature fallback ladder

- 대표 후보: `phase3_004_iter_018`, CER `0.157355`; `phase3_014_iter_046`, CER `0.158540`
- 구현 성격: 조합형
- 핵심: 처음에는 deterministic decode를 하고, logprob/compression gate 실패 시 temperature를 올려 재디코드한다.
- 코드 주석상 ladder 예: `0.0 -> 0.2 -> 0.4 -> 0.6 -> 0.8 -> 1.0`
- 의미: bad greedy commit을 재검토할 수 있다.
- 한계: 높은 temperature는 rare token, hallucinated tail, 반복을 만들 수 있어 compression/no-speech/align trim이 필요하다.

### 6) Gzip compression degeneracy gate

- 대표 후보: `phase3_013_iter_009`, CER `0.170861`; champion 계열에도 포함
- 구현 성격: 조합형
- 핵심: decoded text를 zlib/gzip으로 압축해 compression ratio가 너무 높으면 반복/degenerate output으로 판단한다.
- 의미: logprob가 높아도 반복 문장은 압축이 잘 되므로 별도 hallucination proxy로 쓸 수 있다.
- 한계: 짧은 문장에서는 ratio가 불안정하다.

### 7) No-speech probability guard

- 대표 후보: `phase3_014_iter_048`, CER `0.158976`; `phase3_014_iter_087`, CER `0.161799`
- 구현 성격: 조합형
- 핵심: `return_no_speech_prob=True`를 사용해 silence/near-silence window에서 text commit을 조심한다.
- 의미: 무음 구간 hallucination 방어.
- 한계: speech probability가 부정확한 모델/조건에서는 과삭제 위험이 있다.

------------------------------------------------------------------------

## C. Beam / N-best / selection 계열

### 8) Beam search + N-best

- 대표 후보: `phase3_013_iter_011`, CER `0.158801`; `phase3_013_iter_022`, CER `0.153887`
- 구현 성격: 조합형
- 핵심: `beam_size > 1`, `num_hypotheses > 1`로 greedy 대신 global cumulative logprob 기반 후보 search를 사용한다.
- 코드 주석 요지: turbo decoder는 4 layers이고 encoder가 비용 대부분이므로, beam 확장은 상대적으로 감당 가능하다고 판단했다.
- 의미: dominant error가 substitution일 때, greedy local commit을 완화한다.
- 한계: beam 후보가 서로 correlated될 수 있고, fluent-but-wrong 후보가 여전히 score에서 이길 수 있다.

### 9) Beam patience 조정

- 대표 후보: `phase3_013_iter_012`, CER `0.156789`
- 구현 성격: 조합형
- 핵심: `patience=2.0` 등으로 beam candidate를 더 오래 유지한다.
- 의미: prefix-level early pruning을 늦춰 substitution fork를 살린다.
- 한계: 비용 증가. beam 후보가 늘어도 acoustic evidence가 부족하면 같은 오류를 공유할 수 있다.

### 10) Length/coverage selection

- 대표 후보: `phase3_004_iter_023`, CER `0.158967`; `phase3_014_iter_073`, CER `0.163507`
- 구현 성격: 조합형
- 핵심: N-best 중 score만 보지 않고 length ratio, coverage, longest-within-margin 등을 고려한다.
- 의미: deletion-dominant 시기에는 짧은 후보를 벌하는 것이 유효하다.
- 한계: 무조건 긴 후보를 고르면 insertion/hallucination이 늘 수 있다.

### 11) Selective `<|notimestamps|>` clean-text decode

- 대표 후보: `phase3_013_iter_013`, CER `0.155107`; champion 계열에도 포함
- 구현 성격: 조합형
- 핵심: timestamp mode는 seek용으로 쓰고, text quality가 중요한 구간은 `<|notimestamps|>` clean decode branch를 사용한다.
- 의미: timestamp token이 text decode capacity를 방해할 수 있다는 가정에서 역할을 분리했다.
- 한계: 모든 구간을 no-timestamp로 바꾸면 seek가 깨질 수 있으므로 selective 적용이 중요하다.

### 12) Cross-decode-mode fallback

- 대표 후보: `phase3_004_iter_040`, CER `0.161564`
- 구현 성격: 조합형
- 핵심: timestamp mode branch와 no-timestamp branch를 비교해 window별로 선택한다.
- 의미: seek 안정성과 clean transcript 품질을 tradeoff로 다룬다.
- 한계: branch 선택 기준이 약하면 coverage가 흔들린다.

------------------------------------------------------------------------

## D. Language / script control 계열

### 13) Per-window language detection override

- 대표 후보: `phase3_013_iter_022`, CER **`0.153887`**
- 구현 성격: 조합형. 순수 기여도는 확인 불가.
- 핵심: `model.detect_language(features)`로 window별 language posterior를 읽고, `<|ko|>` 고정을 일부 완화한다.
- 의미: 한국어 콜센터 음성에도 영어 약어, 외래어, 상품명, code-switch가 섞일 수 있다. 모든 window를 Korean token으로만 밀면 substitution이 생길 수 있다.
- 한계: best 후보에 beam/fallback/RMS/context/align/notimestamps 등 여러 기법이 함께 들어 있어 language override 단독 효과는 별도 ablation 필요.

### 14) Ambiguous language dual-branch

- 대표 후보: `phase3_013_iter_023`, CER `0.154541`
- 구현 성격: 조합형
- 핵심: language posterior가 애매한 window에서 Korean branch와 detected-language branch를 둘 다 decode하고 더 나은 쪽을 선택한다.
- 의미: 무조건 override보다 안전한 형태. 애매한 구간에만 비용을 지불한다.
- 한계: branch selection 기준이 score에 치우치면 LM-biased wrong branch가 선택될 수 있다.

### 24) CJK/Kana/script suppress

- 대표 후보: `phase3_013_iter_017`, CER `0.156762`
- 구현 성격: 조합형. champion 계열 backbone 위에 script suppress를 추가한 형태.
- 핵심: Whisper multilingual vocab을 훑고, token decode 결과에 특정 Unicode script가 포함되면 해당 token id를 `suppress_tokens`에 넣는다.
- 실제 suppress 범위:

``` sourceCode
_SUPPRESS_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0xFF65, 0xFF9F),  # Halfwidth Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)
```

- 실제 적용 방식:

``` sourceCode
cross_script_ids = []
for token_id in range(tokenizer.vocab_size):
    s = tokenizer.decode([token_id])
    if any(lo <= ord(c) <= hi for lo, hi in _SUPPRESS_RANGES for c in s):
        cross_script_ids.append(token_id)

suppress_tokens = [-1, *cross_script_ids]
```

- 의미: 한국어 음성에서 일본어 kana, 중국 한자, CJK compatibility token이 튀는 것을 막는다.
- 중요한 점: Hangul syllables `U+AC00..D7A3`는 suppress하지 않았다. Latin도 suppress하지 않았다. 따라서 `AI`, `ARS`, `DB` 같은 실제 영어 약어는 살릴 수 있다.
- 한계: 중국식 한자어가 실제로 필요한 데이터셋이라면 과억제 가능. 한국어 콜센터 STT에서는 비교적 안전한 편.

------------------------------------------------------------------------

## E. Align / acoustic verification 계열

### 15) `model.align()` 기반 tail trim

- 대표 후보: `phase3_013_iter_014`, CER `0.155185`; `phase3_013_iter_016`, CER `0.154828`
- 구현 성격: 조합형
- 핵심: decoded token sequence를 다시 acoustic frames에 align하고, text token posterior가 낮은 tail run이 있으면 tail을 잘라낸다.
- 코드상 guard 예: `_ALIGN_PROB_FLOOR = 0.3`, `_ALIGN_MIN_RUN = 8`
- 의미: decoder가 만든 fluent tail이 실제 audio evidence를 갖는지 확인한다.
- 한계: align 자체가 틀리면 정상 tail을 자를 수 있다.

### 16) Align-based N-best rerank

- 대표 후보: `phase3_013_iter_021`, CER `0.156057`; `phase3_014_iter_023`, CER `0.161886`
- 구현 성격: 조합형
- 핵심: beam 후보들을 decoder score가 아니라 align posterior로 재평가한다.
- 의미: LM-biased fluent wrong candidate 대신 acoustic evidence가 좋은 후보를 고를 수 있다.
- 한계: align 비용이 추가되고, 후보 간 text length 차이가 크면 비교가 어렵다.

### 17) Alignment field 기반 seek

- 대표 후보: `phase3_004_iter_044`, CER `0.312410`
- 구현 성격: 실패/탐색형
- 핵심: timestamp token이 아니라 `model.align()`이 반환하는 alignment field로 window advancement를 잡으려는 시도.
- 의미: decoder timestamp보다 forced alignment를 seek 기준으로 쓰려는 참신한 접근.
- 실패 원인 추정: align result를 long-form seek 기준으로 안정적으로 변환하기 어렵고, coverage가 깨졌을 가능성이 높다.

------------------------------------------------------------------------

## F. Glossary / lexicon / context 계열

### 18) Static glossary prompt prefix

- 대표 후보: `phase3_014_iter_039`, CER `0.159037`
- 구현 성격: 조합형
- 핵심: 보험/콜센터 domain term을 `<|startofprev|>` 또는 prompt prefix로 넣는다.
- 의미: domain prior를 decoder에 soft하게 주입한다.
- 한계: soft prompt는 실제 acoustic fork에서 반드시 이기지 않는다.

### 19) Glossary branch + logprob margin

- 대표 후보: `phase3_014_iter_046`, CER `0.158540`; `phase3_014_iter_051`, CER `0.158671`
- 구현 성격: 조합형
- 핵심: 일반 branch와 glossary-conditioned branch를 둘 다 decode하고, glossary branch score가 약간 낮아도 margin 안이면 채택한다.
- 코드 주석상 의도: score channel은 fluent-but-wrong spelling에 LM bias를 가질 수 있으므로 strict greater-than 대신 margin adoption 사용.
- 의미: 도메인 용어는 score가 약간 낮아도 정답일 수 있다.
- 한계: glossary가 부정확하면 false positive가 생긴다.

### 20) Lexicon-guided N-best rescoring

- 대표 후보: `phase3_030_iter_068`, CER `0.171305`
- 구현 성격: 조합형
- 핵심: N-best 후보에 domain lexicon 포함 여부/가중치를 반영해 재선택한다.
- 의미: score-only selection의 LM bias를 보정한다.
- 한계: lexicon coverage가 낮거나 오염되면 성능이 낮다.

### 21) Self-harvested glossary / global term memory

- 대표 후보: `phase3_014_iter_034`, CER `0.178415`; `phase3_030_iter_099~103`, CER 약 `0.168~0.199`
- 구현 성격: 탐색형
- 핵심: 한 통화 안에서 반복 등장하는 term을 모아 다음 window prior로 사용한다.
- 의미: 통화 하나는 동일 보험 상품/상담 주제를 반복하므로 내부 term redundancy를 활용할 수 있다.
- 한계: 초반 오인식 term을 memory에 넣으면 오류가 전파된다.

### 22) Confidence-gated `<|startofprev|>` carry

- 대표 후보: `phase3_016_iter_005`, CER `0.160474`; `phase3_014_iter_066`, CER `0.160291`
- 구현 성격: 조합형
- 핵심: 이전 window text를 무조건 carry하지 않고, score/compression 등 confidence가 좋은 경우에만 `<|startofprev|>` context로 사용한다.
- 의미: context의 장점과 오류 전파 위험을 같이 관리한다.
- 한계: gate가 보수적이면 context 효과가 약하고, gate가 느슨하면 오염된다.

### 23) Bidirectional / right-context conditioning

- 대표 후보: `phase3_004_iter_026`, CER `0.162278`; `phase3_030_iter_105`, CER `0.171314`
- 구현 성격: 조합형
- 실제 코드 기준 설명:
  - Pass 1에서 전체 오디오를 window별로 한 번 decode한다.
  - 각 window의 text, score, gate_failed 여부를 저장한다.
  - Pass 2에서 gate-failing window만 다시 decode한다.
  - 이때 context는 `texts[i-1]`와 `texts[i+1]`, 즉 이전 window와 다음 window의 1차 transcript를 합친다.
  - 이 context를 tokenizer로 encode해 `<|startofprev|>` 앞에 넣는다.

코드 구조:

``` sourceCode
ctx_parts = []
if i > 0 and texts[i - 1]:
    ctx_parts.append(texts[i - 1])
if i + 1 < len(windows) and texts[i + 1]:
    ctx_parts.append(texts[i + 1])
ctx = " ".join(ctx_parts).strip()
ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)[-_MAX_PREV_TOKENS:]
ctx_prompt = [sot_prev, *ctx_ids, *prompt_tokens]
new_ids, new_score = _decode(features, ctx_prompt)
```

- 왜 참신한가:
  - Whisper는 원래 causal decoding이므로 미래 transcript를 볼 수 없다.
  - 하지만 offline STT에서는 1차 pass 이후 미래 window text가 이미 존재한다.
  - 이 후보는 `<|startofprev|>` 슬롯을 “previous text”가 아니라 “surrounding context”로 재활용했다.

예시:

``` text
Pass 1:
W1: 보험금 청구 관련해서
W2: [low confidence] 서류를 보내주시면 되고
W3: 담당자가 확인 후 연락드립니다

Pass 2 for W2:
context = W1 + W3
```

- 한계:
  - 미래 context도 1차 STT 결과라 오류를 포함할 수 있다.
  - `<|startofprev|>`는 본래 과거 문맥용이라 미래 문맥을 넣는 것은 hack이다.
  - CER는 최고권은 아니지만, offline two-pass STT 관점에서는 연구 가치가 있다.

------------------------------------------------------------------------

## G. Suppression / decoding constraints 계열

### 25) Rank-disagreement suppress-and-redecode

- 대표 후보: `phase3_014_iter_029`, CER `0.164474`; `phase3_014_iter_088`, CER `0.165031`
- 구현 성격: 탐색형
- 핵심: beam rank0와 rank1이 특정 span에서 disagreement를 보일 때, rank0의 의심 token을 suppress하고 재디코드한다.
- 의미: 단순 rerank가 아니라 search space를 다시 열어주는 방식이다.
- 한계: 잘못 suppress하면 정답 token을 금지할 수 있다.

### 26) Repetition penalty / no-repeat-ngram

- 대표 후보: `phase3_004_iter_017`, CER `0.160204`; `phase3_030_iter_039`, CER `0.163158`
- 구현 성격: 조합형
- 핵심: 반복 token/ngram을 decode-time constraint로 직접 억제한다.
- 의미: hallucination loop 방어.
- 한계: 한국어 정상 반복 표현까지 눌러 deletion을 만들 수 있다.

------------------------------------------------------------------------

## H. Consensus / MBR / ROVER 계열

### 27) MBR over N-best

- 대표 후보: `phase3_014_iter_077`, CER `0.163820`
- 구현 성격: 탐색형
- 핵심: N-best 후보 중 다른 후보들과 평균 character-level similarity가 가장 높은 후보를 고른다.
- 의미: score channel의 LM bias를 빼고 후보 간 agreement를 사용한다.
- 한계: N-best 후보가 서로 correlated되어 있으면 새로운 정보가 적다.

### 28) Word-level majority / beam consensus

- 대표 후보: `phase3_014_iter_058`, CER `0.161860`; `phase3_030_iter_026`, CER `0.164439`
- 구현 성격: 탐색형
- 핵심: beam 후보들을 단어/토큰 단위로 align하고 다수결로 일부 span을 바꾼다.
- 의미: rank0의 local error를 beam sibling들이 교정할 수 있다.
- 한계: beam 후보들은 같은 search tree에서 나온 correlated witness라 confident-but-wrong 오류를 같이 공유할 수 있다.

### 29) Independent sampling consensus / ROVER

- 대표 후보: `phase3_014_iter_082`, CER `0.162417`; `phase3_014_iter_083`, CER `0.162940`
- 구현 성격: 탐색형
- 핵심 질문: “한 chunk에 대해 여러 번 디코딩한다는 뜻인가?”
  - 맞다. 같은 30초 chunk에 대해 sampling 후보 여러 개를 만든다.
  - 단, Python for-loop로 4번 호출한 것이 아니라, CT2 `generate()` 한 번에 `num_hypotheses=4`를 요청한다.

코드 값:

``` sourceCode
_N_SAMPLES = 4
_SAMPLE_TEMP = 0.4
_MIN_VOTES = 3

sample_res = generate(
    features,
    [prompt_tokens],
    beam_size=1,
    num_hypotheses=_N_SAMPLES,
    sampling_topk=0,
    sampling_temperature=_SAMPLE_TEMP,
)
```

- beam N-best와 다른 점:
  - beam N-best는 하나의 beam search tree에서 나온 sibling 후보들이다.
  - sampling consensus는 `beam_size=1 + sampling_temperature>0`로 independent draw를 여러 개 만든다.
  - 코드 주석은 `sampling_topk=0`을 full distribution sampling으로 가정한다.

`phase3_014_iter_082` 방식:

- beam backbone text를 authoritative base로 둔다.
- sampling 후보 4개를 만든다.
- `difflib.SequenceMatcher`로 base와 sample들을 align한다.
- base의 `replace` span에 대해 같은 replacement가 3표 이상이면 교체한다.

예시:

``` text
Backbone: 보험료 납임 기간을 확인해 주세요
Sample 1: 보험료 납입 기간을 확인해 주세요
Sample 2: 보험료 납입 기간을 확인해 주세요
Sample 3: 보험료 납입 기간을 확인해 주세요
Sample 4: 보험료 납임 기간을 확인해 주세요

3표 이상: 납임 -> 납입
```

`phase3_014_iter_083` 방식:

- beam 후보 1개 + sampling 후보 4개를 pool로 만든다.

- coverage guard로 너무 짧거나 긴 sampling 후보를 버린다.

- character-level 평균 similarity가 가장 높은 whole hypothesis를 고른다.

- 의미:

  - correlated beam 후보로는 안 보이는 confident-but-wrong commit을 independent sample이 반박할 수 있다.

- 한계:

  - sampling은 hallucination도 만들 수 있다.
  - span-level replacement는 비교적 안전하지만, whole-hypothesis MBR은 평균적이지만 틀린 후보를 고를 수 있다.

------------------------------------------------------------------------

## I. Post-correction / self-consistency 계열

### 30) Majority spelling canonicalization

- 대표 후보: `phase3_014_iter_037`, CER `0.161895`
- 구현 성격: 분리형에 가까움. 강한 decode backbone 이후 text post-correction으로 추가된 형태.
- 핵심: 최종 transcript 자체를 corpus로 보고, 많이 나온 spelling을 canonical anchor로 삼아 드문 variant를 교정한다.
- 코드 조건:

``` sourceCode
_CANON_MIN_COUNT = 4
_RARE_MAX_COUNT = 1
_MIN_WORD_LEN = 4
_MAX_INTERIOR_EDITS = 1
```

조건 해석:

- 4회 이상 등장한 단어만 canonical anchor 가능
- 1회만 등장한 단어만 correction 대상
- 4글자 이상만 대상
- 첫 글자와 마지막 글자는 보존
- 내부 음절 차이가 1개 이하일 때만 교정

예시:

``` text
피보험자 피보험자 피보험자 피보험자 피보헝자
=> 피보헝자 -> 피보험자
```

``` text
보험계약자 보험계약자 보험계약자 보험계약자 보험게약자
=> 보험게약자 -> 보험계약자
```

- 왜 유효한가:
  - 한 통화는 같은 보험/상담 주제를 반복하므로 동일 domain term이 여러 번 나온다.
  - STT 오류는 내부 한 음절만 흔들리는 경우가 많다.
- 한계:
  - 자주 나온 단어가 정답이라는 보장이 필요하다.
  - 짧은 단어/조사에 적용하면 false correction이 크기 때문에 guard가 필요하다.

### 31) Jamo self-consensus / glossary correction

- 대표 후보: `phase3_014_iter_041`, CER `0.160030`; `phase3_014_iter_067`, CER `0.160309`
- 구현 성격: 분리형에 가까움
- 핵심: 한글 syllable을 초성/중성/종성으로 분해해 음운적으로 가까운 오인식을 교정한다.

코드상 한글 분해:

``` sourceCode
_HANGUL_BASE = 0xAC00
_CHO_SPAN = 588  # 21 * 28
_JONG_SPAN = 28

base = ord(ch) - _HANGUL_BASE
cho = base // _CHO_SPAN
jung = (base % _CHO_SPAN) // _JONG_SPAN
jong = base % _JONG_SPAN
```

`phase3_014_iter_041`: glossary 기반 jamo correction

- 고정 glossary 예:

``` text
보험 보험료 보험금 계약 보장 가입 약관 청약 해지 갱신 특약 납입 만기
수익자 피보험자 보험사 상담사 고객님 본인 확인 동의 안내 상품 가입자
```

- 조건:
  - glossary term과 길이가 같음
  - 조사/어미는 보존
  - 딱 한 음절만 다름
  - 다른 음절의 중성, 즉 모음은 같음
  - 초성/종성 같은 자음 계열만 다름

예시:

``` text
피보헙자 -> 피보험자
보헙료를 -> 보험료를
납임 -> 납입
```

`phase3_014_iter_067`: self-consensus jamo correction

- glossary 없이 한 call 안에서 자주 나온 spelling을 canonical로 삼는다.
- 예:

``` text
계약자 계약자 계약자 게약자
=> 게약자 -> 계약자
```

- 의미:
  - 한국어 STT에서는 8kHz 전화 대역에서 자음/받침 혼동이 많다.
  - 일반 edit distance보다 jamo 비교가 더 한국어 음운 오류에 맞다.
- 한계:
  - 모음이 다른 경우까지 고치면 위험하다.
  - domain glossary가 잘못되면 false correction이 생긴다.
  - align posterior나 confidence guard와 결합하면 더 안전하다.

### 32) Per-step logit distribution glossary snap

- 대표 후보: `phase3_014_iter_052`, CER `0.161799`
- 구현 성격: 탐색형
- 핵심: aggregate score가 아니라 token step별 vocab distribution을 사용해 glossary candidate로 snap하려는 시도.
- 의미: 최종 text만 보는 post-correction보다 decoder 내부 evidence를 활용한다.
- 한계: backend return channel에 강하게 의존하고 구현 난이도가 높다.

------------------------------------------------------------------------

## J. Audio / feature frontend 계열

### 33) Loudness / RMS normalization

- 대표 후보: `phase3_030_iter_018`, CER `0.163271`; `phase3_014_iter_009`, CER `0.171497`
- 구현 성격: 탐색형
- 핵심: 조용한 speech frame의 amplitude/RMS를 보강한다.
- 의미: 낮은 볼륨으로 인한 deletion/substitution을 줄이려는 시도.
- 한계: 사람 목소리뿐 아니라 noise도 같이 커진다. Whisper feature extractor가 일부 normalization을 이미 수행하므로 효과가 제한될 수 있다.

### 34) Pre-emphasis / high-pass / spectral shaping

- 대표 후보: `phase3_013_iter_026`, CER `0.158575`; `phase3_014_iter_013`, CER `0.170747`
- 구현 성격: 조합형
- 핵심: 고역/자음 cue를 강조해 consonant confusion을 줄이려는 waveform-level 전처리.
- 의미: 전화/회의 음성에서 자음 구분이 약한 경우 유효할 수 있다.
- 한계: 치찰음, 기계음, 잡음도 함께 강조될 수 있다.

### 35) Feature-domain CMN / mel ramp / unsharp / variance equalization

- 대표 후보:
  - `phase3_014_iter_022` CMN, CER `0.215926`
  - `phase3_014_iter_031` mel ramp, CER `0.177561`
  - `phase3_014_iter_035` frame DRC, CER `0.189159`
  - `phase3_014_iter_036` mel unsharp, CER `0.186702`
  - `phase3_014_iter_055` temporal unsharp, CER `0.184349`
  - `phase3_014_iter_063` variance EQ, CER `0.165598`
- 구현 성격: 탐색형
- 질문 핵심: “mel단에서 조작했다는 뜻인가?”
  - 맞다. waveform을 바꾸는 것이 아니라, `processor(...)`가 만든 `inputs.input_features`, 즉 Whisper encoder 입력 log-mel feature를 직접 바꾼다.

공통 흐름:

``` sourceCode
inputs = processor(chunk, sampling_rate=sr, return_tensors="np")
feats = inputs.input_features
feats = modify_logmel_features(feats)
features = to_storage_view(feats)
generate(features, ...)
```

세부 기법:

1.  CMN, cepstral/channel mean normalization에 가까운 방식

``` sourceCode
bin_mean = feats.mean(axis=-1, keepdims=True)
global_mean = feats.mean()
feats = feats - bin_mean + global_mean
```

- 목적: 전화 채널 coloration 제거
- 결과: `0.215926`, 좋지 않음

2.  Mel ramp

- 상위 mel bin에 ramp를 더해 고주파 쪽을 인위적으로 올린다.
- 목적: 자음 cue 보강
- 결과: `0.177561`

3.  Frame dynamic range compression/expansion

- frame level energy를 보고 조용한 speech frame을 들어 올린다.
- 결과: `0.189159`

4.  Mel unsharp

``` sourceCode
F_sharp = F + alpha * (F - smooth_mel(F))
```

- mel frequency axis의 contrast를 강화한다.
- 결과: `0.186702`

5.  Temporal unsharp

``` sourceCode
F_sharp = F + alpha * (F - smooth_time(F))
```

- 시간축 transient, 자음 시작점 같은 변화를 강조한다.
- 결과: `0.184349`

6.  Variance equalization

``` sourceCode
std = feats.std(axis=time)
gain = target / std
gain = clip(gain, 1.0, 3.0)
feats *= gain
```

- dynamic range가 죽은 mel bin을 살린다.
- 결과: `0.165598`, feature-domain 조작 중 가장 양호.

해석:

- 매우 참신하지만, Whisper가 학습한 log-mel feature 분포를 직접 깨뜨리는 위험이 있다.
- 단독 적용보다 raw/witness branch로 쓰는 편이 안전하다.

### 36) VTLN / acoustic perturbation ensemble

- 대표 후보: `phase3_014_iter_064`, CER `0.162139`; `phase3_030_iter_076~080`, CER 약 `0.171~0.174`
- 구현 성격: 탐색형
- 핵심: feature warp / formant shift 등 다른 acoustic view로 decode하고 원본 결과와 비교한다.
- 의미: speaker/channel mismatch에 대한 witness view.
- 한계: warp view 자체를 최종으로 믿으면 위험. raw decode 기본 + witness 교정이 더 안전하다.

### 37) Artificial bandwidth extension / nonlinear frontend

- 대표 후보: `phase3_014_iter_071`, CER `0.176864`
- 구현 성격: 탐색형
- 핵심 질문: “전화 대역에서 사라진 고역을 복원한다는 게 무슨 뜻인가?”
  - 실제로 사라진 정보를 복원한 것은 아니다.
  - full-wave rectification 같은 nonlinear operation으로 harmonic 성분을 만들어, high-band 비슷한 witness signal을 만든다.

코드 구조:

``` sourceCode
exc = np.abs(x)
exc = highpass_or_smooth_remove(exc)
y = x + beta * exc
```

코드 파라미터:

``` sourceCode
_ABE_BETA = 0.6
_ABE_SMOOTH_MS = 1.0
```

해석:

- 원본 전화 음성은 대략 0.3-3.4kHz 대역에 제한된다.
- `abs(x)`는 비선형 연산이라 sum/difference harmonic 성분을 만들 수 있다.
- 느린 envelope를 제거한 뒤 원본에 섞어 자음 cue 비슷한 신호를 추가하려는 시도다.

적용 방식:

- 원본 beam decode가 backbone이다.
- artificial bandwidth-extended audio는 witness로만 decode한다.
- `difflib`로 content를 align한 뒤, equal-length replace span만 witness로 교체한다.

예시:

``` text
Backbone: 보험료 납임 기간
ABE view: 보험료 납입 기간
교체: 납임 -> 납입
```

한계:

- 실제 학습 기반 bandwidth extension이 아니라 rule-based signal trick이다.
- CER `0.176864`로 최고권은 아니었다.
- 그래도 “새로운 acoustic view를 witness로만 사용한다”는 설계는 의미 있다.

------------------------------------------------------------------------

## K. Segmentation ensemble / overlap / seam 계열

### 38) Phase-shifted dual-grid decoding

- 대표 후보: `phase3_014_iter_068`, CER `0.161860`; `phase3_014_iter_026`, CER `0.167410`
- 구현 성격: 탐색형
- 핵심 질문: “30초/25초 이런 식인가? seam이 뭐지?”
  - seam은 window 경계다.
  - 예: `0~30s | 30~60s | 60~90s`에서 30초, 60초 지점이 seam이다.
  - 어떤 단어가 seam에 걸리면 앞 window에서는 뒤가 부족하고, 뒤 window에서는 앞이 부족하다.

기본 grid:

``` text
Grid A: 0~30s, 30~60s, 60~90s
```

15초 shift grid:

``` text
Grid B: 15~45s, 45~75s, 75~105s
```

코드 파라미터:

``` sourceCode
_PHASE_SHIFT_SECONDS = 15.0
_CHUNK_SECONDS = 30.0
_SEAM_RADIUS_CHARS = 14
```

의도:

``` text
Grid A에서 seam에 걸린 단어가 Grid B에서는 window 중앙에 오도록 만든다.
```

예시:

``` text
Grid A:
0~30 | 30~60
... 보험금 청구서류를 ...
          ^ 30초 seam

Grid B:
15~45
청구서류가 window 중앙부에 위치
```

reconciliation 방식:

- Grid A는 authoritative backbone이다. `beam_size=5`.
- Grid B는 witness다. `beam_size=1`.
- 두 transcript를 `difflib`로 align한다.
- Grid A의 seam 근처 replace span만 Grid B text로 교체한다.

코드 주석상 설계:

``` sourceCode
# A Grid-A 'replace' span is rescued by Grid B only if it begins/ends within
# _SEAM_RADIUS_CHARS of a Grid-A seam.
```

의미:

- 전체 transcript를 ensemble하는 것이 아니라 seam-local repair에 제한했다.
- window boundary 문제를 겨냥한 구조적 방법이다.

한계:

- text-level diff alignment가 꼬이면 잘못된 span을 바꿀 수 있다.
- 비용은 대략 두 grid decode에 가까워진다. 다만 witness grid는 greedy라 비용을 줄였다.

### 39) Overlap window + stitch

- 대표 후보: `phase3_030_iter_104`, CER `0.290896`; `phase3_014_iter_090`, CER `0.478722`
- 구현 성격: 실패/탐색형
- 핵심: overlap 구간을 두고 text/token matching으로 window를 이어 붙인다.
- 의미: seam 문제를 직접 다루려는 가장 직관적인 방법.
- 실패 원인 추정: overlap text matching이 한국어 발화/Whisper timestamp와 잘 맞지 않아 deletion/duplication이 발생했을 가능성이 높다.

### 40) Central-band commit / left-context run-up

- 대표 후보:
  - `phase3_014_iter_057`, central-band commit, CER `0.494249`
  - `phase3_014_iter_061`, left-context run-up, CER `0.410435`
- 구현 성격: 실패/탐색형

#### 40-1. Central-band commit

아이디어:

- 각 30초 window의 edge는 불안정하다고 보고 중앙부만 commit한다.
- window는 30초, hop은 20초다.
- 즉 10초 overlap이 있고, 양쪽 5초를 context로 둔다.

코드 파라미터:

``` sourceCode
_CHUNK_SECONDS = 30.0
_HOP_SECONDS = 20.0
_CONTEXT_SECONDS = 5.0
```

window 예시:

``` text
window 1: 0~30s   -> central commit 대략 0~25s 또는 5~25s
window 2: 20~50s  -> central commit 25~45s
window 3: 40~70s  -> central commit 45~65s
```

의도:

``` text
모든 commit text가 window edge가 아니라 중앙에서 decode되도록 만들자.
```

실패 원인 추정:

- 구현은 timestamp-delimited segment의 start time을 기준으로 commit band 포함 여부를 판단한다.
- Whisper segment start가 부정확하거나 segment가 길게 묶이면, 실제로 필요한 text가 band 밖으로 빠져 삭제될 수 있다.
- 그래서 CER가 `0.494249`로 크게 악화되었다.

#### 40-2. Left-context run-up

아이디어:

- 다음 window를 정확히 `seek`부터 시작하지 않고, `seek - 5초`부터 decode한다.
- 앞 5초는 context용으로만 사용하고, output에서는 discard한다.

코드 파라미터:

``` sourceCode
_RUNUP_SECONDS = 5.0
_CHUNK_SECONDS = 30.0
```

예시:

``` text
원래:
seek = 30s
window = 30~60s

left-runup:
seek = 30s
decode buffer = 25~55s
25~30s text는 버리고 30s 이후만 commit
```

의도:

- 30초 경계 바로 뒤의 단어가 cold-start로 인식되는 문제를 줄인다.
- decoder가 앞 문맥을 조금 본 상태에서 boundary 이후를 decode하게 한다.

실패 원인 추정:

- discard 기준이 timestamp에 의존한다.
- timestamp가 정확하지 않으면 버리면 안 되는 text까지 버리거나, 이미 나온 부분을 중복 emit할 수 있다.
- 결과 CER `0.410435`로 좋지 않았다.

------------------------------------------------------------------------

## 5. Ablation Confidence 정리

### 5.1 순수 기여도 확인이 어려운 강한 조합형

아래 기법들은 상위 후보에 들어 있지만, 여러 기법이 함께 포함되어 있어 단독 효과를 말하면 안 된다.

| 기법                         | 대표 후보                    |                 CER | 해석                                            |
|------------------------------|------------------------------|--------------------:|-------------------------------------------------|
| per-window language override | `phase3_013_iter_022`        |            0.153887 | 최고 후보에 포함. 단독 효과 별도 ablation 필요  |
| beam/N-best                  | `phase3_013_iter_011`, `022` | 0.158801 / 0.153887 | substitution axis에 강하지만 다른 gate와 결합됨 |
| align tail trim              | `phase3_013_iter_014`, `016` | 0.155185 / 0.154828 | acoustic verification이 유망하나 backbone 포함  |
| selective notimestamps       | `phase3_013_iter_013`        |            0.155107 | timestamp seek 계열과 결합됨                    |
| temperature fallback         | `phase3_004_iter_018`        |            0.157355 | score/compression gate와 결합됨                 |
| CJK/Kana suppress            | `phase3_013_iter_017`        |            0.156762 | strong backbone 위에 포함됨                     |

### 5.2 비교적 분리된 post-correction 계열

| 기법                               | 대표 후보             |      CER | 해석                                      |
|------------------------------------|-----------------------|---------:|-------------------------------------------|
| majority spelling canonicalization | `phase3_014_iter_037` | 0.161895 | decoded text 후처리 stage라 비교적 분리됨 |
| glossary jamo correction           | `phase3_014_iter_041` | 0.160030 | glossary 기반 post-correction stage       |
| self-consensus jamo correction     | `phase3_014_iter_067` | 0.160309 | 통화 내부 frequent spelling 기반          |

### 5.3 실패했지만 연구 가치가 있는 계열

| 기법                           | 대표 후보             |      CER | 실패 의미                           |
|--------------------------------|-----------------------|---------:|-------------------------------------|
| central-band commit            | `phase3_014_iter_057` | 0.494249 | coverage/commit boundary가 깨짐     |
| left-context run-up            | `phase3_014_iter_061` | 0.410435 | discard 기준이 timestamp에 민감     |
| overlap stitch                 | `phase3_030_iter_104` | 0.290896 | text-level stitching이 불안정       |
| alignment field seek           | `phase3_004_iter_044` | 0.312410 | align 기반 seek 변환이 불안정       |
| artificial bandwidth extension | `phase3_014_iter_071` | 0.176864 | 참신하지만 rule-based frontend 한계 |

------------------------------------------------------------------------

## 6. 다음 ablation 계획

제공 자료만으로는 순수 기여도를 분리하기 어렵다. 다음 단계에서는 champion backbone 하나를 고정하고, 기법을 하나씩 on/off하는 ablation matrix가 필요하다.

권장 baseline:

``` text
B0 = timestamp seek + beam_size=5 + score/compression gate + RMS/silence cut
```

권장 ablation 축:

| 축                 | 실험                                                              |
|--------------------|-------------------------------------------------------------------|
| language override  | B0 vs B0 + detect_language override vs B0 + ambiguous dual-branch |
| script suppress    | B0 vs B0 + CJK/Kana suppress                                      |
| align verification | B0 vs B0 + align tail trim vs B0 + align N-best rerank            |
| glossary           | B0 vs static glossary vs glossary branch + margin                 |
| post-correction    | B0 vs majority spelling vs jamo glossary vs jamo self-consensus   |
| sampling consensus | B0 vs beam consensus vs independent sampling span vote            |
| seam repair        | B0 vs phase-shifted dual-grid seam-local repair                   |
| frontend witness   | B0 vs pre-emphasis witness vs VTLN witness vs ABE witness         |

각 실험에서 반드시 기록할 metric:

``` text
CER
substitution ratio
deletion ratio
insertion ratio
length ratio
hallucination hit rate
repeated text rate
audio coverage rate
runtime / RTF
GPU memory
```

------------------------------------------------------------------------

## 7. 결론

이 실험 기록의 핵심 가치는 “최고 CER 하나”보다, autoresearch가 실제 코드 후보를 통해 STT long-form decoding의 넓은 설계 공간을 탐색했다는 데 있다.

특히 의미 있는 발견 축은 다음이다.

1.  Whisper/CT2의 return channel을 적극적으로 사용했다.
    - `scores`, `no_speech_prob`, `sequences_ids`, timestamp token, align posterior, language posterior
2.  장문 전사의 핵심 문제를 windowing/seek/commit 문제로 재정의했다.
3.  substitution-dominant 구간에서 beam/N-best/align/glossary/language override를 결합했다.
4.  post-correction을 단순 문자열 치환이 아니라 self-consistency/jamo/acoustic guard 문제로 접근했다.
5.  실패한 후보들도 seam, context, frontend witness 등 후속 연구 축을 남겼다.

단, 현재 자료만으로는 대부분의 기법에 대해 “단독으로 CER를 얼마 개선했다”고 말할 수 없다. 가장 정확한 표현은 다음이다.

> 이 문서의 CER 수치는 각 기법이 구현된 후보의 성능 기록이다.  
> 순수 기여도는 별도 ablation이 필요하다.
