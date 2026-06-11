# Pipeline — 디코딩 동작 설명

`whisper-lm-fusion`의 `transcribe()`가 오디오 한 개를 처리하는 **실제 내부 흐름**과,
각 단계를 어떤 파라미터로 바꾸는지를 운영자 관점에서 설명한다.

- 파라미터 전체 레퍼런스: [`parameters.md`](parameters.md)
- sweep용 control surface 계약: [`../research/pipeline_control_surface.md`](../research/pipeline_control_surface.md)
- 정본 계약/경계: [`../SSOT.md`](../SSOT.md)

> 이 파이프라인은 faster-whisper 복제가 아니라 phase3 self-evolve에서 검증된 전략을
> 제품형으로 압축한 것이다. 사용자는 코드를 고치지 않고 파라미터로 내부 로직을 조합한다.

---

## 1. 한눈에 보는 흐름

`transcribe()`는 긴 오디오를 윈도우 단위로 순회하며, 윈도우마다 아래 단계를 돈다.

```
[전처리, 1회]
  · 파형 정규화(mono float32) + 프레임 RMS 에너지 계산
  · suppress_cjk_kana=True면 vocab 스캔으로 CJK/Kana suppress 토큰 1회 구축(캐시)

[윈도우 루프, seek < len(audio) 동안 반복]
  1. silence/RMS cut       decide_cut → (cut, silence_cut)
  2. feature extract       backend.extract_features + use_timestamps 결정
  3. language policy        _language_decision → primary / secondary 언어
  4. prompt + context       _build_prompt (언어별 SOT + <|startofprev|> context)
  5. CT2 generate (N-best)  backend.generate  ← 여기서 LM fusion kwargs 주입
  6. candidate 생성         make_candidates (text/logprob/degenerate/no_speech)
  7. selection + gate       hypothesis_from_candidates
                              = select_candidate(selection_policy) + no-speech drop
  8. conditional fallback   should_fallback → temperature_fallback로 재디코드
  9. 후보 재선택            누적 후보 전체로 재선택
 10. optional align trim     _maybe_align_trim (trigger 충족 시만)
 11. context update          _update_context (context_policy)
 12. seek advance            advance_seek (silence_cut/timestamp 기반)

[종료] 윈도우별 텍스트를 이어붙여 TranscriptionResult 반환
```

---

## 2. 단계별 상세 + 파라미터 매핑

### 1) Silence / RMS cut — 어디서 자를지
프레임 RMS로 윈도우 꼬리의 저에너지(pause) 지점을 찾아 문장 중간 절단을 피한다. VAD 모델은
쓰지 않는다(가벼운 RMS만).
- `window_seconds`(30) · `min_advance_seconds`(20) · `silence_percentile`(20) · `timestamp_resolution`(0.02)
- pause를 못 찾으면 윈도우 끝(`silence_cut=False`)에서 자른다.

### 2) Feature extract
백엔드가 청크를 log-mel 등 네이티브 feature로 변환. 마지막 청크가 아니고 silence cut도
아니면 `use_timestamps=True`로 timestamp 토큰을 출력시켜 seek에 활용한다.

### 3) Language policy — 어떤 언어 토큰으로 디코드할지
- `language_policy="fixed"` (기본): `language`(예 `ko`) 고정.
- `"per_window_confident"`: `backend.detect_language` posterior가 `language_override_prob`(0.7)
  이상이면 그 언어로 교체.
- `"dual_band"`: posterior가 `[dual_language_low_prob, dual_language_high_prob)` 애매 구간이면
  **primary + 감지 언어 두 갈래로 각각 디코드**해 후보를 한 풀에 합친다.

### 4) Prompt + context
SOT(`<|lang|><|task|>`) 구성, `context_policy != "off"`면 직전 윈도우 텍스트를
`<|startofprev|>` 뒤에 `max_context_tokens`(200)까지 붙인다.

### 5) CT2 generate (N-best)
`backend.generate`로 beam/N-best 디코드. **LM fusion이 켜져 있으면 여기서**
`lm_fusion_*` kwargs가 함께 들어간다(켜짐 조건은 [`fusion.md`](fusion.md)).
- `beam_size`(5) · `num_hypotheses`(5) · `patience`(2.0) · `repetition_penalty` · `no_repeat_ngram_size`
- ⚠️ `num_hypotheses`는 백엔드에서 `beam_size`로 클램프된다(둘 다 5면 무해).

### 6) Candidate 생성
각 후보를 `(token_ids, text, logprob, degenerate, no_speech_prob, source, language)`로 정규화.
`degenerate`는 gzip 압축비 > `compression_ratio_threshold`(2.4)로 반복/degenerate 감지.

### 7) Selection + gate — 어느 후보를 고를지
`selection_policy`가 내부 선택 로직을 바꾼다.
- `"axis_aware"`(기본): degenerate 후보 배제 후 logprob 최대.
- `"logprob"`: 순수 점수.
- `"longer_within_margin"`(또는 `prefer_longer_within_margin=True`): best logprob의
  `score_margin`(0.10) 안이면서 `min_length_ratio_for_longer`(1.05) 이상 긴 clean 후보 채택
  → deletion 보정.
- `"token_mbr"`: clean N-best의 토큰 medoid 선택.
- 이어서 no-speech 게이트: `no_speech_prob`가 높고 `no_speech_logprob_threshold` 미만이면
  텍스트를 drop(`dropped_no_speech`).

### 8) Conditional fallback — 실패 윈도우만 재디코드
`fallback_policy`가 off가 아니고 선택 후보가 게이트를 못 넘으면, `temperature_fallback`
(예 `(0.2,0.4,0.6)`) 사다리로 재디코드한다. 매 윈도우가 아니라 **실패한 윈도우만**.
- 트리거: `off` / `gate_fail` / `low_logprob` / `degenerate` / `always`
- `fallback_sampling_topk`(0이면 `sampling_topk` 사용)

### 9) 후보 재선택
fallback 후보를 기존 후보와 **합친 전체 풀**에서 7)의 정책으로 다시 고른다.

### 10) Optional align trim
`align_tail_trim=True`이고 트리거(`align_trigger_low_logprob`/`align_trigger_degenerate`)가
충족되면, 백엔드 align posterior로 신뢰 낮은 꼬리를 잘라낸다. 백엔드/빌드 미지원 시 no-op.
- `align_prob_floor`(0.3) · `align_min_run`(8)

### 11) Context update
`context_policy`에 따라 다음 윈도우로 넘길 context를 갱신.
- `"off"`: 안 넘김. `"always"`: emit된 텍스트면 항상 넘김.
  `"confidence_gated"`(기본): logprob 양호 & 비-degenerate일 때만 넘김(오류 전파 방지).

### 12) Seek advance
다음 시작 위치 결정. silence cut이면 cut 지점으로, 아니면 마지막 timestamp 토큰 기반
적응 전진(`min_advance_seconds` 가드).

---

## 3. 출력

`TranscriptionResult.text`는 항상. opt-in으로:
- `return_segments=True` → `.segments` (윈도우별 start/end/logprob/no_speech)
- `return_scores=True` → `.scores`
- `return_nbest=True` → `.nbest` (윈도우별 후보 목록, `source`/`language` 포함)

---

## 4. 운영 팁

- **기본값 = 보수적 backbone**. 끄면(전 정책 기본) 안정적 baseline. 강한 옵션은 opt-in.
- 빠른/조밀 발화 deletion이 많으면: `longer_within_margin` + `fallback_policy="gate_fail"`.
- 무음 hallucination이 많으면: `no_speech_threshold`/`no_speech_logprob_threshold` 조정.
- 일본어 kana/한자 튐: `suppress_cjk_kana=True`(기본) 유지 — 한글/영문/숫자는 보존.
- 어떤 조합이 좋은지는 **별도 sweep runner**가 결정(라이브러리 밖). 권장 그룹은
  [`../research/pipeline_control_surface.md`](../research/pipeline_control_surface.md) §3.
