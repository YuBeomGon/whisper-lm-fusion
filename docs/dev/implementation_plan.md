# whisper-lm-fusion 구현 플랜

**일자**: 2026-06-09

이 문서는 `whisper-lm-fusion` 1차 구현의 작업 플랜 기록이다. 현재 정본 계약은
[`docs/SSOT.md`](../SSOT.md)를 따른다.

> **기본 방침**: 검증된 long-form 디코딩 동작은 유지하되, 파일 구조와 API는
> generic wrapper 원칙대로 설계한다.
> 로직은 처음부터 **확장 가능한 형태**로 둔다 (decoding_strategy의 후속 전략을 나중에 끼울 수 있게).

관련 문서:
- `docs/SSOT.md` — 정본 계약 / 공개 전 체크리스트
- `docs/design.md` — interface / 책임 경계
- `docs/archive/principles.md` — 설계 원칙 (transcribe + 파라미터 표면화)
- `docs/research/decoding_strategy.md` — 내부 후보 전략 노트

---

## 1. 목표 / 1차 범위

**1차 (thin)**: `load()` + `transcribe()` 최소 API, long-form 내재화, tokenizer_hash 검증.

**비범위(후속)**: batching, VAD, segment merge, word timestamp, diarization. 처음부터 크게 만들지 않는다.

---

## 2. 베이스로 가져오는 디코딩 로직

검증된 long-form backbone을 코드 그대로 복붙하지 않고 wrapper 구조에 맞게 재구성한다.
구현 단위마다 **확장 포인트**를 열어 둔다.

| 베이스 로직 | 현재 구현 단위 | 확장 포인트 |
|---|---|---|
| **30초 window 루프** | `engine.py` | window/chunk 분할을 교체 가능한 segmenter로 |
| **RMS silence cut** | `_decode.py` | cut 결정 함수를 분리 (VAD/RMS-trough 교체 대비) |
| **timestamp adaptive seek** | `_decode.py` | seek 정책을 hook으로 |
| **N-best gate** | `_decode.py` | 후보 선택기를 strategy로 분리 (axis-aware/MBR/align rerank 후속) |

> 이 4개는 검증된 backbone이라 동작은 유지한다. 단, **하드코딩이 아니라 교체 가능한 단위**로 둬서
> decoding_strategy의 후속 전략(language override, align trim, glossary margin 등)을 나중에 끼운다.

---

## 3. 확장성 설계 원칙

- **파라미터 표면화** — 디코딩 옵션은 `transcribe(audio, *, beam_size=..., alpha=..., ...)` 인자로 노출 (principles §1).
- **메커니즘은 숨김** — window 루프/seek/cut/선택기는 내부. 사용자는 파라미터만.
- **교체 가능 단위** — segmenter / cut 결정 / seek 정책 / 후보 선택기를 함수(또는 작은 hook)로 분리해, 내부 알고리즘이 바뀌어도 API는 안정.
- **합리적 기본값** — 아무 것도 안 줘도 동작. 기본값은 decoding_strategy 정본값 사용.
- **generic 경계** — 도메인 기본값(특정 `.binary` 경로 등) 제거. path/파라미터만 받는다.

---

## 4. 현재 API 표면

정본은 `docs/SSOT.md`와 README를 따른다.

```python
# (B) serving init — 부팅 1회
engine = whisper_lm_fusion.load(
    model_path,
    lm_path=None,            # 없으면 fusion off, 일반 STT로 동작
    device="cuda",
    compute_type="float16",
    # fusion 기본값 (요청에서 override 가능)
    alpha_default=0.0,
    topk_default=50,
    fusion_mode="topk",
)

# (C) per request — 디코딩 옵션은 파라미터로
result = engine.transcribe(
    audio, sr, *,
    # fusion
    alpha=0.0, topk=50, lm_enabled=False,
    # search
    beam_size=5, num_hypotheses=5, patience=2.0,
    sampling_temperature=0.0,
    # gate (선택)
    # ...
    # output opt-in
    return_segments=False,
)   # -> TranscriptionResult
```

- `load()` = model/tokenizer 보관(부팅 1회). `transcribe()`는 매 요청 재로드 X.
- 결과는 `text` 기본, 현재 구현된 opt-in 결과는 `segments`.
- `return_scores` / `return_nbest`는 config 표면에 있으나 구현 완료 전까지 공개 API로 홍보하지 않는다.

---

## 5. 모듈 구조 (우리가 정함)

외부 runner의 파일 분할을 따르지 않는다. wrapper에 맞게 새로 구성.

- 처음엔 작게 시작 (단일 패키지 `src/whisper_lm_fusion/`).
- 교체 가능 단위는 **함수 경계로** 분리하되, 파일 과분할은 피한다.
- 현재: `engine.py`(load/transcribe), `config.py`(파라미터 스키마), `backends/`(CT2 load/generate), `_decode.py`(순수 디코딩 helper).

> 구체 파일 분할은 정본 계약보다 낮은 수준의 구현 디테일이다.

---

## 6. generic화 / 검증 게이트

- **tokenizer_hash / ct2_model_hash 검증** — `load()`에서 LM metadata가 현재 모델과 다르면 LM 로드 거부 (design.md §3). token-id mismatch는 조용히 품질만 깎으므로 게이트.
- **도메인 제거** — 특정 corpus/`.binary` 기본 경로, domain term, glossary 내용은 wrapper에 두지 않는다.
- **LM optional** — `lm_path=None`이면 fusion off로 정상 동작.

---

## 7. 작업 순서

1. backend 정리 — CT2 Whisper load / processor / generate / storage_view (lazy import).
2. config 스키마 — load 시점 vs request 시점 파라미터 분리.
3. `load()` — model/tokenizer 보관 + metadata hash 게이트.
4. `transcribe()` — window 루프 / RMS cut / timestamp seek / N-best gate를 **교체 가능 단위로** 구성, 파라미터 표면화.
5. output 옵션 — text 기본 + segments opt-in. scores/nbest는 후속 구현.
6. smoke test — 아래 검증 기준.

---

## 8. 검증 기준

- **fusion off = baseline 동일** — `lm_enabled=False` 또는 `alpha=0`일 때 baseline CT2와 출력 텍스트 일치 (design.md §7). 회귀 테스트로 고정.
- **smoke test** — 짧은 오디오 1개가 `load()` → `transcribe()`로 text를 반환.
- **LM 없는 빌드 동작** — KenLM optional dependency로 import 실패해도 일반 STT 동작.

---

## 9. 비범위 (후속)

batching, VAD, segment merge, word timestamp, diarization, 그리고 decoding_strategy의
고급 전략(language override / align trim / glossary margin / MBR rerank 등).
→ 3절의 교체 가능 단위로 **나중에 끼울 수 있게** 자리만 열어 둔다.
