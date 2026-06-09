# whisper-lm-fusion 설계 원칙 & OSS 준비

**일자**: 2026-06-09

이 문서는 `whisper-lm-fusion`를 오픈소스로 공개하기 위한 **핵심 원칙**만 정리한다.
정본 계약과 공개 전 체크리스트는 [`docs/SSOT.md`](../SSOT.md)를 따른다.

관련 문서:
- `docs/SSOT.md` — 정본 계약 / OSS readiness
- `docs/design.md` — interface / 책임 경계
- `docs/research/decoding_strategy.md` — 내부 후보 전략 노트

---

## 1. 핵심 원칙: faster-whisper처럼 "하나의 transcribe + 풍부한 파라미터"

이 라이브러리의 본질이자 제품 그 자체.

> **long-form 디코딩의 복잡성은 안으로 숨기고, `transcribe(audio, **params)` 하나로 모든 걸 조절하게 한다.**

- 사용자는 내부 window 루프 / seek / fallback / align 을 **몰라도** 된다.
- 사용자는 **파라미터만 바꿔서** 품질·속도·도메인 적응을 튜닝한다.
- 노출하는 것 = 파라미터. 숨기는 것 = 메커니즘.

이는 단순 관례가 아니라 `design.md`의 철학과 일치한다:
**wrapper = "audio + params -> text", 도메인은 모른다.**

### 1.1 따라오는 규칙

| 규칙 | 의미 |
|---|---|
| 노출은 **파라미터**로 | 디코딩 전략(beam/patience/fallback/fusion 등)은 transcribe 인자로 표면화 |
| 숨김은 **메커니즘** | window 루프, seek 로직, align 호출 시점 등 내부 구현은 감춤 |
| **합리적 기본값** | 아무 것도 안 줘도 동작. 필요한 것만 override |
| **점진적 노출** | 1차엔 핵심 파라미터만, 고급 옵션은 후속 추가 (시그니처 안정성 유지) |
| **generic 경계** | glossary 내용·domain term·corpus는 미포함. wrapper는 받아서 적용만 |

> 내부 알고리즘이 정해지지 않아도 이 원칙은 유효하다. 내부가 바뀌어도
> **파라미터 표면(parameter surface)은 안정적으로 유지**하는 것이 목표다.

---

## 2. 쓰기 좋게 만드는 것 (중요한 것만)

다른 사람이 막힘없이 쓰게 하는 최소 조건.

- **빠른 시작 5줄** — README 맨 위, 복붙 가능한 `load` → `transcribe` 예제.
- **설치 한 줄** — `pip install whisper-lm-fusion`. CTranslate2 backend는 `[ct2]` extra로 분리.
- **fusion 없이도 동작** — patched CT2/LM 없으면 일반 STT로 그냥 돌아가게. 첫 사용이 막히지 않도록.
- **타입 힌트 + docstring** — 파라미터 단위·범위 명시(`alpha 0~1`). IDE 자동완성으로 문서 대체.
- **친절한 에러** — tokenizer_hash mismatch 등 거부 시 "왜/어떻게 고치는지"까지. 조용한 실패 금지.
- **결과는 opt-in** — `text` 기본, 현재 공개 구현은 `segments`만 opt-in. `scores/nbest`는 구현 후 공개 문서에 올린다.

---

## 3. OSS 공개에 꼭 필요한 것

공개 전 체크리스트는 [`docs/SSOT.md` §6](../SSOT.md#6-open-source-readiness-checklist)에 둔다.
여기에는 원칙만 남긴다.

- `README.md`는 설치, quick start, 현재 지원 API, 한계만 빠르게 보여준다.
- `LICENSE`, CI, 패키지 metadata는 GitHub 공개 전에 반드시 맞춘다.
- 모델·오디오·KenLM `.binary`는 레포에 넣지 않는다.
- 내부 실험 로그는 공개 문서와 분리하거나 sanitize한다.

---

## 4. 지금 결정된 것 / 안 된 것

| 결정됨 | 미결정 |
|---|---|
| transcribe + 파라미터로 노출하는 **원칙** | `return_scores` / `return_nbest` 구현 여부 |
| 메커니즘은 숨긴다는 **경계** | patched CTranslate2 공개 후 설치 문서 |
| 현재 API의 1차 범위 | LICENSE 종류, 패키지명, GitHub URL |

다음 단계 후보: `docs/SSOT.md` 기준으로 README와 패키지 metadata를 공개용으로 고정한다.
