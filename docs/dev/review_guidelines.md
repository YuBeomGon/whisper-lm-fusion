# whisper-lm-fusion 리뷰 관점

**일자**: 2026-06-11

이 문서는 현재 프로젝트 목적에 맞춰 코드와 문서를 리뷰할 때 볼 관점을 정리한다.
정본 계약은 [`docs/SSOT.md`](../SSOT.md)를 따른다.

## 1. 프로젝트 목적 정합성

이 프로젝트의 목적은 faster-whisper와 **사용법/API만** 유사하게 제공하면서, **내부 로직은
복제가 아닌 자체** STT decoding pipeline과 optional LM fusion을 도입하는 것이다.

리뷰할 때는 먼저 다음을 확인한다.

- 사용자는 `load()` / `transcribe()` 중심의 익숙한 API로 쓸 수 있는가?
- 내부 동작은 faster-whisper 복제가 아니라 자체 pipeline으로 구성되어 있는가?
- LM fusion은 baseline STT와 분리되어 on/off 가능한 선택 기능인가?

## 2. Public API 관점

- `load()`와 `Engine.transcribe()`의 시그니처가 README, SSOT, 실제 코드에서 일치하는가?
- faster-whisper와 비슷한 옵션과 이 프로젝트 고유 옵션이 혼동 없이 구분되는가?
- 기본값만으로 plain STT가 동작하고, 추가 파라미터로 pipeline 동작을 바꿀 수 있는가?
- 아직 안정화되지 않은 옵션을 완료된 공개 기능처럼 문서화하지 않았는가?

## 3. Pipeline 로직 관점

현재 pipeline은 engine 내부에 있어도 괜찮다. 중요한 것은 단계와 책임이 읽히는지이다.

- windowing, silence cut, timestamp seek가 의도한 순서로 연결되는가?
- prompt 구성, language policy, context carry가 옵션에 따라 예측 가능하게 바뀌는가?
- N-best 선택, fallback, no-speech gate가 각각 독립적으로 테스트 가능한가?
- 정책 enum(`selection_policy` / `fallback_policy` / `language_policy` / `context_policy`)이
  값에 따라 의도한 내부 로직으로 분기하는가?
- 임계값과 정책이 코드에 숨은 하드코딩이 아니라 `DecodeOptions`로 표면화되어 있는가?

## 4. LM Fusion 관점

LM fusion은 pipeline 전체를 바꾸는 기능이 아니라, 각 backend `generate()` 호출에 주입되는
optional decode option이어야 한다.

- `lm_enabled=False` 또는 `alpha <= 0`이면 fusion kwargs가 전달되지 않는가?
- `lm_enabled=True`일 때만 `lm_fusion_model_path`, `lm_fusion_alpha`,
  `lm_fusion_asr_topk`, `lm_fusion_debug`가 backend로 내려가는가?
- `lm_path`, `alpha_default`, `topk_default`의 load-time 기본값과 request-time override가
  명확히 분리되어 있는가?
- KenLM metadata 검증 실패가 조용한 품질 저하가 아니라 명확한 오류로 드러나는가?

## 5. 문서와 코드 일치성

- README는 사용법과 목적을 간단히 설명하고, 상세 정책은 SSOT로 연결하는가?
- SSOT는 책임 경계, public API, fusion 계약, 미완성 기능을 정확히 말하는가?
- `docs/dev/` 문서는 구현 의도와 리뷰 기준을 남기되 정본 정책을 중복 정의하지 않는가?
- 문서 예시의 파라미터 이름과 실제 `DecodeOptions` / `load()` 인자가 일치하는가?

## 6. 테스트 관점

- CT2 없이 fake backend로 wrapper pipeline을 검증하는 테스트가 유지되는가?
- 옵션별 분기, 특히 fallback, language policy, context, no-speech gate가 검증되는가?
- LM fusion on/off 경계와 metadata mismatch가 테스트되는가?
- 문서만 바꾼 경우에는 실행 테스트 생략 사유를 작업 설명에 남기는가?

## 7. 우선순위

리뷰 순서는 다음이 좋다.

1. 프로젝트 목적과 문서 표현이 맞는지 확인한다.
2. public API와 옵션 계약이 실제 코드와 일치하는지 확인한다.
3. LM fusion이 optional boundary를 지키는지 확인한다.
4. pipeline 옵션 분기와 테스트 커버리지를 확인한다.
5. OSS 공개 관점에서 내부 경로, 미완성 기능 광고, 에러 메시지를 확인한다.

## 8. 리뷰 보고서 관리

개발 중 리뷰는 여러 번 반복되므로, 매 회차 결과를 다음 규칙으로 남긴다.

- **위치**: `docs/dev/reviews/`
- **네이밍**: `YYYY-MM-DD-<주제>.md` (예: `2026-06-11-control-surface.md`). 한 회차 = 한 파일.
- **양식**: `docs/dev/reviews/_TEMPLATE.md`를 복사해 사용. 발견 사항은 위 §1~7 섹션 번호로 분류한다.
- **인덱스**: `docs/dev/reviews/README.md`에 한 줄(날짜·대상·결과)씩 추가한다.

원칙:

- 과거 보고서는 **수정하지 않고 추가만** 한다(이력 보존).
- 매 리뷰 시작 시 직전 보고서의 "후속 액션"을 먼저 확인해 회귀를 막는다.
- 보고서는 개발 산출물이므로 git에 커밋하되, 내부 경로/민감 정보는 제외한다.
- 정본 정책은 [`SSOT.md`](../SSOT.md)에 두고, 보고서는 그 계약 대비 점검 결과만 기록한다.
