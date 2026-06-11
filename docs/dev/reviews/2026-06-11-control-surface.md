# Review 2026-06-11 — control-surface

- **대상**: `1f1c8b1` (`1f1c8b1 Fix install docs: source install + WITH_KENLM=ON build, ABI caveat`)
- **리뷰 기준**: [`../review_guidelines.md`](../review_guidelines.md) §1~7 · 정본 [`../../SSOT.md`](../../SSOT.md) · [`../../research/pipeline_control_surface.md`](../../research/pipeline_control_surface.md)
- **결과 요약**: ✅ 통과 관찰 7건 / ⚠️ 개선 6건 / ❌ 차단 0건
- **테스트**: `PYTHONPATH=src python -m pytest -q` → `28 passed in 0.07s`

핵심: **계약 위반(❌)은 없다.** LM fusion optional boundary, load/request 기본값 분리, metadata gate,
정책 enum 분기는 모두 SSOT/control-surface 계약대로 동작한다. 발견 사항은 전부 문서·코드 표현
불일치(§5)와 테스트 커버리지 공백(§6) 중심의 개선 항목이다. 가장 중요한 것은 README가
(a) 이미 구현된 temperature fallback ladder를 "미구현"으로 잘못 표기하고, (b) 이 프로젝트의 간판
기능인 정책 enum 4종을 Parameters 표에서 누락한 점이다.

## 발견 사항

| # | 섹션(§1~7) | 심각도 | 내용 | 조치/제안 |
|---|---|---|---|---|
| 1 | §1 | ✅ | `load()`/`transcribe()` 중심 API, 내부는 faster-whisper 복제가 아닌 자체 window-loop+선택기(`engine.py:288`, `_decode.py`). backend 추상화로 정책이 backend-agnostic(`backends/base.py:1`). | 통과 |
| 2 | §2 | ✅ | `load()` 시그니처(`engine.py:397`)가 SSOT §3 / README 예시와 일치. `return_scores`/`return_nbest`/`fusion_mode`는 config에만 존재하고 README가 "아직 공개 동작 아님"으로 명시(`README.md:140`). | 통과 |
| 3 | §2 | ⚠️ | README Parameters 표(`README.md:124~132`)에 간판 기능인 정책 enum 4종(`selection_policy`/`fallback_policy`/`language_policy`/`context_policy`)이 빠져 있음. config에는 존재(`config.py:104,116,121,140`)하고 control-surface 문서가 다루지만, README만 보면 핵심 control surface가 안 보임. | README 표에 정책 enum 행 추가하거나, control-surface 문서로의 링크를 Parameters 절에 명시. |
| 4 | §3 | ✅ | windowing→silence cut→timestamp seek 순서가 의도대로 연결(`engine.py:315~352`, `_decode.decide_cut`/`advance_seek`). 임계값·정책은 전부 `DecodeOptions`로 표면화, 엔진 내 하드코딩 없음. silence cut일 때 `use_timestamps=False`로 prompt 분기(`engine.py:320`). | 통과 |
| 5 | §3 | ✅ | 정책 enum이 실제로 내부 분기를 만듦: `selection_policy`→`select_candidate`(`_decode.py:137`), `fallback_policy`→`should_fallback`(`_decode.py:214`), `language_policy`→`_language_decision`(`engine.py:111`), `context_policy`→`_update_context`(`engine.py:267`). dual_band은 2차 언어 분기까지 구현(`engine.py:184`). | 통과 |
| 6 | §3 | ⚠️ | `context_policy="always"`라도 직전 window가 no-speech로 drop되면 `_update_context`가 `always` 분기(`engine.py:278`) 도달 전에 조기 반환(`engine.py:275~276`). 의도된 가드일 수 있으나 "always" 라벨과 미세하게 다름. | 동작은 합리적. 주석/문서에 "always는 emit된 텍스트 기준"임을 한 줄 명기 권장. |
| 7 | §3 | ⚠️ | `num_hypotheses`가 backend에서 `beam_size`로 클램프됨(`ct2.py:54`). 기본값(둘 다 5)에선 무해하나, 사용자가 `num_hypotheses>beam_size`로 sweep하면 조용히 축소됨. | README/주석에 "num_hypotheses ≤ beam_size로 클램프" 명기. |
| 8 | §4 | ✅ | fusion optional boundary 정확: `lm_enabled=False` 또는 `lm_path=None`이면 `FusionOptions(enabled=False)`(`engine.py:43`). `enabled=True`라도 `alpha<=0`이면 `to_generate_kwargs()`가 `{}` 반환(`config.py:69`). 즉 alpha<=0 또는 no lm_path는 baseline과 동일 — SSOT §4 계약 충족. pipeline-wide fork 없이 generate kwargs로만 주입. | 통과 |
| 9 | §4 | ✅ | load-time 기본값(`alpha_default`/`topk_default`, `config.py:46`)과 request-time override(`DecodeOptions.alpha/topk=None`, `config.py:151`)가 `_resolve_fusion`(`engine.py:49~50`)에서 명확히 분리. `lm_fusion_beta`는 config·kwargs 어디에도 없음(SSOT §4 준수). | 통과 |
| 10 | §4 | ✅ | metadata mismatch가 조용한 저하가 아니라 `MetadataMismatchError`로 표면화(`metadata.py:60~69`), `load()`에서 strict 검증 호출(`engine.py:426~433`). | 통과 |
| 11 | §5 | ⚠️ | **문서/코드 불일치**: README가 "a faster-whisper-style temperature fallback ladder is **not yet implemented**"(`README.md:138`)라고 했으나, ladder는 실제 구현·통과 테스트 존재(`engine.py:200~231`, `should_fallback` `_decode.py:214`, test `test_temperature_fallback_adds_candidates_and_can_win`). 구현된 기능을 미구현으로 과소 표기. | README 문장 수정: temperature fallback은 `fallback_policy`+`temperature_fallback`로 구현됨을 반영. |
| 12 | §5 | ⚠️ | README config 주석(`config.py:3`)이 `docs/implementation_plan.md`를, design 주석들이 `design.md §5.1`/`§7`를 참조하나 실제 경로는 `docs/dev/implementation_plan.md`이고 design.md에 §5.1/§7 번호 매김이 없음(grep 미검출). 경로/섹션 참조가 어긋남. | 주석의 상대경로·섹션 번호를 실제 문서 구조에 맞게 정정. |
| 13 | §6 | ⚠️ | **fusion ON 양성 테스트 부재**: `lm_enabled=True`+유효 `lm_path`+`alpha>0`일 때 `lm_fusion_*` kwargs가 실제로 backend에 전달되는지 검증하는 테스트가 없음. 현재는 OFF 경계(`to_generate_kwargs()=={}`, test_engine.py:89)와 미지원 backend 에러만 커버. | fake backend로 fusion ON 시 `fusion.to_generate_kwargs()`에 4개 키가 들어가는지 assert하는 테스트 추가. |
| 14 | §6 | ⚠️ | 정책 분기 테스트 공백: `dual_band`(2차 언어 decode), `context_policy`(carry/no-carry), `align_tail_trim` hook 경로에 대한 fake-backend 테스트 없음. fallback/per_window_language/no-speech gate/cjk suppress는 커버됨. `load()` 레벨 metadata mismatch 통합 테스트도 없음(`verify_metadata` 단위 테스트만 존재). | dual_band·context carry·load() metadata-mismatch 테스트 추가. |

## 후속 액션 (다음 회차 확인 대상)

- [ ] (§5) README `README.md:138` temperature fallback "미구현" 표기 수정 → 구현됨 반영. (#11)
- [ ] (§2) README Parameters 표에 정책 enum 4종 추가/링크. (#3)
- [ ] (§6) fusion ON 양성 테스트 추가(`lm_fusion_*` kwargs 전달 검증). (#13)
- [ ] (§6) dual_band / context_policy / load()-metadata-mismatch 테스트 추가. (#14)
- [ ] (§5) 코드 주석의 문서 경로·섹션 참조 정정(implementation_plan, design §5.1/§7). (#12)
- [ ] (§3) num_hypotheses 클램프(#7)와 context_policy="always" no-speech 가드(#6)를 문서에 한 줄씩 명기.

## 직전 회차 액션 처리

- 본 보고서가 **최초 회차**이므로 직전 회차 액션 없음(이력 시작점).
