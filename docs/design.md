# whisper-lm-fusion Design

**Date**: 2026-06-09

This document describes the design of `whisper-lm-fusion`. The canonical project
contract lives in [`docs/SSOT.md`](SSOT.md); this file expands the wrapper
responsibility boundary and backend/fusion rules.

## 1. whisper-lm-fusion란

faster-whisper 스타일의 **generic STT wrapper**다. CTranslate2 Whisper 백엔드를
감싸고, 선택적으로 patched CTranslate2의 KenLM BPE fusion을 사용한다. long-form
디코딩 파이프라인은 내부에 두고 파라미터만 밖으로 노출한다.

```
caller / pipeline  ──call──▶  whisper-lm-fusion  ──▶  backend
                               audio + params -> text
```

의존 방향은 **pipeline → wrapper → backend** 단방향이다.

## 2. 핵심 원칙: wrapper는 generic, 도메인을 모른다

wrapper가 아는 것 (이게 전부):

- audio 입력
- model path
- lm binary path (이미 만들어진 `.binary`)
- alpha / topk / fusion mode
- decoding options (beam_size, num_hypotheses, patience, temperature 등)
- Whisper tokenizer (token-id ↔ text, **decode 용도**)

wrapper에 **넣지 않는 것** (전부 caller/pipeline 소유):

- domain glossary / blocklist / replacement rules
- domain corpus 및 corpus 생성 규칙
- KenLM 빌드(토큰화 + lmplz + build_binary)
- 후처리 / 평가셋 / 리포트 포맷
- alpha/topk 의 "좋은 값" (이건 artifact metadata = pipeline 소유)

> 도메인 로직이 wrapper로 새면 공유 라이브러리가 무너진다. wrapper = "audio + params ->
> text"만.

## 3. KenLM artifact는 받기만 한다 (빌드 안 함)

- KenLM `.binary` 빌드는 **전부 pipeline에서** 오프라인 1회 수행한다. wrapper는 빌드하지
  않는다.
- wrapper는 `load()` 시 완성된 `.binary` path를 받아 CT2에 전달할 뿐이다.
- **매 init·매 요청마다 LM을 다시 굽지 않는다.**

### tokenizer 일치 보증 (유일한 구조적 리스크)

KenLM corpus는 Whisper BPE token-id 기반이라, corpus를 토큰화한 tokenizer(pipeline,
build-time)와 wrapper가 decode에 쓰는 tokenizer(serving-time)가 **정확히 같아야** 한다.

- wrapper는 domain text·corpus를 **모른다.** pipeline이 wrapper와 **동일 모델명**으로
  tokenizer를 따로 로드해 빌드까지 끝낸다.
- 두 tokenizer 일치는 **artifact metadata의 `tokenizer_hash`** 로만 보증한다.
- wrapper는 `load()` 시 metadata의 `tokenizer_hash` / `ct2_model_hash`가 자신의 현재
  모델과 다르면 **LM 로드를 거부**한다. (token-id mismatch는 조용히 품질만 깎으므로 게이트)

### artifact metadata 스키마 (pipeline이 생성, wrapper가 검증)
```json
{
  "model_name": "large-v3-turbo",
  "tokenizer_hash": "...",
  "ct2_model_hash": "...",
  "kenlm_order": 5,
  "asr_topk": 50,
  "fusion_mode": "topk",
  "corpus_version": "domain_20260607",
  "alpha_default": 0.20,
  "topk_default": 50
}
```

## 4. 3단계 실행 시점

| 단계 | 주체 | 내용 |
|---|---|---|
| (A) build-time | **pipeline** | domain text → 토큰화 → lmplz/build_binary → `.binary` + metadata. 오프라인 1회. |
| (B) serving init | **wrapper** | `load(model, lm_path)`: bin을 CT2에 전달 + tokenizer_hash 검증. |
| (C) per request | **wrapper** | `transcribe(audio, alpha, lm_enabled, ...)`: 매 요청 fusion on/off·세기 토글. |

(C)의 fusion 토글은 generate kwargs라 비용이 없다. `alpha=0` 또는 `lm_enabled=False`면
fusion off.

## 5. API

```python
# (B) serving init — 부팅 1회
engine = wrapper.load(
    model_path,                 # CT2 변환된 Whisper 모델 경로
    lm_path=None,               # pipeline이 만든 .binary (없으면 fusion 불가)
    alpha_default=0.0,
    topk_default=50,
    fusion_mode="topk",
    device="cuda",
    compute_type="float16",
)

# (C) per request — fusion on/off는 인자로
result = engine.transcribe(
    audio, sr, *,
    alpha=0.0,                  # 0 이면 off
    topk=50,
    lm_enabled=False,
    beam_size=5,
    num_hypotheses=5,
    patience=2.0,
    sampling_temperature=0.0,
)                               # -> TranscriptionResult
text = result.text
```

- corpus·tokenize·빌드 API는 **노출하지 않는다.**
- domain customization = caller/pipeline이 다른 `.binary`/파라미터를 넘기는 것뿐.

### 5.1 입력 계약 (정본)

wrapper의 입력은 **이미 디코딩된 파형**이다. 파일 디코딩·리샘플·채널 분리는 wrapper가
하지 않는다.

- **정본 입력**: `audio: np.ndarray (float32, mono)` + `sr: int`. 권장은 **16kHz mono**.
- **호출자(pipeline) 책임**: 파일 I/O, 코덱 디코딩(wav/mp3/...), **리샘플(→16k)**,
  **l/r 채널 분리·선택**, VAD 등 도메인 전처리. 이건 운영·도메인 결정이라 generic wrapper가
  알 필요 없고, 무거운 디코딩 의존성을 라이브러리에 끌어들이지 않기 위함이다.
- **output**: 기본 `text`. 현재 opt-in 구현은 `segments`(단위: window/segment, 각
  `text/start/end/logprob/no_speech_prob`)다. `scores` / `nbest`는 config 표면에
  남아 있으나 공개 동작으로 문서화하려면 구현을 먼저 채워야 한다.
- **비범위(현재)**: word-level timestamp(=align 강제 → 로직 결합), 파일 path 입력 편의,
  내부 채널 분리. path 입력은 후속에 optional extra로만 고려한다.

> 즉 경계는 `np.ndarray(16k mono float32)`다. 스모크에서 한 리샘플/mono 변환도 원래
> 호출자(pipeline) 몫이다.

## 6. 내부 디코딩 구성

현재 wrapper 코어는 faster-whisper 기본값 복제가 아니라 phase3 self-evolve에서 발견된 전략을 generic control surface로 압축한다. 주요 노브는 `beam_size`, `num_hypotheses`, `selection_policy`, `fallback_policy`, `temperature_fallback`, `language_policy`, `suppress_cjk_kana`, `align_tail_trim`, `context_policy`다. 자세한 sweep 표면은 [`docs/research/pipeline_control_surface.md`](research/pipeline_control_surface.md)를 따른다.

현재 wrapper 코어는 아래 단위로 분리되어 있다.

| 구성 | 역할 |
|---|---|
| `engine.py` | long-form window loop, prompt build, backend 호출, 결과 조립 |
| `_decode.py` | RMS silence cut, timestamp-aware seek, N-best selection |
| `backends/` | CT2 Whisper load, feature extraction, generate pass-through |
| `config.py` | load-time / request-time 파라미터 스키마 |

포함하지 않는 것: evaluation CLI, metrics, domain terms, dataset normalization, KenLM
build, domain reports.

## 7. backend(CT2 fork) 사용 규칙

- patched CTranslate2 Python binding은 `lm_fusion_model_path`, `lm_fusion_alpha`,
  `lm_fusion_asr_topk`, `lm_fusion_debug`만 노출한다. **`lm_fusion_beta`는 없음** — 전달
  금지.
- `lm_fusion` off일 때 baseline CT2와 **디코드 출력이 동일**해야 한다. (bit-level logit이
  아니라 출력 텍스트 일치 + score 허용오차 기준)
- KenLM 없는 빌드도 가능하도록 optional dependency로 다룬다.

patched CTranslate2는 fork에 있다: [YuBeomGon/CTranslate2 @ `feature/kenlm-bpe-fusion`](https://github.com/YuBeomGon/CTranslate2/tree/feature/kenlm-bpe-fusion).
정본 설치는 이 fork를 `WITH_KENLM=ON`으로 소스
빌드 후 Python binding 설치(`pip install ./python`)다. `scripts/ct2_env.sh`는 이미
빌드된 로컬 checkout을 가리키는 편의 스크립트일 뿐, 설치 경로가 아니다.

## 8. 속도 메모 (참고)

- fusion on은 off 대비 **고정 +26% latency** (RTF 0.0154→0.0194). 원인은 per-beam-step
  GPU→CPU 동기화 + KenLM scoring, `beam_size × topk × step 수`에 비례.
- 단건 latency 영향은 작지만(RTF≈0.02) GPU 점유 +26%는 배치 throughput 비용.
- 근거 문서는 공개 전 sanitize가 필요하다. 공개 README에는 환경/데이터셋이 검증된 수치만
  싣는다.

## 9. 1차 범위 / 비범위

**1차 (thin)**: `load()` + `transcribe()` 최소 API, long-form 내재화, tokenizer_hash 검증.

**비범위(후속)**: batching, VAD, segment merge, word timestamp, diarization. 처음부터 크게
만들지 않는다.
