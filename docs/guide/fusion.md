# LM Fusion — KenLM shallow fusion 운영 가이드

도메인 어휘가 정해져 있을 때, 재학습 없이 KenLM BPE 언어모델을 **shallow fusion**으로 끼워
인식 정확도를 올린다. 요청 단위로 on/off 가능하며, 끄면 baseline STT와 **완전히 동일**하다.

> 정본 계약은 [`../SSOT.md`](../SSOT.md) §4. 이 문서는 운영 절차만 다룬다.

---

## 1. 동작 조건 (언제 fusion이 실제로 켜지나)

세 조건이 **모두** 충족돼야 fusion이 적용된다. 하나라도 빠지면 자동으로 plain decode.

```
lm_path 존재 (load 시)   AND   lm_enabled=True   AND   alpha > 0
```

내부적으로:
- `lm_enabled=False` 또는 `lm_path=None` → `FusionOptions(enabled=False)`
- `alpha <= 0` → `to_generate_kwargs()`가 `{}` 반환 → 백엔드에 `lm_fusion_*` 미전달

즉 **alpha=0 또는 미설정은 baseline과 비트 단위로 동일**하다.

전달되는 generate kwargs (4개뿐, `lm_fusion_beta`는 절대 전달 안 함):
`lm_fusion_model_path`, `lm_fusion_alpha`, `lm_fusion_asr_topk`, `lm_fusion_debug`.

---

## 2. 사전 준비 — 패치된 CTranslate2 빌드

fusion은 **stock PyPI ctranslate2로는 동작하지 않는다.** KenLM BPE 패치가 들어간 fork를
`WITH_KENLM=ON`으로 소스 빌드해야 한다.

- fork: [YuBeomGon/CTranslate2 @ `feature/kenlm-bpe-fusion`](https://github.com/YuBeomGon/CTranslate2/tree/feature/kenlm-bpe-fusion)
- 단일 `WITH_KENLM=ON` 빌드가 baseline + fusion 둘 다 지원(파라미터 안 주면 baseline).
- ⚠️ stock ctranslate2와 패치 lib은 **ABI 비호환 — 둘 다 설치 금지**. fusion 쓸 땐 `[ct2]` extra를
  설치하지 말고 fork만 사용. 이미 빌드된 체크아웃은 `scripts/ct2_env.sh`로 연결.

설치/빌드 상세는 [`../../README.md`](../../README.md) "KenLM fusion" 섹션.

---

## 3. KenLM `.binary` + 메타데이터

wrapper는 KenLM을 **빌드하지 않는다.** 이미 만들어진 `.binary`를 받아 경로만 백엔드로 넘긴다
(코퍼스/빌드는 호출자 파이프라인 몫).

메타데이터 게이트(조용한 품질 저하 방지): `.binary` 옆에 `<lm>.binary.meta.json`을 두면
`load(verify_lm_metadata=True)`가 `tokenizer_hash` / `ct2_model_hash`를 대조하고, 불일치 시
`MetadataMismatchError`로 **거부**한다. KenLM이 다른 tokenizer로 빌드됐는데 조용히 token-id가
어긋나는 사고를 막는다.

```json
{ "tokenizer_hash": "...", "ct2_model_hash": "...", "kenlm_order": 5, "asr_topk": 50 }
```

검증을 끄려면 `verify_lm_metadata=False`(권장하지 않음).

---

## 4. 사용 예

```python
import whisper_lm_fusion, soundfile as sf

engine = whisper_lm_fusion.load(
    "path/to/ct2-model",
    lm_path="path/to/domain.binary",   # 메타데이터 사이드카가 옆에 있어야 함
    alpha_default=0.2,                 # load-time 기본 alpha
    tokenizer_hash="<hash>",           # 메타데이터 대조
)

audio, sr = sf.read("call.wav")

# fusion on
text = engine.transcribe(audio, sr, lm_enabled=True, alpha=0.2).text

# fusion off — baseline과 동일
base = engine.transcribe(audio, sr, lm_enabled=False).text
```

---

## 5. 튜닝 포인트

| 노브 | 의미 | 가이드 |
|---|---|---|
| `alpha` | KenLM 점수 가중 | 0.1~0.3에서 시작. 너무 크면 LM-bias로 과교정 |
| `topk`(`lm_fusion_asr_topk`) | KenLM 재점수 대상 ASR top-k | 기본 50. k 밖 토큰은 rescue 안 함 |
| `fusion_debug` | 디버그 로그 | 점수 확인용, 운영에선 off |

alpha/topk를 데이터로 최적화하는 것은 **외부 sweep runner** 몫이다(라이브러리는 메커니즘만 제공).

---

## 6. 자주 겪는 문제

- `ModuleNotFoundError: ctranslate2` → fork 미빌드 또는 env 미연결. `source scripts/ct2_env.sh`.
- fusion을 켰는데 baseline과 결과가 같다 → `alpha<=0`이거나 `lm_path`/`lm_enabled` 누락 확인(§1).
- `MetadataMismatchError` → KenLM이 다른 tokenizer/model로 빌드됨. 올바른 사이드카 또는 해시 확인.
- runtime error on fusion request → `WITH_KENLM=OFF`로 빌드된 CT2. ON으로 재빌드.
