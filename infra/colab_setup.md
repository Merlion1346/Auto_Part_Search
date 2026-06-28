# Google Colab에서 데이터 수집 + Qwen3-14B QLoRA 학습하기

데이터 수집과 LoRA 튜닝을 **모두 Google Colab Pro**에서 실행합니다.
바로 실행 가능한 노트북: [`notebooks/auto_search_colab.ipynb`](../notebooks/auto_search_colab.ipynb)

## 권장 런타임

Qwen3-14B QLoRA(4-bit) 기준:

| GPU | VRAM | 적합도 | 비고 |
|-----|------|--------|------|
| **A100 40GB** (Colab Pro) | 40GB | ✅ 권장 | 기본값(batch 2 / seq 2048) 그대로 |
| L4 24GB (Colab Pro) | 24GB | ⚠️ 가능, `--batch-size 1` 권장 | bf16 사용 |
| T4 16GB (무료) | 16GB | ⚠️ 빠듯, `--batch-size 1 --max-seq-len 1024` 필요 | bf16 미지원 → fp16 |

`런타임 → 런타임 유형 변경`에서 GPU(권장 A100)를 선택하세요.

## 핵심 개념: Colab은 휘발성 → Drive에 저장

Colab 런타임은 일정 시간 후 초기화됩니다. 그래서 노트북은:

- 데이터시트 PDF / 카탈로그 / `train.jsonl` / 학습된 어댑터를 **Google Drive**(`/content/drive/MyDrive/auto_search`)에 저장합니다.
- 런타임이 끊겨도 Drive의 산출물로 이어서 작업할 수 있습니다.

## 노트북 단계 요약

1. **GPU 확인** — `nvidia-smi`
2. **Drive 마운트** — 저장 경로 `MyDrive/auto_search` 생성
3. **코드/의존성** — 저장소 clone + `default-jre`(PDF 파싱) + `pip install -r requirements.txt`
4. **llama.cpp 서버 기동** — llama.cpp를 CUDA로 빌드 후 `llama-server` 로 Qwen3-14B(GGUF, `-hf` 자동 다운로드)를 OpenAI 호환(`:8000/v1`)으로 서빙. reasoning(thinking)은 `--reasoning-budget 0` 으로 끔
   - Ollama/vLLM/외부 API(DashScope 등)를 쓰면 이 단계 대신 `.env`의 `LLM_*`만 교체
5. **시크릿** — Colab 보안 비밀(🔑)에 `MOUSER_API_KEY` 등록 → `.env` 자동 작성
6. **데이터 수집** — `pipeline.build_catalog` → `parts_catalog.json`(Drive)
7. **전처리** — `train/preprocess.py --augment` → `train.jsonl`(Drive)
8. **학습** — `train/train_qlora.py` → 어댑터(Drive)
9. **추론 테스트** — `src/recommend.py`

## 키 발급

- **Mouser**: <https://www.mouser.com/api-hub> (Search API v1)
- **Digi-Key**: <https://developer.digikey.com> (OAuth2 client credentials)
- 둘 중 하나만 있어도 수집 가능 (`--source mouser` / `--source digikey`)

## 자주 겪는 문제

- **버전 충돌**: `requirements.txt` 는 torch를 핀하지 않고 최소 버전만 지정하므로
  Colab 기본 torch/transformers를 그대로 사용합니다. 핵심은 `bitsandbytes>=0.46.1`
  (transformers 5.x의 4-bit 양자화 요구) — 옛 버전(0.45.x)이 남아 있으면
  `!pip install -U bitsandbytes` 로 갱신 후 학습 셀을 다시 실행하세요.
- **llama-server `-hf` 다운로드 실패**: llama.cpp가 libcurl 없이 빌드되면 `-hf`가 동작하지
  않습니다. `libcurl4-openssl-dev` 설치 후 `-DLLAMA_CURL=ON` 으로 재빌드하세요(노트북 4번 셀에 포함됨).
  GGUF(~9GB) 첫 다운로드가 오래 걸릴 수 있으니, 서버가 응답할 때까지 기다린 뒤 다음 셀을 실행하세요.
- **(대안) Ollama 사용 시 `requires zstd`**: `!apt-get -qq install -y zstd` 후
  `ollama serve` + `ollama pull qwen3:14b` 로 띄우고 `.env`의 `LLM_*`를 Ollama 값으로 바꾸세요.
- **T4/L4에서 OOM**: `--batch-size 1`, (그래도 부족하면) `--max-seq-len 1024` 로 낮춥니다.
- **세션 종료로 학습 중단**: 산출물이 Drive에 있으므로 마지막 체크포인트에서 재개하거나
  다시 8단계만 실행하면 됩니다. 장시간 학습은 Pro의 백그라운드 실행 옵션을 권장합니다.

## (선택) 배포용 병합 체크포인트

```python
!python train/train_qlora.py --merge \
    --output-dir "$DRIVE_ROOT/models/qwen3-14b-parts-lora"
# → ...-merged (fp16/bf16 단일 모델, 약 28GB) 생성
```
