# 부품 자동추천 AI 시스템

회로도 설계 시, **자연어로 입력한 요구사항에 부합하는 부품을 추천**하는 AI 시스템입니다.

사용자가 "5V 입력에 3.3V 출력, 1A 이상 LDO 레귤레이터" 와 같이 자연어로 요구사항을 입력하면,
시스템이 요구사항을 해석하여 가장 적합한 부품 이름을 추천합니다.

---

## ✨ 주요 기능

- **자연어 요구사항 입력**: 전기적 사양을 자연어 문장으로 자유롭게 입력
- **요구사항 해석**: 입력 문장에서 부품 종류, 전기적 파라미터(전압·전류·패키지 등), 제약조건 추출
- **부품 추천**: 요구사항과 부합하는 부품 이름을 우선순위와 함께 추천
- **근거 제시**: 각 부품이 추천된 이유(매칭된 사양)를 함께 제공
- **회로 BOM 추천**: 회로 용도를 입력하면 필요한 부품 목록(BOM) 전체를 추천 (데이터시트 대표 응용 회로 학습)
- **데이터시트 기반 학습**: 부품 데이터시트 데이터셋으로 학습한 LLM이 부품 사양을 이해하고 추천

---

## 🧭 동작 흐름

```
[부품 데이터시트 데이터셋]
        │  (학습)
        ▼
 ┌──────────────┐
 │   학습된 LLM  │   ← 데이터시트로 부품 사양·특성을 학습
 └──────────────┘
        ▲
        │
[자연어 요구사항]
        │
        ▼
 ┌──────────────┐
 │ 요구사항 해석 │   ← 부품 종류 / 전기적 사양 / 제약조건 추출
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │  부품 매칭·   │   ← 학습된 지식으로 적합 부품 추론
 │   추천 추론   │
 └──────────────┘
        │
        ▼
 [추천 부품 목록]   ← 부품 이름 + 추천 근거
```

---

## 🚀 시작하기

### 실행 환경

데이터 수집과 QLoRA 학습을 **모두 Google Colab Pro**에서 실행합니다.
바로 실행 가능한 노트북: **[`notebooks/auto_search_colab.ipynb`](notebooks/auto_search_colab.ipynb)**
(전체 절차는 [infra/colab_setup.md](infra/colab_setup.md) 참고)

- 권장 런타임: **Colab Pro — NVIDIA A100 40GB** (L4 24GB / T4 16GB도 가능하나 배치/시퀀스 축소 필요)
- 산출물은 휘발성 런타임 대신 **Google Drive**(`MyDrive/auto_search`)에 저장
- 의존성: 노트북이 `default-jre`(PDF 파싱) + `requirements.txt` 를 자동 설치
- 수집용 LLM: 노트북에서 [Ollama](https://ollama.com) `qwen3:14b` 를 띄워 사용 (외부 API로 교체 가능)

> 로컬에서 개별 스크립트를 돌려보려면 Python 3.10+, Java 11+, CUDA 12.x GPU와
> `pip install -r requirements.txt` 가 필요합니다. 아래 명령들은 노트북 셀에서도 동일하게 동작합니다.

### 1) 데이터 수집·파싱 (유통사 API + LLM)

유통사 API(Digi-Key/Mouser)로 부품을 검색해 데이터시트를 내려받고, 텍스트를 추출한 뒤
LLM으로 사양을 정형화하여 부품 카탈로그(JSON)를 만듭니다.

```bash
cp .env.example .env                  # MOUSER_API_KEY / LLM 엔드포인트 입력
python -m pipeline.build_catalog --source mouser \
    --keywords-file pipeline/keywords.example.txt --limit 15
# → data/parts_catalog.json 생성
```

### 2) 학습 데이터 전처리

부품 카탈로그(JSON)를 학습용 대화 데이터셋(JSONL)으로 변환합니다.

```bash
# 규칙 기반: 부품당 요구사항 1건
python train/preprocess.py --input-glob "data/*catalog*.json"

# LLM 증강: 부품당 다양한 표현의 요구사항을 생성해 샘플 수를 늘림 (.env 의 LLM_* 사용)
python train/preprocess.py --augment --num-variations 5
```

> **요구사항 다양화(paraphrase augmentation)**: `--augment` 를 켜면 같은 부품에 대해
> LLM이 구어체/격식체/영어 혼용/약어 등 다양한 표현의 요구사항을 N개 생성합니다.
> 같은 부품을 여러 어투로 학습시켜 추천 모델의 일반화 성능을 높입니다.
> (LLM 호출 실패 시 규칙 기반 샘플 1건으로 자동 폴백)

### 3) 파인튜닝 (Qwen3-14B + QLoRA)

```bash
python train/train_qlora.py \
    --train-file data/train.jsonl \
    --output-dir models/qwen3-14b-parts-lora
```

> 기본값(14B · batch 2 · grad_accum 8 · seq_len 2048 · gradient checkpointing)은
> Colab Pro A100 40GB 기준입니다. VRAM이 작은 GPU(L4/T4)에서는 `--batch-size 1`,
> `--max-seq-len 1024` 로 낮추세요(T4는 bf16 미지원 → fp16 자동). Colab 전체 절차는
> [infra/colab_setup.md](infra/colab_setup.md) 참고.

### 4) 부품 추천 (추론)

```bash
python src/recommend.py --adapter models/qwen3-14b-parts-lora \
    "5V를 3.3V로 변환하고 1A 이상 출력하는 LDO 추천해줘"
```

---

## 💡 사용 예시

**입력**

```
5V를 3.3V로 변환하고 출력 전류 1A 이상 지원하는 LDO 레귤레이터
```

**출력**

```
추천 부품: AMS1117-3.3
근거: 고정 3.3V 출력 LDO로 최대 1A를 지원하며 입력 전압 범위가 5V를 포함합니다
      (드롭아웃 약 1.3V). 5V→3.3V 변환과 1A 이상 조건에 부합. SOT-223 패키지.
```

---

## 📁 프로젝트 구조

```
auto_search/
├── README.md
├── requirements.txt
├── .env.example                # API 키 / LLM 엔드포인트 템플릿
├── pipeline/                   # 데이터 수집·파싱 파이프라인
│   ├── config.py               # 환경 변수 설정
│   ├── collect/                # 유통사 API 수집기
│   │   ├── digikey.py          #   Digi-Key Product Info V4
│   │   ├── mouser.py           #   Mouser Search API v1
│   │   └── download.py         #   데이터시트 PDF 다운로드
│   ├── parse/
│   │   ├── pdf_extract.py      #   OpenDataLoader PDF→Markdown 추출
│   │   └── llm_extract.py      #   LLM 사양·회로 정형화(JSON)
│   ├── build_catalog.py        # 파이프라인 오케스트레이터
│   └── keywords.example.txt    # 검색 키워드 예시
├── data/
│   ├── sample_parts_catalog.json   # 부품 사양 카탈로그(형식 예시)
│   ├── sample_dataset.jsonl        # 학습 대화 데이터(형식 예시)
│   └── datasheets/                 # 내려받은 PDF (생성됨)
├── train/
│   ├── preprocess.py           # 카탈로그 → 학습 데이터셋 변환
│   └── train_qlora.py          # Qwen3-14B QLoRA 파인튜닝
├── src/
│   └── recommend.py            # 학습된 어댑터로 부품 추천(추론)
├── notebooks/
│   └── auto_search_colab.ipynb # Colab 올인원 파이프라인(수집+학습+추론)
├── models/                     # 학습된 LoRA 어댑터/체크포인트 (생성됨)
└── infra/
    └── colab_setup.md          # Google Colab 학습 환경 가이드
```

> 현재 저장소 초기 단계입니다. 구조는 개발 진행에 따라 변경될 수 있습니다.

---

## 🛠 기술 스택

- **언어**: Python 3.10+
- **베이스 모델**: [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- **파인튜닝**: QLoRA (4-bit NF4 양자화 + LoRA 어댑터)
- **프레임워크**: Hugging Face `transformers` · `trl`(SFTTrainer) · `peft` · `bitsandbytes`
- **학습 인프라**: Google Colab Pro (NVIDIA A100 40GB)
- **학습 데이터**: 부품 데이터시트(전기적 사양, 패키지, 동작 조건 등)

## 📚 학습 데이터셋

LLM은 부품 데이터시트를 가공한 데이터셋으로 학습합니다.

- **수집**: 유통사 API(Digi-Key / Mouser)로 부품 검색 → 정형 사양 + 데이터시트 PDF 확보
- **파싱**: `OpenDataLoader`로 데이터시트 PDF → 구조 보존 Markdown 변환 → LLM(OpenAI 호환 엔드포인트)으로 다음을 정형화(JSON)
  - 부품 사양(`specs`) + 요약(`datasheet_summary`)
  - **대표 응용 회로(`reference_circuits`)** — 데이터시트의 Typical Application 섹션에서 회로 BOM 추출
- **카탈로그**: 유통사 정형 사양 + LLM 추출 결과 병합 → `data/parts_catalog.json`
- **전처리**: 카탈로그에서 두 종류의 학습 샘플 생성
  - 단일 부품 추천: `요구사항 → 부품 1개 + 근거`
  - 회로 BOM 추천: `회로 설명 → 전체 부품 목록(BOM)`
- **포맷**: TRL conversational 포맷(JSONL) — `{"messages": [system, user, assistant]}`
- **학습 목표**: 자연어 요구사항 ↔ 부품 사양, 그리고 회로 용도 ↔ 부품 조합(BOM)의 매핑 관계 학습

---

## 📌 로드맵

- [x] QLoRA 학습/추론 파이프라인 스캐폴딩 (Qwen3-14B)
- [x] 데이터 수집·파싱 파이프라인 (유통사 API + LLM 사양 추출)
- [x] 요구사항 다양화 증강 (paraphrase augmentation)
- [x] 레퍼런스 회로 → 전체 BOM 학습 샘플 생성
- [ ] 대규모 부품 카탈로그 구축 (키워드 확장·수집 자동화)
- [ ] LLM 추출 사양 검증·정합성 체크 (단위/범위 밸리데이션)
- [ ] 데이터셋 기반 본 학습(파인튜닝) 및 평가
- [ ] 추천 정확도 평가 지표·벤치마크 구축
- [ ] CLI / 웹 인터페이스 제공
- [ ] 회로도 설계 툴(EDA) 연동

---

## 📄 라이선스

별도 명시 전까지 모든 권리는 작성자에게 있습니다.
