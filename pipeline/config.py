"""환경 변수 기반 설정. `.env`(또는 셸 환경)에서 API 키/엔드포인트를 읽는다.

.env.example 를 복사해 .env 로 채운 뒤 사용하세요. 키는 절대 커밋하지 마세요.
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv 미설치 시 셸 환경만 사용
    pass


@dataclass
class Config:
    # --- Digi-Key (OAuth2 client credentials) ---
    digikey_client_id: str = os.getenv("DIGIKEY_CLIENT_ID", "")
    digikey_client_secret: str = os.getenv("DIGIKEY_CLIENT_SECRET", "")
    # 운영 전환 전 샌드박스로 테스트하려면 base를 sandbox-api.digikey.com 으로
    digikey_base: str = os.getenv("DIGIKEY_BASE", "https://api.digikey.com")

    # --- Mouser (API key) ---
    mouser_api_key: str = os.getenv("MOUSER_API_KEY", "")
    mouser_base: str = os.getenv("MOUSER_BASE", "https://api.mouser.com")

    # --- 사양 추출용 LLM (OpenAI 호환 엔드포인트) ---
    # Colab에서 데이터 수집 시: 노트북에서 llama.cpp 의 llama-server 를 OpenAI 호환
    #   엔드포인트로 띄워 사용(아래 기본값 그대로). response_format=json_object 가
    #   grammar 로 JSON 을 강제하므로 Qwen3 thinking 토큰이 섞여도 파싱이 깨지지 않는다.
    # 다른 옵션:
    #   Ollama:  http://localhost:11434/v1  (LLM_API_KEY=ollama, LLM_MODEL=qwen3:14b)
    #   vLLM:    http://localhost:8000/v1   (LLM_MODEL=Qwen/Qwen3-14B)
    #   DashScope 클라우드: https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (LLM_MODEL=qwen-plus)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3-14b")

    # --- 경로 ---
    pdf_dir: str = os.getenv("PDF_DIR", "data/datasheets")
    catalog_path: str = os.getenv("CATALOG_PATH", "data/parts_catalog.json")


cfg = Config()
