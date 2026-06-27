"""Digi-Key Product Information V4 API 수집기.

인증: OAuth2 client credentials (2-legged). 토큰은 약 30분 유효 → 만료 시 재발급.
검색: POST /products/v4/search/keyword

필요 자격증명: DIGIKEY_CLIENT_ID, DIGIKEY_CLIENT_SECRET
(https://developer.digikey.com 에서 앱 생성 후 발급)
"""

import time

import requests

from ..config import cfg


class DigiKeyClient:
    def __init__(self):
        if not (cfg.digikey_client_id and cfg.digikey_client_secret):
            raise RuntimeError("DIGIKEY_CLIENT_ID / DIGIKEY_CLIENT_SECRET 가 필요합니다.")
        self.base = cfg.digikey_base.rstrip("/")
        self._token = None
        self._token_exp = 0.0

    # --- 인증 ---
    def _access_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        resp = requests.post(
            f"{self.base}/v1/oauth2/token",
            data={
                "client_id": cfg.digikey_client_id,
                "client_secret": cfg.digikey_client_secret,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_exp = time.time() + int(data.get("expires_in", 1800))
        return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "X-DIGIKEY-Client-Id": cfg.digikey_client_id,
            "Content-Type": "application/json",
            "X-DIGIKEY-Locale-Language": "en",
            "X-DIGIKEY-Locale-Currency": "USD",
        }

    # --- 검색 ---
    def search(self, keyword: str, limit: int = 25) -> list[dict]:
        resp = requests.post(
            f"{self.base}/products/v4/search/keyword",
            headers=self._headers(),
            json={"Keywords": keyword, "Limit": limit, "Offset": 0},
            timeout=60,
        )
        resp.raise_for_status()
        products = resp.json().get("Products", [])
        return [self._normalize(p) for p in products]

    @staticmethod
    def _normalize(p: dict) -> dict:
        specs = {}
        for param in p.get("Parameters", []):
            name = param.get("ParameterText") or param.get("Parameter")
            value = param.get("ValueText") or param.get("Value")
            if name and value:
                specs[name] = value
        category = p.get("Category", {})
        return {
            "part_name": p.get("ManufacturerProductNumber", ""),
            "manufacturer": (p.get("Manufacturer") or {}).get("Name", ""),
            "category": category.get("Name", "") if isinstance(category, dict) else "",
            "datasheet_url": p.get("DatasheetUrl") or None,
            "api_specs": specs,
            "source": "digikey",
        }
