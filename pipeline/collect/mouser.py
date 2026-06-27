"""Mouser Search API v1 수집기.

인증: API 키를 쿼리 파라미터(?apiKey=)로 전달.
검색: POST /api/v1/search/keyword

필요 자격증명: MOUSER_API_KEY
(https://www.mouser.com/api-hub/ 에서 발급)
"""

import requests

from ..config import cfg


class MouserClient:
    def __init__(self):
        if not cfg.mouser_api_key:
            raise RuntimeError("MOUSER_API_KEY 가 필요합니다.")
        self.base = cfg.mouser_base.rstrip("/")

    def search(self, keyword: str, limit: int = 25) -> list[dict]:
        resp = requests.post(
            f"{self.base}/api/v1/search/keyword",
            params={"apiKey": cfg.mouser_api_key},
            json={
                "SearchByKeywordRequest": {
                    "keyword": keyword,
                    "records": limit,
                    "startingRecord": 0,
                }
            },
            timeout=60,
        )
        resp.raise_for_status()
        parts = (resp.json().get("SearchResults") or {}).get("Parts", [])
        return [self._normalize(p) for p in parts]

    @staticmethod
    def _normalize(p: dict) -> dict:
        specs = {}
        for attr in p.get("ProductAttributes", []):
            name = attr.get("AttributeName")
            value = attr.get("AttributeValue")
            if name and value:
                specs[name] = value
        return {
            "part_name": p.get("ManufacturerPartNumber", ""),
            "manufacturer": p.get("Manufacturer", ""),
            "category": p.get("Category", ""),
            "datasheet_url": p.get("DataSheetUrl") or None,
            "api_specs": specs,
            "source": "mouser",
        }
