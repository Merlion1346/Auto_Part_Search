"""유통사 API에서 부품 사양과 데이터시트 URL을 수집한다.

각 수집기는 공통 dict 스키마를 반환한다:
    {
        "part_name": str,        # 제조사 부품 번호
        "manufacturer": str,
        "category": str,
        "datasheet_url": str | None,
        "api_specs": dict,       # 유통사가 제공한 정형 파라미터 (이름 -> 값)
        "source": "digikey" | "mouser",
    }
"""

from .digikey import DigiKeyClient
from .mouser import MouserClient

__all__ = ["DigiKeyClient", "MouserClient", "get_client"]


def get_client(source: str):
    """source 이름으로 수집기 인스턴스를 반환한다."""
    source = source.lower()
    if source == "digikey":
        return DigiKeyClient()
    if source == "mouser":
        return MouserClient()
    raise ValueError(f"지원하지 않는 수집 소스: {source} (digikey | mouser)")
