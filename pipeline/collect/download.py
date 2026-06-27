"""데이터시트 PDF 다운로드. URL → 로컬 파일 경로."""

import hashlib
import os

import requests

from ..config import cfg

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; parts-pipeline/0.1)"}


def download_pdf(url: str, part_name: str, dest_dir: str | None = None) -> str | None:
    """데이터시트 PDF를 내려받아 로컬 경로를 반환. 실패 시 None."""
    if not url:
        return None
    dest_dir = dest_dir or cfg.pdf_dir
    os.makedirs(dest_dir, exist_ok=True)

    # 부품명 + URL 해시로 안전한 파일명 생성 (충돌/특수문자 방지)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in part_name)[:60]
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    path = os.path.join(dest_dir, f"{safe}_{digest}.pdf")

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path  # 캐시 재사용

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower() and not resp.content[:4] == b"%PDF":
            print(f"[download] PDF 아님, 건너뜀: {url} ({ctype})")
            return None
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except requests.RequestException as e:
        print(f"[download] 실패: {url} -> {e}")
        return None
