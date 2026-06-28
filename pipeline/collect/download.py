"""데이터시트 PDF 다운로드. URL → 로컬 파일 경로."""

import hashlib
import os

import requests

from ..config import cfg

# 실제 브라우저처럼 보이는 헤더. 봇 UA로는 Mouser/Cloudflare가 PDF 대신
# 안티봇 HTML 페이지를 반환해 다운로드가 'PDF 아님'으로 건너뛰어진다.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


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

    # 일부 사이트는 같은 호스트의 Referer가 있어야 PDF를 내준다
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    headers = {**HEADERS, "Referer": f"{parts.scheme}://{parts.netloc}/"}

    try:
        resp = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower() and not resp.content[:4] == b"%PDF":
            print(f"[download] PDF 아님, 건너뜀: {url} "
                  f"({ctype}, {len(resp.content)} bytes)")
            return None
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except requests.RequestException as e:
        print(f"[download] 실패: {url} -> {e}")
        return None
