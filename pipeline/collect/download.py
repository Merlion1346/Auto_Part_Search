"""데이터시트 PDF 다운로드. URL → 로컬 파일 경로.

Mouser 등은 Cloudflare 뒤에 데이터시트를 두는 경우가 많아, 일반 요청에는 PDF 대신
챌린지 HTML이 돌아온다. 가능하면 cloudscraper(Cloudflare 우회)로 받고, 없으면
requests 세션으로 폴백한다.
"""

import hashlib
import os
from urllib.parse import urlsplit

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

_session = None
_skip_warned = False  # 차단 스킵 로그는 최초 1회만 출력


def _get_session():
    """cloudscraper 세션을 우선 생성(Cloudflare 우회), 없으면 requests 세션."""
    global _session
    if _session is None:
        try:
            import cloudscraper

            _session = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
        except Exception:
            _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


def _block_hint(content: bytes) -> str:
    head = content[:3000].lower()
    if b"access to this page has been denied" in head or b"perimeterx" in head \
            or b"px-captcha" in head or b"/_px" in head or b"human" in head:
        # Mouser는 PerimeterX 봇 차단을 쓴다. 데이터센터(예: Colab) IP는 거의 차단된다.
        return " (PerimeterX 봇 차단 — 데이터센터 IP는 우회 불가에 가까움)"
    if any(s in head for s in (b"cloudflare", b"just a moment", b"cf-challenge",
                               b"challenge-platform")):
        return " (Cloudflare 챌린지로 보임)"
    return ""


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
    parts = urlsplit(url)
    headers = {"Referer": f"{parts.scheme}://{parts.netloc}/"}

    try:
        resp = _get_session().get(url, headers=headers, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        is_pdf = "pdf" in ctype.lower() or resp.content[:4] == b"%PDF"
        if not is_pdf:
            # 차단(PerimeterX 등)은 부품마다 반복되므로 최초 1회만 안내하고 이후엔 조용히 None
            global _skip_warned
            if not _skip_warned:
                hint = _block_hint(resp.content) or f" ({ctype})"
                print(f"[download] 데이터시트 다운로드가 차단됨{hint}. "
                      f"이후 동일 건은 생략하고 API 사양만으로 진행합니다.")
                _skip_warned = True
            return None
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except requests.RequestException:
        return None
