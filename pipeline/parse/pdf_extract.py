"""OpenDataLoader로 데이터시트 PDF에서 텍스트(Markdown)를 추출한다.

opendataloader_pdf 는 PDF를 정확한 읽기 순서/표 구조를 보존한 Markdown 으로 변환한다.
변환 결과는 파일로 떨어지므로(문자열 직접 반환 없음) 임시 디렉터리에 출력 후 읽어들인다.

요구사항:
  - pip install opendataloader-pdf
  - Java 11+ (JVM 기반). `java -version` 으로 확인, 없으면 https://adoptium.net 에서 설치.
"""

import tempfile
from pathlib import Path

import opendataloader_pdf


def extract_text(pdf_path: str, max_chars: int = 16000) -> str:
    """PDF를 Markdown으로 변환해 텍스트를 반환. 앞부분 max_chars 만 사용(토큰 절약)."""
    with tempfile.TemporaryDirectory() as out_dir:
        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=out_dir,
            format="markdown",
            quiet=True,
        )
        # 변환된 .md 파일을 찾아 합친다(보통 입력 PDF당 1개 생성).
        md_files = sorted(Path(out_dir).rglob("*.md")) + sorted(Path(out_dir).rglob("*.markdown"))
        text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in md_files)

    return text[:max_chars]
