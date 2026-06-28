"""데이터 수집·파싱 파이프라인 오케스트레이터.

검색 키워드 목록 → 유통사 API 검색 → 데이터시트 다운로드 → 텍스트 추출
→ LLM 사양 추출 → data/parts_catalog.json 생성.

생성된 카탈로그는 train/preprocess.py 의 입력으로 바로 사용된다.

사용:
    python -m pipeline.build_catalog --source digikey \
        --keywords "3.3V LDO regulator" "logic level N-channel MOSFET" \
        --limit 15

    # 또는 키워드를 파일(한 줄에 하나)로
    python -m pipeline.build_catalog --source mouser --keywords-file keywords.txt
"""

import argparse
import json
import os
import random
import time

from .collect import get_client
from .collect.download import download_pdf
from .config import cfg
from .parse import extract_specs, extract_text


def _sleep(delay: float) -> None:
    """부품 간 지연. 기계적으로 보이지 않게 ±30% 지터를 섞는다."""
    if delay <= 0:
        return
    time.sleep(delay * random.uniform(0.7, 1.3))


def build(source: str, keywords: list[str], limit: int, out_path: str,
          delay: float = 3.0) -> int:
    client = get_client(source)
    catalog: dict[str, dict] = {}  # part_name -> record (중복 제거)
    n_with_pdf = 0  # 데이터시트를 실제로 받은 부품 수
    first = True  # 첫 부품 앞에는 지연을 두지 않는다

    for kw in keywords:
        print(f"\n[검색] ({source}) '{kw}'")
        try:
            results = client.search(kw, limit=limit)
        except Exception as e:
            print(f"  검색 실패: {e}")
            continue
        print(f"  {len(results)}개 부품 수신")

        for part in results:
            name = part.get("part_name")
            if not name or name in catalog:
                continue

            if not first:
                _sleep(delay)  # 부품(데이터시트 처리) 사이 간격
            first = False

            pdf_path = download_pdf(part.get("datasheet_url"), name)
            text = ""
            if pdf_path:
                try:
                    text = extract_text(pdf_path)
                    n_with_pdf += 1
                except Exception as e:
                    print(f"  [{name}] PDF 파싱 실패: {e}")
            # 데이터시트를 못 받아도 유통사 API 사양만으로 카탈로그를 만든다(조용히 진행)

            extracted = extract_specs(part, text)
            if not extracted:
                continue

            catalog[name] = {
                "part_name": name,
                "manufacturer": part.get("manufacturer", ""),
                "category": extracted.get("category", part.get("category", "")),
                "specs": extracted.get("specs", {}),
                "datasheet_summary": extracted.get("datasheet_summary", ""),
                "reference_circuits": extracted.get("reference_circuits", []),
                "datasheet_url": part.get("datasheet_url"),
                "source": source,
            }
            print(f"  [OK] {name}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    records = list(catalog.values())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    n = len(records)
    print(f"\n총 {n}개 부품을 {out_path} 에 저장했습니다 "
          f"(데이터시트 {n_with_pdf}개 확보 / API 사양만 {n - n_with_pdf}개).")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["digikey", "mouser"], default="mouser")
    ap.add_argument("--keywords", nargs="*", default=[], help="검색 키워드(공백 구분)")
    ap.add_argument("--keywords-file", help="키워드 파일(한 줄에 하나)")
    ap.add_argument("--limit", type=int, default=25, help="키워드당 최대 부품 수")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="부품 사이 지연(초, 기본 3). rate-limit 완화용 (±30%% 지터 자동 적용). 0이면 끔")
    ap.add_argument("--out", default=cfg.catalog_path)
    args = ap.parse_args()

    keywords = list(args.keywords)
    if args.keywords_file:
        with open(args.keywords_file, encoding="utf-8") as f:
            keywords += [ln.strip() for ln in f if ln.strip()]
    if not keywords:
        raise SystemExit("키워드가 필요합니다 (--keywords 또는 --keywords-file).")

    build(args.source, keywords, args.limit, args.out, delay=args.delay)


if __name__ == "__main__":
    main()
