"""부품 카탈로그(JSON)를 QLoRA 학습용 대화 데이터셋(JSONL)으로 변환.

입력:  data/*catalog*.json  (parts_catalog.json / sample_parts_catalog.json 형식)
출력:  data/train.jsonl     (TRL conversational 포맷: {"messages": [...]})

생성하는 학습 샘플 두 종류:
  1) 단일 부품 추천: "요구사항 → 부품 1개 + 근거"
  2) 회로 BOM 추천: "회로 설명 → 전체 부품 목록(BOM)"
     (카탈로그의 reference_circuits = 데이터시트 대표 응용 회로에서 생성)

두 가지 모드:
  - 기본(규칙 기반): 요구사항/회로 설명 문장 1개를 합성
  - 증강(--augment): LLM으로 다양한 표현의 요구사항/회로 설명 N개를 생성
                     같은 부품·회로를 여러 어투/표현으로 학습시켜 일반화를 높인다.

증강은 .env 의 OpenAI 호환 LLM 설정을 사용한다 (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).

예시:
  python train/preprocess.py --augment --num-variations 5
"""

import argparse
import glob
import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SYSTEM_PROMPT = (
    "당신은 회로 설계용 부품 추천 전문가입니다. 사용자의 자연어 요구사항을 "
    "분석하여 부합하는 부품 이름을 추천하고, 추천 근거를 함께 제시하세요."
)


# --------------------------------------------------------------------------- #
# 규칙 기반 요구사항/응답 생성
# --------------------------------------------------------------------------- #
def synth_requirement(part: dict) -> str:
    """부품 사양으로부터 자연어 요구사항 문장을 합성한다(규칙 기반)."""
    cat = part.get("category", "부품")
    s = part.get("specs", {})
    bits = []
    if "output_voltage_v" in s:
        bits.append(f"출력 {s['output_voltage_v']}V")
    if "output_current_max_a" in s:
        bits.append(f"{s['output_current_max_a']}A 이상")
    if "vds_max_v" in s:
        bits.append(f"VDS {s['vds_max_v']}V급")
    if s.get("logic_level"):
        bits.append("로직 레벨 게이트 구동 가능한")
    if "supply_voltage_v" in s:
        lo, hi = s["supply_voltage_v"]
        bits.append(f"{lo}~{hi}V 전원에서 동작하는")
    cond = ", ".join(bits) if bits else "범용"
    return f"{cond} {cat}이 필요해. 추천해줘."


def synth_answer(part: dict) -> str:
    summary = part.get("datasheet_summary", "")
    return f"추천 부품: {part['part_name']}\n근거: {summary}"


def to_record(requirement: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": requirement},
            {"role": "assistant", "content": answer},
        ]
    }


# --------------------------------------------------------------------------- #
# 레퍼런스 회로 → 전체 BOM 샘플
# --------------------------------------------------------------------------- #
def circuit_requirement(circuit: dict) -> str:
    """레퍼런스 회로로부터 회로 설명형 요구사항을 합성한다(규칙 기반)."""
    app = circuit.get("application") or circuit.get("description") or "회로"
    return f"{app}를 설계하려고 해. 필요한 부품 목록(BOM)을 추천해줘."


def circuit_answer(circuit: dict) -> str:
    """회로의 BOM을 번호 매긴 목록(부품 + 값 + 역할)으로 포맷."""
    lines = ["추천 부품 목록(BOM):"]
    for i, item in enumerate(circuit.get("bom", []), 1):
        comp = item.get("component", "").strip()
        if not comp:
            continue
        val = (item.get("value") or "").strip()
        role = (item.get("role") or "").strip()
        val_s = f" {val}" if val else ""
        role_s = f" — {role}" if role else ""
        lines.append(f"{i}. {comp}{val_s}{role_s}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM 기반 요구사항 증강 (paraphrase augmentation)
# --------------------------------------------------------------------------- #
_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        )
    return _llm_client


AUG_SYSTEM = (
    "너는 회로 설계자가 부품을 찾을 때 쓸 법한 자연어 요구사항 문장을 생성하는 도우미다. "
    "반드시 유효한 JSON만 출력한다."
)

AUG_PROMPT = """\
아래 부품을 추천받을 법한 '자연어 요구사항 문장'을 {n}개 만들어라.

[부품]
- 이름: {part_name}
- 분류: {category}
- 사양: {specs}
- 설명: {summary}

조건:
- 각 문장은 이 부품이 정답이 되도록 핵심 사양(전압/전류/패키지/용도 등)을 반영한다.
- 표현을 최대한 다양화한다: 구어체/격식체, 짧은 키워드형/완결 문장형, 한국어 위주이되
  일부는 영어 용어 혼용, 실무에서 쓰는 약어, 사양을 다르게 강조한 표현 등.
- 부품 이름(정답)은 문장에 절대 포함하지 않는다.

다음 JSON 형식으로만 응답하라:
{{"requirements": ["문장1", "문장2", ...]}}"""


def llm_generate_requirements(part: dict, n: int) -> list[str]:
    """LLM으로 부품에 대한 다양한 요구사항 문장 n개를 생성. 실패 시 빈 리스트."""
    prompt = AUG_PROMPT.format(
        n=n,
        part_name=part.get("part_name", ""),
        category=part.get("category", ""),
        specs=json.dumps(part.get("specs", {}), ensure_ascii=False),
        summary=part.get("datasheet_summary", ""),
    )
    try:
        resp = _get_llm().chat.completions.create(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            messages=[
                {"role": "system", "content": AUG_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,  # 다양성 확보를 위해 높게
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        reqs = data.get("requirements", [])
        # 부품 이름 누출 방지 + 빈 문자열 제거
        name = part.get("part_name", "")
        return [r.strip() for r in reqs if r.strip() and name not in r]
    except Exception as e:  # API/파싱 오류 → 규칙 기반만 사용
        print(f"  [augment] '{part.get('part_name')}' 생성 실패: {e}")
        return []


# --------------------------------------------------------------------------- #
def build_records(part: dict, augment: bool, n: int) -> list[dict]:
    """부품 1건 → 학습 샘플 리스트. 규칙 기반 1건 + (증강 시) LLM 생성 n건."""
    requirements = [synth_requirement(part)]  # 베이스라인은 항상 포함
    if augment:
        requirements.extend(llm_generate_requirements(part, n))

    answer = synth_answer(part)
    return [to_record(r, answer) for r in _dedupe(requirements)]


def _dedupe(items: list[str]) -> list[str]:
    seen, uniq = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


AUG_CIRCUIT_PROMPT = """\
아래는 어떤 회로의 용도와 설명이다.

[회로]
- 용도: {application}
- 설명: {description}

이 회로를 만들고 싶은 사람이 쓸 법한 '요구사항 문장'을 {n}개 만들어라.

조건:
- 각 문장은 이 회로를 설계/구현하려는 의도를 담고, 필요한 부품(BOM) 추천을 요청한다.
- 표현을 다양화한다: 구어체/격식체, 짧은 형/완결 문장형, 영어 용어 혼용 등.
- 구체적인 부품 이름이나 부품 값은 문장에 포함하지 않는다(용도/기능 중심으로 기술).

다음 JSON 형식으로만 응답하라:
{{"requirements": ["문장1", "문장2", ...]}}"""


def llm_generate_circuit_requirements(circuit: dict, n: int) -> list[str]:
    """LLM으로 회로 설명형 요구사항 n개를 생성. 실패 시 빈 리스트."""
    prompt = AUG_CIRCUIT_PROMPT.format(
        n=n,
        application=circuit.get("application", ""),
        description=circuit.get("description", ""),
    )
    try:
        resp = _get_llm().chat.completions.create(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            messages=[
                {"role": "system", "content": AUG_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return [r.strip() for r in data.get("requirements", []) if r.strip()]
    except Exception as e:
        print(f"  [augment-circuit] 생성 실패: {e}")
        return []


def build_circuit_records(part: dict, augment: bool, n: int) -> list[dict]:
    """부품의 reference_circuits → '회로 설명 → 전체 BOM' 샘플 리스트."""
    records = []
    for circuit in part.get("reference_circuits", []):
        if not circuit.get("bom"):
            continue
        answer = circuit_answer(circuit)
        reqs = [circuit_requirement(circuit)]
        if augment:
            reqs.extend(llm_generate_circuit_requirements(circuit, n))
        records.extend(to_record(r, answer) for r in _dedupe(reqs))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="data/*catalog*.json")
    ap.add_argument("--output", default="data/train.jsonl")
    ap.add_argument("--augment", action="store_true",
                    help="LLM으로 요구사항을 다양화하여 부품당 샘플 수를 늘린다")
    ap.add_argument("--num-variations", type=int, default=5,
                    help="부품당 LLM 생성 요구사항 개수(--augment 시)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"입력 파일을 찾지 못했습니다: {args.input_glob}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_parts, n_part_samples, n_circuit_samples = 0, 0, 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, encoding="utf-8") as f:
                parts = json.load(f)
            for part in parts:
                part_recs = build_records(part, args.augment, args.num_variations)
                circuit_recs = build_circuit_records(part, args.augment, args.num_variations)
                for rec in part_recs + circuit_recs:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_part_samples += len(part_recs)
                n_circuit_samples += len(circuit_recs)
                n_parts += 1
                if args.augment:
                    print(f"  [{part.get('part_name')}] 부품샘플 {len(part_recs)} "
                          f"+ 회로샘플 {len(circuit_recs)}")

    total = n_part_samples + n_circuit_samples
    print(f"\n부품 {n_parts}개 → 학습 샘플 {total}건"
          f" (단일부품 {n_part_samples} + 회로BOM {n_circuit_samples})을"
          f" {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
