"""LLM으로 데이터시트 텍스트에서 정형 사양을 추출한다.

OpenAI 호환 엔드포인트(DashScope의 Qwen, 로컬 vLLM 서빙 등)를 사용한다.
유통사 API가 준 정형 파라미터(api_specs)를 힌트로 함께 넣어 정확도를 높인다.

출력 스키마:
    {
        "category": str,                # 부품 분류(한국어)
        "specs": dict,                  # 정규화된 전기적 사양 (수치+단위)
        "datasheet_summary": str,       # 1~2문장 한국어 요약
        "reference_circuits": [         # 데이터시트의 대표 응용 회로(Typical Application)
            {
                "application": str,     #   용도/회로 이름
                "description": str,     #   회로 동작 1~2문장
                "bom": [                #   회로 구성 부품 목록
                    {"component": str, "value": str, "role": str}
                ]
            }
        ]
    }
"""

import json

from openai import OpenAI

from ..config import cfg

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    return _client


SYSTEM = (
    "너는 전자부품 데이터시트에서 핵심 전기적 사양을 추출하는 도우미다. "
    "반드시 유효한 JSON만 출력한다. 데이터시트에 없는 값은 추측하지 말고 생략한다."
)

PROMPT_TEMPLATE = """\
다음은 부품 '{part_name}' ({manufacturer})의 데이터시트 발췌와 유통사 제공 사양이다.

[유통사 사양]
{api_specs}

[데이터시트 발췌]
{datasheet_text}

위 정보를 바탕으로 아래 JSON 스키마로만 응답하라:
{{
  "category": "부품 분류(예: LDO 레귤레이터, N채널 MOSFET, 타이머 IC)",
  "specs": {{
    "예: output_voltage_v": 3.3,
    "예: output_current_max_a": 1.0,
    "...": "수치는 숫자로, 단위는 키 이름(_v, _a, _ma, _w, _mohm 등)에 표기"
  }},
  "datasheet_summary": "이 부품의 용도와 핵심 사양을 1~2문장 한국어로 요약",
  "reference_circuits": [
    {{
      "application": "대표 응용 회로 이름/용도 (예: 5V→3.3V 전원 회로)",
      "description": "이 회로가 무엇을 하는지 1~2문장 한국어 설명",
      "bom": [
        {{"component": "{part_name}", "value": "", "role": "이 부품의 회로 내 역할(예: 메인 LDO)"}},
        {{"component": "주변 부품명(예: 입력 커패시터)", "value": "예: 10uF", "role": "예: 입력 디커플링"}}
      ]
    }}
  ]
}}

reference_circuits 작성 규칙:
- 데이터시트의 'Typical Application'/'Application Circuit' 섹션에 실제 표기된 주변 부품만 포함한다.
- 데이터시트에 응용 회로가 없으면 reference_circuits 는 빈 배열([])로 둔다. 절대 지어내지 않는다.
- BOM 의 첫 항목은 항상 메인 부품('{part_name}')으로 한다."""


def extract_specs(part: dict, datasheet_text: str) -> dict | None:
    """part(수집기 dict) + 데이터시트 텍스트 → 정규화 사양 dict. 실패 시 None."""
    api_specs = json.dumps(part.get("api_specs", {}), ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.format(
        part_name=part.get("part_name", ""),
        manufacturer=part.get("manufacturer", ""),
        api_specs=api_specs,
        datasheet_text=datasheet_text or "(데이터시트 텍스트 없음 — 유통사 사양만 사용)",
    )
    try:
        resp = _get_client().chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[llm_extract] JSON 파싱 실패 ({part.get('part_name')}): {e}")
        return None
    except Exception as e:  # API/네트워크 오류
        print(f"[llm_extract] LLM 호출 실패 ({part.get('part_name')}): {e}")
        return None
