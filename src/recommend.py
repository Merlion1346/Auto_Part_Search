"""학습된 LoRA 어댑터로 부품을 추천한다.

  python src/recommend.py --adapter models/qwen3-8b-parts-lora \
      "5V를 3.3V로 변환하고 1A 이상 출력하는 LDO 추천해줘"
"""

import argparse

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "당신은 회로 설계용 부품 추천 전문가입니다. 사용자의 자연어 요구사항을 "
    "분석하여 부합하는 부품 이름을 추천하고, 추천 근거를 함께 제시하세요."
)


def load(adapter_path: str):
    base_id = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    return model, tokenizer


def recommend(model, tokenizer, requirement: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": requirement},
    ]
    # Qwen3는 thinking 모드 지원. 추천 근거 추론이 필요하면 enable_thinking=True 로 변경.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/qwen3-8b-parts-lora")
    ap.add_argument("requirement", help="자연어 요구사항")
    args = ap.parse_args()

    model, tokenizer = load(args.adapter)
    print(recommend(model, tokenizer, args.requirement))


if __name__ == "__main__":
    main()
