"""Qwen3-14B QLoRA 파인튜닝 (4-bit NF4 + LoRA).

Google Colab Pro의 A100 40GB에서 실행하도록 맞춰져 있다.
기본값(14B + batch 2 + grad_accum 8 + seq_len 2048 + gradient checkpointing)은
A100 40GB 한 장 기준이다. VRAM이 작은 GPU(L4 24GB / T4 16GB 등)에서는
--batch-size 1 / --max-seq-len 1024 로 낮춘다(T4는 bf16 미지원 → fp16 자동 사용).

  python train/preprocess.py
  python train/train_qlora.py \
      --train-file data/train.jsonl \
      --output-dir models/qwen3-14b-parts-lora

검증 환경: Colab 기본 스택(transformers 5.x, trl 1.x) + bitsandbytes>=0.46.1
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-14B")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--output-dir", default="models/qwen3-14b-parts-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    # A100 40GB 기준. VRAM이 작은 GPU(L4/T4)면 1024로 낮춘다.
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--merge", action="store_true",
                    help="학습 후 LoRA를 베이스에 병합한 체크포인트도 저장")
    return ap.parse_args()


def main():
    args = parse_args()

    # GPU가 bf16을 지원하면 bf16, 아니면 fp16 (예: Colab T4는 bf16 미지원 → fp16)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"compute dtype = {'bfloat16' if use_bf16 else 'float16'}")

    # 4-bit NF4 양자화 (QLoRA의 핵심) — 14B 모델을 A100 40GB 한 장에 적재
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA: 어텐션 + MLP 투영 레이어 전체 타깃 (Qwen3 아키텍처)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_seq_len,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,          # Google Drive 용량 절약: 체크포인트 2개만 유지
        report_to="tensorboard",
        # Colab은 vCPU가 적으므로(보통 2~8코어) 낮게 유지
        dataloader_num_workers=2,
        # bitsandbytes 4-bit에 맞는 옵티마이저. paged_*는 OOM 시 옵티마이저 상태를
        # CPU RAM으로 페이징해 VRAM 압박을 완화한다.
        optim="paged_adamw_8bit",
        # SFTTrainer가 conversational(messages) 포맷을 자동으로 chat template 적용
        model_init_kwargs={
            "quantization_config": bnb_config,
            "torch_dtype": compute_dtype,
            "device_map": "auto",
        },
    )

    trainer = SFTTrainer(
        model=args.model_id,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA 어댑터를 {args.output_dir}에 저장했습니다.")

    if args.merge:
        from peft import AutoPeftModelForCausalLM
        merged_dir = args.output_dir + "-merged"
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = model.merge_and_unload(safe_merge=True)
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"병합 모델을 {merged_dir}에 저장했습니다.")


if __name__ == "__main__":
    main()
