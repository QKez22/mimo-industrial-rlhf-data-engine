"""
new_rlhf SFT 推理对比脚本

一、脚本用途
1. 对比基础模型与 SFT-LoRA 模型在同一批问题上的回答差异。
2. 主要用于 SFT 训练后做快速人工质检，不用于论文最终评测。
3. 优先直接加载 merged SFT 完整模型，避免 base + LoRA 推理时的显存分发问题。

二、默认输入
1. base_model_path: 本地基础模型
2. sft_model_path:  model/sft_v0/merged
3. question_data_path: 局部验证集_150条.csv

三、默认输出
1. logs/sft/sft_compare_results.csv
2. 输出字段：
   question_type / question / reference_answer / base_answer / sft_answer

四、运行说明
1. 默认随机抽样 20 条问题进行对比，可通过 --sample_size 修改。
2. 该脚本默认用 merged 模型做推理，更贴近后续 RM / PPO 的真实使用方式。
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import LOCAL_MODEL_PATH, LOCAL_VAL_FILE, SFT_LOG_DIR, SFT_MODEL_DIR, ensure_dir


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def build_prompt(question: str) -> str:
    """构建统一的 Qwen 对话 prompt。"""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 180) -> str:
    """使用给定模型生成回答。"""
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant\n"
    answer = generated.split(marker, 1)[1] if marker in generated else generated
    return answer.replace("<|im_end|>", "").strip()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SFT 推理对比：基础模型 vs SFT merged 模型")
    parser.add_argument("--base_model_path", default=str(LOCAL_MODEL_PATH))
    parser.add_argument("--sft_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--question_data_path", default=str(LOCAL_VAL_FILE))
    parser.add_argument("--output_csv", default=str(SFT_LOG_DIR / "sft_compare_results.csv"))
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    return parser.parse_args()


def load_question_rows(path: str, sample_size: int, seed: int):
    """读取问题池并按需要随机抽样。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "question" not in df.columns:
        raise ValueError(f"问题列缺失: {path}")
    df = df.dropna(subset=["question"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"问题池为空: {path}")
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.output_csv).parent)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    base_model.eval()

    # SFT 模型：直接加载 merged 完整模型。
    # 这样更接近后续 RM / PPO 的实际使用方式，也避免 Windows 下的 offload_dir 问题。
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    sft_model.eval()

    rows = []
    for item in load_question_rows(args.question_data_path, args.sample_size, args.seed):
        question = str(item["question"]).strip()
        rows.append(
            {
                "question_type": item.get("question_type", ""),
                "question": question,
                "reference_answer": item.get("answer1", ""),
                "base_answer": generate_answer(base_model, tokenizer, question, args.max_new_tokens),
                "sft_answer": generate_answer(sft_model, tokenizer, question, args.max_new_tokens),
            }
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_type", "question", "reference_answer", "base_answer", "sft_answer"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"SFT 对比结果已保存: {args.output_csv}")


if __name__ == "__main__":
    main()
