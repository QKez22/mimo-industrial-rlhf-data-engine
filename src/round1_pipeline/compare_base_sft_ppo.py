"""
new_rlhf 第一轮效果对比脚本（Base / SFT / PPO）

一、脚本用途
1. 在同一批问题上分别生成 Base、SFT、PPO 三个模型的回答。
2. 使用已经训练好的 RM 对三组回答统一打分，形成可量化的效果对比。
3. 输出详细结果表和汇总结果表，直接展示第一轮 RLHF 的最终效果。

二、默认输入
1. 问题池：data/01_测试集与验证集/局部验证集_150条.csv
2. Base 模型：本地基础模型
3. SFT 模型：model/sft_v0/merged
4. PPO 模型：model/ppo_v1/final_lora，挂载在 SFT merged 上
5. RM 打分器：model/rm_v1/final_lora，挂载在 SFT merged 上

三、默认输出
1. 明细文件：
   logs/eval/base_sft_ppo_generations.csv
2. 汇总文件：
   logs/eval/base_sft_ppo_summary.csv

四、关键实现说明
1. 为了避免显存或内存峰值过高，本脚本不会同时加载 Base、SFT、PPO 三个生成模型。
2. 实际执行顺序是：先加载 Base 生成完全部样本并释放，再加载 SFT，再加载 PPO。
3. RM 打分器单独加载一次，用来对三类回答统一打分，保证比较口径一致。
4. 默认使用局部验证集_150条.csv 全量 150 条样本做最终效果验证。
5. 如果后续只是想做快速抽查，再手动传入更小的 --sample_size。
6. 默认只做贪心生成，不做随机采样，便于复现实验结果。
"""

import argparse
import csv
import gc
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import (
    EVAL_LOG_DIR,
    LOCAL_MODEL_PATH,
    LOCAL_VAL_FILE,
    PPO_V1_MODEL_DIR,
    RM_V1_MODEL_DIR,
    SFT_MODEL_DIR,
    ensure_dir,
)


def parse_args() -> argparse.Namespace:
    """解析对比脚本参数。"""
    parser = argparse.ArgumentParser(description="Base / SFT / PPO 结果对比")
    parser.add_argument("--question_data_path", default=str(LOCAL_VAL_FILE))
    parser.add_argument("--base_model_path", default=str(LOCAL_MODEL_PATH))
    parser.add_argument("--sft_merged_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_lora_path", default=str(PPO_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--rm_base_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--rm_lora_path", default=str(RM_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--sample_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--max_score_length", type=int, default=512)
    parser.add_argument("--output_detail_csv", default=str(EVAL_LOG_DIR / "base_sft_ppo_generations.csv"))
    parser.add_argument("--output_summary_csv", default=str(EVAL_LOG_DIR / "base_sft_ppo_summary.csv"))
    parser.add_argument("--gen_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rm_device", default="cpu")
    return parser.parse_args()


def build_generation_prompt(question: str) -> str:
    """构造生成用 prompt。"""
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"


def build_rm_prompt(question: str, answer: str) -> str:
    """构造 RM 打分用 prompt。"""
    return (
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer.strip()}<|im_end|>"
    )


def load_questions(path: str, sample_size: int, seed: int) -> List[Dict[str, str]]:
    """读取问题池并抽样。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "question" not in df.columns:
        raise ValueError(f"问题数据缺少列 question；文件：{path}")

    df = df.dropna(subset=["question"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"问题池为空：{path}")

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


def cleanup_model(model) -> None:
    """释放模型占用的显存与内存。"""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, device: str, max_new_tokens: int) -> str:
    """调用指定模型生成回答。"""
    inputs = tokenizer(build_generation_prompt(question), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant\n"
    answer = text.split(marker, 1)[1] if marker in text else text
    return answer.replace("<|im_end|>", "").strip()


@torch.no_grad()
def rm_score(rm_model, rm_tokenizer, device: str, question: str, answer: str, max_length: int) -> float:
    """用 RM 对单条回答打分。"""
    encoded = rm_tokenizer(
        build_rm_prompt(question, answer),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return float(rm_model(**encoded).logits.float().view(-1).item())


def avg(values: List[float]) -> float:
    """计算均值，空列表时返回 0。"""
    return sum(values) / max(len(values), 1)


def load_generation_model(model_label: str, args: argparse.Namespace, dtype: torch.dtype):
    """按模型类型加载 Base、SFT 或 PPO 生成模型。"""
    if model_label == "base":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    elif model_label == "sft":
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_merged_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    elif model_label == "ppo":
        ppo_base = AutoModelForCausalLM.from_pretrained(
            args.sft_merged_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(ppo_base, args.ppo_lora_path)
    else:
        raise ValueError(f"不支持的模型标签：{model_label}")

    model.to(args.gen_device)
    model.eval()
    return model


def run_generation_pass(
    model_label: str,
    model,
    tokenizer,
    rm_model,
    rm_tokenizer,
    questions: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[List[str], List[float], List[float]]:
    """完成单个模型的整批生成与打分。"""
    answers: List[str] = []
    scores: List[float] = []
    lengths: List[float] = []
    total = len(questions)

    print(f"[Compare] 开始处理 {model_label.upper()}，共 {total} 条样本。")
    for idx, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        answer = generate_answer(model, tokenizer, question, args.gen_device, args.max_new_tokens)
        score = rm_score(rm_model, rm_tokenizer, args.rm_device, question, answer, args.max_score_length)

        answers.append(answer)
        scores.append(score)
        lengths.append(float(len(answer)))

        print(
            f"[Compare] {model_label.upper()} {idx}/{total} "
            f"rm_score={score:.4f} answer_len={len(answer)}"
        )

    return answers, scores, lengths


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_dir(Path(args.output_detail_csv).parent)
    ensure_dir(Path(args.output_summary_csv).parent)

    gen_dtype = torch.float16 if args.gen_device == "cuda" else torch.float32
    score_dtype = torch.float32

    questions = load_questions(args.question_data_path, args.sample_size, args.seed)
    print(f"[Compare] 已抽取 {len(questions)} 条问题，开始进行 Base / SFT / PPO 对比。")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_merged_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_lora_path, trust_remote_code=True, use_fast=False)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    print("[Compare] 加载 RM 打分器。")
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.rm_base_model_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=score_dtype,
    )
    rm_base.config.pad_token_id = rm_tokenizer.pad_token_id
    rm_model = PeftModel.from_pretrained(rm_base, args.rm_lora_path)
    rm_model.to(args.rm_device)
    rm_model.eval()

    model_answers: Dict[str, List[str]] = {}
    model_scores: Dict[str, List[float]] = {}
    model_lengths: Dict[str, List[float]] = {}

    for model_label in ["base", "sft", "ppo"]:
        print(f"[Compare] 加载 {model_label.upper()} 模型。")
        model = load_generation_model(model_label, args, gen_dtype)
        answers, scores, lengths = run_generation_pass(
            model_label=model_label,
            model=model,
            tokenizer=tokenizer,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            questions=questions,
            args=args,
        )
        model_answers[model_label] = answers
        model_scores[model_label] = scores
        model_lengths[model_label] = lengths
        cleanup_model(model)
        print(f"[Compare] {model_label.upper()} 已完成并释放模型。")

    rows = []
    for idx, item in enumerate(questions, start=1):
        base_score = model_scores["base"][idx - 1]
        sft_score = model_scores["sft"][idx - 1]
        ppo_score = model_scores["ppo"][idx - 1]

        rows.append(
            {
                "id": idx,
                "question_id": item.get("question_id", ""),
                "question_type": item.get("question_type", ""),
                "question": str(item["question"]).strip(),
                "reference_answer": item.get("answer1", ""),
                "base_answer": model_answers["base"][idx - 1],
                "sft_answer": model_answers["sft"][idx - 1],
                "ppo_answer": model_answers["ppo"][idx - 1],
                "base_rm_score": base_score,
                "sft_rm_score": sft_score,
                "ppo_rm_score": ppo_score,
                "sft_minus_base": sft_score - base_score,
                "ppo_minus_sft": ppo_score - sft_score,
                "ppo_minus_base": ppo_score - base_score,
            }
        )

    with open(args.output_detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question_id",
                "question_type",
                "question",
                "reference_answer",
                "base_answer",
                "sft_answer",
                "ppo_answer",
                "base_rm_score",
                "sft_rm_score",
                "ppo_rm_score",
                "sft_minus_base",
                "ppo_minus_sft",
                "ppo_minus_base",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [
        {"metric": "sample_count", "value": len(rows)},
        {"metric": "base_avg_rm_score", "value": avg(model_scores["base"])},
        {"metric": "sft_avg_rm_score", "value": avg(model_scores["sft"])},
        {"metric": "ppo_avg_rm_score", "value": avg(model_scores["ppo"])},
        {
            "metric": "sft_minus_base_avg",
            "value": avg([s - b for s, b in zip(model_scores["sft"], model_scores["base"])]),
        },
        {
            "metric": "ppo_minus_sft_avg",
            "value": avg([p - s for p, s in zip(model_scores["ppo"], model_scores["sft"])]),
        },
        {
            "metric": "ppo_minus_base_avg",
            "value": avg([p - b for p, b in zip(model_scores["ppo"], model_scores["base"])]),
        },
        {
            "metric": "sft_better_than_base_ratio",
            "value": avg([1.0 if s > b else 0.0 for s, b in zip(model_scores["sft"], model_scores["base"])]),
        },
        {
            "metric": "ppo_better_than_sft_ratio",
            "value": avg([1.0 if p > s else 0.0 for p, s in zip(model_scores["ppo"], model_scores["sft"])]),
        },
        {
            "metric": "ppo_better_than_base_ratio",
            "value": avg([1.0 if p > b else 0.0 for p, b in zip(model_scores["ppo"], model_scores["base"])]),
        },
        {"metric": "base_avg_answer_len", "value": avg(model_lengths["base"])},
        {"metric": "sft_avg_answer_len", "value": avg(model_lengths["sft"])},
        {"metric": "ppo_avg_answer_len", "value": avg(model_lengths["ppo"])},
    ]

    with open(args.output_summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    cleanup_model(rm_model)
    cleanup_model(rm_base)

    print(f"[Compare] 对比明细已保存: {args.output_detail_csv}")
    print(f"[Compare] 对比汇总已保存: {args.output_summary_csv}")


if __name__ == "__main__":
    main()
