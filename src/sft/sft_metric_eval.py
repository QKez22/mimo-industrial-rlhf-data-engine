"""
new_rlhf SFT 自动指标评估脚本（Base vs SFT）

一、脚本用途
1. 在局部验证集_150条.csv 上，分别让 Base 模型和 SFT 模型各生成一份回答。
2. 将模型回答与数据中的 answer1 进行逐题比较，计算 BLEU-4 与 ROUGE-L。
3. 输出逐题明细和汇总表，便于把 SFT 阶段的结果整理成更标准的实验指标。

二、默认输入
1. 问题与参考答案文件：data/01_测试集与验证集/局部验证集_150条.csv
2. Base 模型：本地基础模型
3. SFT 模型：model/sft_v0/merged
4. 参考答案列：answer1

三、默认输出
1. 明细文件：
   logs/sft/sft_metric_eval_details.csv
2. 汇总文件：
   logs/sft/sft_metric_eval_summary.csv

四、指标说明
1. BLEU-4：
   用于衡量模型回答与参考答案之间的 n-gram 精确匹配程度。
2. ROUGE-L：
   用于衡量模型回答与参考答案之间的最长公共子序列覆盖情况。
3. 由于当前数据以中文为主，且不额外依赖第三方分词库，本脚本默认采用“去空白后的字级切分”。
   这意味着每个汉字、字母或数字字符会作为一个 token 参与指标计算。
4. 该口径适合当前中文问答实验做稳定对比，但数值不应与英文单词级 BLEU/ROUGE 直接横向比较。

五、默认参数
1. sample_size=150：默认使用局部验证集_150条.csv 全量 150 条。
2. max_new_tokens=180：控制生成答案的最大新增长度。
3. do_sample=False：默认采用贪心生成，保证实验复现性。
4. 先加载 Base，再加载 SFT，避免同时加载两个大模型带来的显存压力。
"""

import argparse
import csv
import gc
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SFT 自动指标评估：Base vs SFT（BLEU-4 / ROUGE-L）")
    parser.add_argument("--base_model_path", default=str(LOCAL_MODEL_PATH))
    parser.add_argument("--sft_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--question_data_path", default=str(LOCAL_VAL_FILE))
    parser.add_argument("--output_detail_csv", default=str(SFT_LOG_DIR / "sft_metric_eval_details.csv"))
    parser.add_argument("--output_summary_csv", default=str(SFT_LOG_DIR / "sft_metric_eval_summary.csv"))
    parser.add_argument("--sample_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """做最小清洗，去掉首尾空白并移除中间空白字符。"""
    return "".join(str(text).strip().split())


def tokenize_zh_chars(text: str) -> List[str]:
    """将中文文本按字符级切分，避免额外依赖第三方分词工具。"""
    return list(normalize_text(text))


def build_prompt(question: str) -> str:
    """统一构造 Qwen 对话格式输入。"""
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    """使用指定模型生成单条回答。"""
    inputs = tokenizer(build_prompt(question), return_tensors="pt").to(model.device)
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


def load_question_rows(path: str, sample_size: int, seed: int) -> List[Dict[str, str]]:
    """读取验证集，并保证 question 与 answer1 可用。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = ["question", "answer1"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"验证集缺少列：{missing}；文件：{path}")

    df = df.dropna(subset=required_columns).reset_index(drop=True)
    df["question"] = df["question"].astype(str).str.strip()
    df["answer1"] = df["answer1"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer1"] != "")].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"验证集为空：{path}")

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


def get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """统计 n-gram 频次。"""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu_4(reference_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    """计算单条样本的 BLEU-4，使用平滑避免短句直接为 0。"""
    if not candidate_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, 5):
        candidate_ngrams = get_ngrams(candidate_tokens, n)
        reference_ngrams = get_ngrams(reference_tokens, n)
        total = sum(candidate_ngrams.values())
        if total == 0:
            # 使用平滑项，避免 0 导致整体分数塌缩。
            precisions.append(1.0 / (2.0 ** n))
            continue

        overlap = sum(min(count, reference_ngrams[gram]) for gram, count in candidate_ngrams.items())
        precisions.append((overlap + 1.0) / (total + 1.0))

    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    if cand_len == 0:
        return 0.0

    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_len / max(cand_len, 1)))

    score = brevity_penalty * math.exp(sum(math.log(max(p, 1e-12)) for p in precisions) / 4.0)
    return float(score)


def lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    """计算最长公共子序列长度。"""
    if not tokens_a or not tokens_b:
        return 0

    dp = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        prev = 0
        for idx, token_b in enumerate(tokens_b, start=1):
            temp = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(reference_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    """计算 ROUGE-L 的 F1 形式分数。"""
    if not reference_tokens or not candidate_tokens:
        return 0.0

    lcs = lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def mean(values: List[float]) -> float:
    """求均值。"""
    return sum(values) / max(len(values), 1)


def cleanup_model(model) -> None:
    """释放模型资源，避免 Base 与 SFT 同时长时间占用显存。"""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    model_name: str,
    model,
    tokenizer,
    rows: List[Dict[str, str]],
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    """对单个模型完成整批生成与指标计算。"""
    evaluated_rows: List[Dict[str, str]] = []
    total = len(rows)

    for idx, item in enumerate(rows, start=1):
        question = str(item["question"]).strip()
        reference_answer = str(item["reference_answer"]).strip()
        model_answer = generate_answer(model, tokenizer, question, max_new_tokens)

        reference_tokens = tokenize_zh_chars(reference_answer)
        model_tokens = tokenize_zh_chars(model_answer)
        bleu_4 = sentence_bleu_4(reference_tokens, model_tokens)
        rouge_l = rouge_l_f1(reference_tokens, model_tokens)

        evaluated_rows.append(
            {
                "id": item["id"],
                "question_type": item["question_type"],
                "question": question,
                "reference_answer": reference_answer,
                "model_name": model_name,
                "model_answer": model_answer,
                "answer_length": len(normalize_text(model_answer)),
                "reference_length": len(reference_tokens),
                "bleu_4": bleu_4,
                "rouge_l": rouge_l,
            }
        )
        print(
            f"[{model_name}] {idx}/{total} "
            f"BLEU-4={bleu_4:.4f} ROUGE-L={rouge_l:.4f} "
            f"answer_len={len(normalize_text(model_answer))}"
        )

    return evaluated_rows


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.output_detail_csv).parent)
    ensure_dir(Path(args.output_summary_csv).parent)

    input_rows = load_question_rows(args.question_data_path, args.sample_size, args.seed)
    prepared_rows = [
        {
            "id": idx,
            "question_type": item.get("question_type", ""),
            "question": str(item["question"]).strip(),
            "reference_answer": str(item["answer1"]).strip(),
        }
        for idx, item in enumerate(input_rows, start=1)
    ]

    print(f"开始进行 SFT 自动指标评估，共 {len(prepared_rows)} 条样本。")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载 Base 模型。")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    base_model.eval()
    base_rows = evaluate_model("base", base_model, tokenizer, prepared_rows, args.max_new_tokens)
    cleanup_model(base_model)

    print("加载 SFT 模型。")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    sft_model.eval()
    sft_rows = evaluate_model("sft", sft_model, tokenizer, prepared_rows, args.max_new_tokens)
    cleanup_model(sft_model)

    all_rows = base_rows + sft_rows
    with open(args.output_detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question_type",
                "question",
                "reference_answer",
                "model_name",
                "model_answer",
                "answer_length",
                "reference_length",
                "bleu_4",
                "rouge_l",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for model_name, model_rows in [("base", base_rows), ("sft", sft_rows)]:
        summary_rows.extend(
            [
                {"model_name": model_name, "metric": "sample_count", "value": len(model_rows)},
                {"model_name": model_name, "metric": "avg_bleu_4", "value": mean([row["bleu_4"] for row in model_rows])},
                {"model_name": model_name, "metric": "avg_rouge_l", "value": mean([row["rouge_l"] for row in model_rows])},
                {
                    "model_name": model_name,
                    "metric": "avg_answer_length",
                    "value": mean([float(row["answer_length"]) for row in model_rows]),
                },
            ]
        )

    with open(args.output_summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"逐题明细已保存：{args.output_detail_csv}")
    print(f"汇总结果已保存：{args.output_summary_csv}")


if __name__ == "__main__":
    main()
