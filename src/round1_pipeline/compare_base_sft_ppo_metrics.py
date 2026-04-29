"""
new_rlhf 第一轮自动指标对比脚本（Base / SFT / PPO）

一、脚本用途
1. 在局部验证集_150条.csv 上，分别让 Base、SFT、PPO 三个模型各生成一份回答。
2. 将三组回答分别与 answer1 逐题比较，计算 BLEU-4 与 ROUGE-L。
3. 输出明细表与汇总表，用于量化比较三个阶段模型的自动文本匹配指标。

二、默认输入
1. 评测集：data/01_测试集与验证集/局部验证集_150条.csv
2. Base 模型：本地基础模型
3. SFT 模型：model/sft_v0/merged
4. PPO 模型：model/ppo_v1/final_lora，挂载在 SFT merged 上
5. 参考答案列：answer1

三、默认输出
1. 明细文件：
   logs/eval/base_sft_ppo_metric_details.csv
2. 汇总文件：
   logs/eval/base_sft_ppo_metric_summary.csv

四、指标口径
1. BLEU-4 与 ROUGE-L 均采用中文字符级切分。
2. 默认使用局部验证集 150 条全量样本。
3. 为了避免显存压力，Base、SFT、PPO 三个生成模型按顺序依次加载与释放。
4. 默认优先复用已经计算完成的 Base/SFT 明细结果，只补跑 PPO，再自动汇总三模型指标。
5. 默认优先读取 model/ppo_v1/merged 作为 PPO 完整模型，适合在 PyCharm 中直接点运行。
6. 只有在 merged PPO 不存在时，才回退到“SFT merged 底模 + PPO LoRA”方式恢复。
7. 每完成一个模型的评测，脚本都会立刻刷新明细文件和汇总文件，避免中途失败导致前面结果丢失。
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import EVAL_LOG_DIR, LOCAL_MODEL_PATH, LOCAL_VAL_FILE, PPO_V1_MODEL_DIR, SFT_MODEL_DIR, ensure_dir


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def parse_args() -> argparse.Namespace:
    """解析自动指标对比参数。"""
    parser = argparse.ArgumentParser(description="Base / SFT / PPO 自动指标对比（BLEU-4 / ROUGE-L）")
    parser.add_argument("--base_model_path", default=str(LOCAL_MODEL_PATH))
    parser.add_argument("--sft_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_lora_path", default=str(PPO_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--question_data_path", default=str(LOCAL_VAL_FILE))
    parser.add_argument("--output_detail_csv", default=str(EVAL_LOG_DIR / "base_sft_ppo_metric_details.csv"))
    parser.add_argument("--output_summary_csv", default=str(EVAL_LOG_DIR / "base_sft_ppo_metric_summary.csv"))
    parser.add_argument("--sample_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument(
        "--ppo_merged_model_path",
        default=str(PPO_V1_MODEL_DIR / "merged"),
        help="默认直接读取 model/ppo_v1/merged，适合在 PyCharm 中直接运行。",
    )
    parser.add_argument("--ppo_offload_dir", default=str(EVAL_LOG_DIR / "ppo_metric_offload"))
    parser.add_argument(
        "--reuse_existing_detail",
        action="store_true",
        default=True,
        help="默认复用已存在的明细 CSV，只补跑缺失模型。",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return "".join(str(text).strip().split())


def tokenize_zh_chars(text: str) -> List[str]:
    return list(normalize_text(text))


def build_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    inputs = tokenizer(build_prompt(question), return_tensors="pt").to(model.device)
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


def load_question_rows(path: str, sample_size: int, seed: int) -> List[Dict[str, str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = ["question", "answer1"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"评测集缺少列：{missing}；文件：{path}")

    df = df.dropna(subset=required_columns).reset_index(drop=True)
    df["question"] = df["question"].astype(str).str.strip()
    df["answer1"] = df["answer1"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer1"] != "")].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"评测集为空：{path}")

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


def get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu_4(reference_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    if not candidate_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, 5):
        candidate_ngrams = get_ngrams(candidate_tokens, n)
        reference_ngrams = get_ngrams(reference_tokens, n)
        total = sum(candidate_ngrams.values())
        if total == 0:
            precisions.append(1.0 / (2.0 ** n))
            continue
        overlap = sum(min(count, reference_ngrams[gram]) for gram, count in candidate_ngrams.items())
        precisions.append((overlap + 1.0) / (total + 1.0))

    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / max(cand_len, 1)))
    return float(brevity_penalty * math.exp(sum(math.log(max(p, 1e-12)) for p in precisions) / 4.0))


def lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
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
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_ppo_model(args: argparse.Namespace):
    """稳定加载 PPO 模型。

    设计原因：
    1. PPO 当前保存的是 LoRA 适配器，不是完整 merged 模型。
    2. 当 device_map="auto" 触发 CPU/磁盘卸载时，PEFT 恢复适配器需要显式提供 offload_dir。
    3. 这里统一收口，避免主流程里散落特殊兼容逻辑。
    """
    if args.ppo_merged_model_path and Path(args.ppo_merged_model_path).exists():
        model = AutoModelForCausalLM.from_pretrained(
            args.ppo_merged_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        model.eval()
        return model

    offload_dir = ensure_dir(Path(args.ppo_offload_dir))
    ppo_base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        offload_folder=str(offload_dir),
        offload_state_dict=True,
    )
    ppo_model = PeftModel.from_pretrained(
        ppo_base,
        args.ppo_lora_path,
        offload_folder=str(offload_dir),
    )
    ppo_model.eval()
    return ppo_model


def load_existing_detail_rows(path: Path) -> List[Dict[str, object]]:
    """读取已存在的明细结果，便于断点续跑。"""
    if not path.exists():
        return []
    return pd.read_csv(path, encoding="utf-8-sig").to_dict("records")


def save_detail_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    """统一写出明细结果。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
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
        writer.writerows(rows)


def build_summary_rows(all_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """根据明细结果生成汇总表。"""
    summary_rows = []
    for model_name in ["base", "sft", "ppo"]:
        model_rows = [row for row in all_rows if row["model_name"] == model_name]
        if not model_rows:
            continue
        summary_rows.extend(
            [
                {"model_name": model_name, "metric": "sample_count", "value": len(model_rows)},
                {"model_name": model_name, "metric": "avg_bleu_4", "value": mean([float(row["bleu_4"]) for row in model_rows])},
                {"model_name": model_name, "metric": "avg_rouge_l", "value": mean([float(row["rouge_l"]) for row in model_rows])},
                {"model_name": model_name, "metric": "avg_answer_length", "value": mean([float(row["answer_length"]) for row in model_rows])},
            ]
        )
    return summary_rows


def save_summary_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    """写出汇总结果。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def persist_progress(detail_path: Path, summary_path: Path, rows: List[Dict[str, object]]) -> None:
    """每完成一个模型后立刻持久化当前进度。"""
    sorted_rows = sorted(rows, key=lambda row: (str(row["model_name"]), int(row["id"])))
    save_detail_rows(detail_path, sorted_rows)
    save_summary_rows(summary_path, build_summary_rows(sorted_rows))


def evaluate_model(model_name: str, model, tokenizer, rows: List[Dict[str, str]], max_new_tokens: int) -> List[Dict[str, object]]:
    evaluated_rows: List[Dict[str, object]] = []
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
        print(f"[{model_name}] {idx}/{total} BLEU-4={bleu_4:.4f} ROUGE-L={rouge_l:.4f}")
    return evaluated_rows


def main() -> None:
    args = parse_args()
    detail_path = Path(args.output_detail_csv)
    summary_path = Path(args.output_summary_csv)
    ensure_dir(detail_path.parent)
    ensure_dir(summary_path.parent)

    input_rows = load_question_rows(args.question_data_path, args.sample_size, args.seed)
    rows = [
        {
            "id": idx,
            "question_type": item.get("question_type", ""),
            "question": str(item["question"]).strip(),
            "reference_answer": str(item["answer1"]).strip(),
        }
        for idx, item in enumerate(input_rows, start=1)
    ]

    existing_rows = load_existing_detail_rows(detail_path) if args.reuse_existing_detail else []
    existing_by_model = {
        model_name: [row for row in existing_rows if str(row.get("model_name", "")).strip() == model_name]
        for model_name in ["base", "sft", "ppo"]
    }

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if existing_by_model["base"]:
        print(f"复用已有 Base 明细，共 {len(existing_by_model['base'])} 条。")
        base_rows = existing_by_model["base"]
    else:
        print("加载 Base 模型。")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        base_model.eval()
        base_rows = evaluate_model("base", base_model, tokenizer, rows, args.max_new_tokens)
        cleanup_model(base_model)
        persist_progress(detail_path, summary_path, base_rows)

    if existing_by_model["sft"]:
        print(f"复用已有 SFT 明细，共 {len(existing_by_model['sft'])} 条。")
        sft_rows = existing_by_model["sft"]
    else:
        print("加载 SFT 模型。")
        sft_model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        sft_model.eval()
        sft_rows = evaluate_model("sft", sft_model, tokenizer, rows, args.max_new_tokens)
        cleanup_model(sft_model)
        persist_progress(
            detail_path,
            summary_path,
            base_rows + sft_rows if not existing_by_model["base"] else existing_rows + sft_rows,
        )

    if existing_by_model["ppo"]:
        print(f"复用已有 PPO 明细，共 {len(existing_by_model['ppo'])} 条。")
        ppo_rows = existing_by_model["ppo"]
    else:
        print("加载 PPO 模型。")
        ppo_model = load_ppo_model(args)
        ppo_model.eval()
        ppo_rows = evaluate_model("ppo", ppo_model, tokenizer, rows, args.max_new_tokens)
        cleanup_model(ppo_model)
        persist_progress(detail_path, summary_path, base_rows + sft_rows + ppo_rows)

    all_rows = sorted(base_rows + sft_rows + ppo_rows, key=lambda row: (str(row["model_name"]), int(row["id"])))
    save_detail_rows(detail_path, all_rows)
    summary_rows = build_summary_rows(all_rows)
    save_summary_rows(summary_path, summary_rows)

    print(f"三模型自动指标明细已保存：{args.output_detail_csv}")
    print(f"三模型自动指标汇总已保存：{args.output_summary_csv}")


if __name__ == "__main__":
    main()
