"""
new_rlhf SFT / PPO-V1 / PPO-V2 局部验证集自动指标对比脚本

一、脚本目标
1. 固定使用 data/01_测试集与验证集/局部验证集_150条.csv 作为同一套评估集。
2. 依次评估三个模型：
   - SFT-V0:  model/sft_v0/merged
   - PPO-V1: model/ppo_v1/merged，若不存在则回退到 SFT-V0 + model/ppo_v1/final_lora
   - PPO-V2: model/ppo_v2/merged，若不存在则回退到 PPO-V1 merged + model/ppo_v2/final_lora
3. 每个模型对 150 道题各生成 1 个回答，并与 answer1 参考答案逐题计算 BLEU-4 与 ROUGE-L。
4. 用相同问题、相同参考答案、相同解码参数，判断 SFT -> PPO-V1 -> PPO-V2 是否有迭代提升。

二、默认评估口径
1. 参考答案列：answer1
2. 指标：
   - BLEU-4：中文字符级 n-gram 精确匹配，带平滑
   - ROUGE-L：中文字符级最长公共子序列 F1
3. 生成参数：
   - do_sample=False
   - max_new_tokens=180
   - repetition_penalty=1.1
4. 这个指标主要用于同项目内部模型纵向比较，不建议和英文单词级 BLEU/ROUGE 横向比较。

三、输出文件
1. 明细表：
   logs/eval/sft_ppo_v1_ppo_v2_local150_metric_details.csv
2. 汇总表：
   logs/eval/sft_ppo_v1_ppo_v2_local150_metric_summary.csv

四、断点续跑
1. 默认复用已经完成的模型结果。
2. 每完成一个模型，就立刻保存明细和汇总，避免中途崩溃导致前面结果丢失。

五、运行方式
直接在 PyCharm 中运行本脚本即可。
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

from pipeline.paths import EVAL_LOG_DIR, LOCAL_VAL_FILE, PPO_V1_MODEL_DIR, PPO_V2_MODEL_DIR, SFT_MODEL_DIR, ensure_dir


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ORDER = ["sft", "ppo_v1", "ppo_v2"]


def parse_args() -> argparse.Namespace:
    """解析评估参数；默认值已经按局部验证集 150 条配置好。"""
    parser = argparse.ArgumentParser(description="SFT / PPO-V1 / PPO-V2 自动指标对比（BLEU-4 / ROUGE-L）")
    parser.add_argument("--question_data_path", default=str(LOCAL_VAL_FILE))
    parser.add_argument("--sft_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_v1_merged_model_path", default=str(PPO_V1_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_v1_lora_path", default=str(PPO_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--ppo_v2_merged_model_path", default=str(PPO_V2_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_v2_lora_path", default=str(PPO_V2_MODEL_DIR / "final_lora"))
    parser.add_argument(
        "--output_detail_csv",
        default=str(EVAL_LOG_DIR / "sft_ppo_v1_ppo_v2_local150_metric_details.csv"),
    )
    parser.add_argument(
        "--output_summary_csv",
        default=str(EVAL_LOG_DIR / "sft_ppo_v1_ppo_v2_local150_metric_summary.csv"),
    )
    parser.add_argument("--sample_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--reuse_existing_detail", action="store_true", default=True)
    parser.add_argument("--offload_dir", default=str(EVAL_LOG_DIR / "sft_ppo_v1_v2_offload"))
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """去除所有空白字符，减少换行和空格对中文字符级指标的干扰。"""
    return "".join(str(text).strip().split())


def tokenize_zh_chars(text: str) -> List[str]:
    """按中文字符级切分；英文、数字和标点也按字符处理。"""
    return list(normalize_text(text))


def build_prompt(question: str) -> str:
    """统一使用 Qwen Chat 格式构造输入。"""
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"


def first_model_device(model) -> torch.device:
    """兼容 device_map='auto'；优先取第一个非 meta 参数所在设备。"""
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device(DEVICE)


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    """生成单条回答，并只截取 assistant 部分。"""
    inputs = tokenizer(build_prompt(question), return_tensors="pt")
    inputs = {key: value.to(first_model_device(model)) for key, value in inputs.items()}
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


def read_csv_flexible(path: str) -> pd.DataFrame:
    """读取 CSV，兼容 utf-8-sig / utf-8 / gb18030 / gbk。"""
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV：{path}; 最后错误：{last_error}")


def load_question_rows(path: str, sample_size: int, seed: int) -> List[Dict[str, str]]:
    """读取局部验证集，并整理成统一字段。"""
    df = read_csv_flexible(path)
    required_columns = ["question", "answer1"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"评估集缺少列：{missing}；文件：{path}")

    df = df.dropna(subset=required_columns).reset_index(drop=True)
    df["question"] = df["question"].astype(str).str.strip()
    df["answer1"] = df["answer1"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer1"] != "")].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"评估集为空：{path}")

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


def get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """提取 n-gram。"""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu_4(reference_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    """计算单条 BLEU-4，加入简单平滑，避免短回答直接归零。"""
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
    """计算 ROUGE-L F1。"""
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def mean(values: List[float]) -> float:
    """安全均值。"""
    return sum(values) / max(len(values), 1)


def cleanup_model(model) -> None:
    """释放模型，降低连续评估三个模型时的显存压力。"""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_sft_model(args: argparse.Namespace):
    """加载 SFT-V0 merged 模型。"""
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    return model


def load_ppo_v1_model(args: argparse.Namespace):
    """优先加载 PPO-V1 merged；不存在时回退到 SFT-V0 + PPO-V1 LoRA。"""
    if Path(args.ppo_v1_merged_model_path).exists():
        print(f"加载 PPO-V1 merged：{args.ppo_v1_merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.ppo_v1_merged_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        model.eval()
        return model

    print("未找到 PPO-V1 merged，回退到 SFT-V0 + PPO-V1 LoRA。")
    offload_dir = ensure_dir(Path(args.offload_dir) / "ppo_v1")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        offload_folder=str(offload_dir),
        offload_state_dict=True,
    )
    model = PeftModel.from_pretrained(base_model, args.ppo_v1_lora_path, offload_folder=str(offload_dir))
    model.eval()
    return model


def load_ppo_v2_model(args: argparse.Namespace):
    """优先加载 PPO-V2 merged；不存在时回退到 PPO-V1 merged + PPO-V2 LoRA。"""
    if Path(args.ppo_v2_merged_model_path).exists():
        print(f"加载 PPO-V2 merged：{args.ppo_v2_merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.ppo_v2_merged_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        model.eval()
        return model

    if not Path(args.ppo_v1_merged_model_path).exists():
        raise FileNotFoundError(
            "未找到 PPO-V2 merged，且回退所需 PPO-V1 merged 也不存在。请先运行 ppo_merge_v2.py，"
            "或确认 model/ppo_v1/merged 存在。"
        )

    print("未找到 PPO-V2 merged，回退到 PPO-V1 merged + PPO-V2 LoRA。")
    offload_dir = ensure_dir(Path(args.offload_dir) / "ppo_v2")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.ppo_v1_merged_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        offload_folder=str(offload_dir),
        offload_state_dict=True,
    )
    model = PeftModel.from_pretrained(base_model, args.ppo_v2_lora_path, offload_folder=str(offload_dir))
    model.eval()
    return model


def load_existing_detail_rows(path: Path) -> List[Dict[str, object]]:
    """读取已存在的明细结果，支持断点续跑。"""
    if not path.exists():
        return []
    return pd.read_csv(path, encoding="utf-8-sig").to_dict("records")


def reusable_rows_for_model(existing_rows: List[Dict[str, object]], model_name: str, expected_ids: set[int]) -> List[Dict[str, object]]:
    """只有模型结果覆盖全部样本时才复用，避免半截结果误判为完成。"""
    rows = [row for row in existing_rows if str(row.get("model_name", "")).strip() == model_name]
    row_ids = {int(row["id"]) for row in rows if str(row.get("id", "")).strip() != ""}
    return rows if row_ids == expected_ids else []


def save_detail_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    """保存逐题明细。"""
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
    """生成汇总指标，并额外计算相邻阶段提升量。"""
    summary_rows: List[Dict[str, object]] = []
    metric_by_model: Dict[str, Dict[str, float]] = {}

    for model_name in MODEL_ORDER:
        model_rows = [row for row in all_rows if row["model_name"] == model_name]
        if not model_rows:
            continue
        metrics = {
            "sample_count": float(len(model_rows)),
            "avg_bleu_4": mean([float(row["bleu_4"]) for row in model_rows]),
            "avg_rouge_l": mean([float(row["rouge_l"]) for row in model_rows]),
            "avg_answer_length": mean([float(row["answer_length"]) for row in model_rows]),
        }
        metric_by_model[model_name] = metrics
        for metric, value in metrics.items():
            summary_rows.append({"model_name": model_name, "metric": metric, "value": value})

    for current_model, previous_model in [("ppo_v1", "sft"), ("ppo_v2", "ppo_v1")]:
        if current_model not in metric_by_model or previous_model not in metric_by_model:
            continue
        for metric in ["avg_bleu_4", "avg_rouge_l"]:
            delta = metric_by_model[current_model][metric] - metric_by_model[previous_model][metric]
            summary_rows.append(
                {
                    "model_name": current_model,
                    "metric": f"delta_{metric}_vs_{previous_model}",
                    "value": delta,
                }
            )
            summary_rows.append(
                {
                    "model_name": current_model,
                    "metric": f"improved_{metric}_vs_{previous_model}",
                    "value": int(delta > 0),
                }
            )

    return summary_rows


def save_summary_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    """保存汇总表。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def persist_progress(detail_path: Path, summary_path: Path, rows: List[Dict[str, object]]) -> None:
    """每完成一个模型立即保存结果。"""
    sorted_rows = sorted(rows, key=lambda row: (str(row["model_name"]), int(row["id"])))
    save_detail_rows(detail_path, sorted_rows)
    save_summary_rows(summary_path, build_summary_rows(sorted_rows))


def evaluate_model(model_name: str, model, tokenizer, rows: List[Dict[str, str]], max_new_tokens: int) -> List[Dict[str, object]]:
    """评估一个模型的全部样本。"""
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


def print_summary(summary_rows: List[Dict[str, object]]) -> None:
    """在控制台打印关键汇总，便于 PyCharm 直接查看。"""
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return
    print("\n========== SFT / PPO-V1 / PPO-V2 局部验证集指标汇总 ==========")
    for model_name in MODEL_ORDER:
        model_df = summary_df[summary_df["model_name"] == model_name]
        if model_df.empty:
            continue
        values = {row["metric"]: float(row["value"]) for _, row in model_df.iterrows()}
        print(
            f"{model_name}: "
            f"BLEU-4={values.get('avg_bleu_4', 0.0):.6f}, "
            f"ROUGE-L={values.get('avg_rouge_l', 0.0):.6f}, "
            f"avg_len={values.get('avg_answer_length', 0.0):.2f}"
        )
    for model_name, previous_model in [("ppo_v1", "sft"), ("ppo_v2", "ppo_v1")]:
        model_df = summary_df[summary_df["model_name"] == model_name]
        values = {row["metric"]: float(row["value"]) for _, row in model_df.iterrows()}
        bleu_delta = values.get(f"delta_avg_bleu_4_vs_{previous_model}")
        rouge_delta = values.get(f"delta_avg_rouge_l_vs_{previous_model}")
        if bleu_delta is not None and rouge_delta is not None:
            print(f"{model_name} vs {previous_model}: BLEU-4 delta={bleu_delta:.6f}, ROUGE-L delta={rouge_delta:.6f}")


def main() -> None:
    args = parse_args()
    detail_path = Path(args.output_detail_csv)
    summary_path = Path(args.output_summary_csv)
    ensure_dir(detail_path.parent)
    ensure_dir(summary_path.parent)

    for path, label in [
        (args.question_data_path, "局部验证集"),
        (args.sft_model_path, "SFT-V0 merged"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} 不存在：{path}")

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
    expected_ids = {int(row["id"]) for row in rows}

    existing_rows = load_existing_detail_rows(detail_path) if args.reuse_existing_detail else []
    completed_rows: List[Dict[str, object]] = []

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_loaders = {
        "sft": load_sft_model,
        "ppo_v1": load_ppo_v1_model,
        "ppo_v2": load_ppo_v2_model,
    }

    print(f"评估集：{args.question_data_path}")
    print(f"样本数：{len(rows)}")
    print(f"输出明细：{detail_path}")
    print(f"输出汇总：{summary_path}")

    for model_name in MODEL_ORDER:
        reusable_rows = reusable_rows_for_model(existing_rows, model_name, expected_ids)
        if reusable_rows:
            print(f"复用已有 {model_name} 明细，共 {len(reusable_rows)} 条。")
            completed_rows.extend(reusable_rows)
            persist_progress(detail_path, summary_path, completed_rows)
            continue

        print(f"\n加载并评估 {model_name}。")
        model = model_loaders[model_name](args)
        model_rows = evaluate_model(model_name, model, tokenizer, rows, args.max_new_tokens)
        cleanup_model(model)
        completed_rows.extend(model_rows)
        persist_progress(detail_path, summary_path, completed_rows)

    all_rows = sorted(completed_rows, key=lambda row: (str(row["model_name"]), int(row["id"])))
    save_detail_rows(detail_path, all_rows)
    summary_rows = build_summary_rows(all_rows)
    save_summary_rows(summary_path, summary_rows)
    print_summary(summary_rows)

    print(f"\n三模型自动指标明细已保存：{detail_path}")
    print(f"三模型自动指标汇总已保存：{summary_path}")


if __name__ == "__main__":
    main()
