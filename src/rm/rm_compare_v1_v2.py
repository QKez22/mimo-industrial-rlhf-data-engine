"""
new_rlhf 第二轮 RM-V1 / RM-V2 同口径对比评估脚本

一、脚本目标
1. 在完全相同的验证集上对比 RM-V1 和 RM-V2，避免把“R1 验证集结果”和“R1+R2 联合验证集结果”直接混在一起比较。
2. 默认使用第二轮偏好验证集：
   data/03_R2_第二轮RLHF/R2_RM偏好验证集.csv
3. 默认使用第一轮直接打分验证集：
   data/02_R1_第一轮RLHF/R1_RM直接打分验证集.csv
4. 分别输出 RM-V1 与 RM-V2 的：
   - 成对偏好验证 accuracy
   - chosen_score - rejected_score 的 margin_mean / margin_min / margin_max
   - 如果验证集中有 val_source，则额外按 R1 / R2 分组输出 accuracy
   - 直接打分验证 MAE / RMSE，以及还原到 10 分制后的 MAE

二、默认模型路径
1. 底座模型：model/sft_v0/merged
2. RM-V1 LoRA：model/rm_v1/final_lora
3. RM-V2 LoRA：model/rm_v2/final_lora

三、评估逻辑
1. RM 是 sequence classification 模型，输入为统一的 Qwen Chat 格式：
   user: question
   assistant: answer
2. 对偏好对，若 better_answer 得分高于 worse_answer，则记为预测正确。
3. 对直接打分样本，将 final_score 除以 score_scale，默认 score_scale=10.0，
   然后计算模型预测奖励与归一化标签之间的 MAE / RMSE。

四、结果保存
1. 控制台会直接打印对比结果，方便在 PyCharm 运行窗口查看。
2. 同时保存 CSV 到：
   model/eval_results/rm_v1_v2_same_val_compare.csv

五、运行方式
直接在 PyCharm 中运行本脚本即可；默认参数已经写好，一般不需要手动传参。
"""

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline.paths import DATA_DIR, EVAL_RESULTS_DIR, RM_POINT_VAL_FILE, RM_V1_MODEL_DIR, RM_V2_MODEL_DIR, SFT_MODEL_DIR, ensure_dir
from rm_train import build_prompt, load_pair_df, normalize_scores, read_csv_required


R2_DIR = DATA_DIR / "03_R2_第二轮RLHF"
DEFAULT_PAIR_VAL_PATH = R2_DIR / "R2_RM偏好验证集.csv"
DEFAULT_POINT_VAL_PATH = RM_POINT_VAL_FILE

DEFAULT_BASE_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_RM_V1_LORA_PATH = RM_V1_MODEL_DIR / "final_lora"
DEFAULT_RM_V2_LORA_PATH = RM_V2_MODEL_DIR / "final_lora"
DEFAULT_OUTPUT_CSV = EVAL_RESULTS_DIR / "rm_v1_v2_same_val_compare.csv"


def parse_args() -> argparse.Namespace:
    """解析默认参数；默认值已经按第二轮 RM 对比实验配置好。"""
    parser = argparse.ArgumentParser(description="Compare RM-V1 and RM-V2 on the same validation data.")
    parser.add_argument("--base_model_path", default=str(DEFAULT_BASE_MODEL_PATH))
    parser.add_argument("--rm_v1_lora_path", default=str(DEFAULT_RM_V1_LORA_PATH))
    parser.add_argument("--rm_v2_lora_path", default=str(DEFAULT_RM_V2_LORA_PATH))
    parser.add_argument("--pair_val_path", default=str(DEFAULT_PAIR_VAL_PATH))
    parser.add_argument("--point_val_path", default=str(DEFAULT_POINT_VAL_PATH))
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--score_scale", type=float, default=10.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mean(values: List[float]) -> float:
    """计算均值；空列表时返回 0，避免除零。"""
    return sum(values) / max(len(values), 1)


def load_reward_model(
    base_model_path: str,
    rm_lora_path: str,
    device: str,
) -> tuple[AutoTokenizer, PeftModel]:
    """加载一个 RM LoRA 模型；每次只加载一个模型，降低显存占用。"""
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        rm_lora_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=dtype,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, rm_lora_path)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_answer(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    max_length: int,
    device: str,
) -> float:
    """对单个 question-answer 输入打奖励分。"""
    encoded = tokenizer(
        build_prompt(question, answer),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return float(model(**encoded).logits.float().view(-1).item())


def evaluate_pairwise(
    model_name: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    pair_raw_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """评估成对偏好验证集，并在存在 val_source 时额外计算 R1/R2 分组 accuracy。"""
    margins: List[float] = []
    correct_flags: List[int] = []

    for row in pair_df.itertuples(index=False):
        chosen_score = score_answer(model, tokenizer, row.question, row.chosen, args.max_length, args.device)
        rejected_score = score_answer(model, tokenizer, row.question, row.rejected, args.max_length, args.device)
        margin = chosen_score - rejected_score
        margins.append(margin)
        correct_flags.append(int(margin > 0))

    metrics: Dict[str, float] = {
        "pair_samples": float(len(pair_df)),
        "pair_accuracy": mean(correct_flags),
        "pair_margin_mean": mean(margins),
        "pair_margin_min": min(margins) if margins else 0.0,
        "pair_margin_max": max(margins) if margins else 0.0,
    }

    if "val_source" in pair_raw_df.columns:
        tmp = pair_raw_df[["val_source"]].copy().reset_index(drop=True)
        tmp["correct"] = correct_flags
        for source, group in tmp.groupby("val_source", dropna=False):
            key = str(source).strip() or "unknown"
            metrics[f"pair_accuracy_{key}"] = float(group["correct"].mean())
            metrics[f"pair_samples_{key}"] = float(len(group))

    print(f"\n[{model_name}] 成对偏好验证")
    print(f"  samples: {int(metrics['pair_samples'])}")
    print(f"  accuracy: {metrics['pair_accuracy']:.6f}")
    print(f"  margin_mean: {metrics['pair_margin_mean']:.6f}")
    print(f"  margin_min: {metrics['pair_margin_min']:.6f}")
    print(f"  margin_max: {metrics['pair_margin_max']:.6f}")
    for key in sorted(metrics):
        if key.startswith("pair_accuracy_"):
            source = key.replace("pair_accuracy_", "")
            sample_key = f"pair_samples_{source}"
            print(f"  accuracy_{source}: {metrics[key]:.6f} / samples={int(metrics[sample_key])}")

    return metrics


def evaluate_pointwise(
    model_name: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    point_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """评估直接打分验证集，重点观察奖励分数是否仍然具备基本校准能力。"""
    labels = normalize_scores(point_df["final_score"], args.score_scale).tolist()
    preds = [
        score_answer(model, tokenizer, row.question, row.answer, args.max_length, args.device)
        for row in point_df.itertuples(index=False)
    ]
    abs_errors = [abs(pred - label) for pred, label in zip(preds, labels)]
    sq_errors = [(pred - label) ** 2 for pred, label in zip(preds, labels)]

    metrics = {
        "point_samples": float(len(point_df)),
        "point_mae_normalized": mean(abs_errors),
        "point_rmse_normalized": math.sqrt(mean(sq_errors)),
        "point_mae_score_scale_10": mean(abs_errors) * args.score_scale,
        "point_pred_mean": mean(preds),
        "point_label_mean": mean(labels),
    }

    print(f"\n[{model_name}] 直接打分验证")
    print(f"  samples: {int(metrics['point_samples'])}")
    print(f"  mae_normalized: {metrics['point_mae_normalized']:.6f}")
    print(f"  rmse_normalized: {metrics['point_rmse_normalized']:.6f}")
    print(f"  mae_score_scale_{args.score_scale:g}: {metrics['point_mae_score_scale_10']:.6f}")
    print(f"  pred_mean: {metrics['point_pred_mean']:.6f}")
    print(f"  label_mean: {metrics['point_label_mean']:.6f}")

    return metrics


def evaluate_one_model(
    model_name: str,
    rm_lora_path: str,
    pair_raw_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    point_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, float | str]:
    """加载并评估一个 RM；评估完成后由 main 负责释放显存。"""
    print(f"\n加载 {model_name}: {rm_lora_path}")
    tokenizer, model = load_reward_model(args.base_model_path, rm_lora_path, args.device)
    pair_metrics = evaluate_pairwise(model_name, model, tokenizer, pair_raw_df, pair_df, args)
    point_metrics = evaluate_pointwise(model_name, model, tokenizer, point_df, args)

    result: Dict[str, float | str] = {"model": model_name, "rm_lora_path": rm_lora_path}
    result.update(pair_metrics)
    result.update(point_metrics)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()

    for path in [args.base_model_path, args.rm_v1_lora_path, args.rm_v2_lora_path, args.pair_val_path, args.point_val_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"路径不存在：{path}")

    pair_raw_df = read_csv_required(args.pair_val_path, ["question", "better_answer", "worse_answer"])
    pair_df = load_pair_df(args.pair_val_path)
    point_df = read_csv_required(args.point_val_path, ["question", "answer", "final_score"])

    print("同口径 RM 对比评估开始")
    print(f"  base_model: {args.base_model_path}")
    print(f"  pair_val: {args.pair_val_path}")
    print(f"  point_val: {args.point_val_path}")
    print(f"  device: {args.device}")

    results = [
        evaluate_one_model("RM-V1", args.rm_v1_lora_path, pair_raw_df, pair_df, point_df, args),
        evaluate_one_model("RM-V2", args.rm_v2_lora_path, pair_raw_df, pair_df, point_df, args),
    ]

    result_df = pd.DataFrame(results)
    ensure_dir(Path(args.output_csv).parent)
    result_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n========== 同口径对比汇总 ==========")
    keep_cols = [
        "model",
        "pair_accuracy",
        "pair_accuracy_R1",
        "pair_accuracy_R2",
        "pair_margin_mean",
        "point_mae_normalized",
        "point_rmse_normalized",
        "point_mae_score_scale_10",
    ]
    keep_cols = [col for col in keep_cols if col in result_df.columns]
    print(result_df[keep_cols].to_string(index=False))
    print(f"\n结果已保存到: {args.output_csv}")


if __name__ == "__main__":
    main()
