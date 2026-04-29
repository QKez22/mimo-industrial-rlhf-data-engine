"""
new_rlhf RM 验证脚本

一、脚本用途
1. 对已经训练完成的 RM LoRA 进行离线评估。
2. 分别在两类验证集上输出结果：
   - 偏好对验证集（pairwise）
   - 直接打分验证集（pointwise）

二、默认输入
1. base_model_path: 默认使用 sft_v0/merged
2. rm_lora_path:    默认使用 rm_v1/final_lora
3. pair_val_path:   R1_RM偏好验证集_250对.csv
4. point_val_path:  R1_RM直接打分验证集.csv

三、输出指标
1. 偏好对：
   - accuracy
   - margin_mean / margin_min / margin_max
2. 直接打分：
   - mae_normalized
   - rmse_normalized
   - mae_score_scale_x（恢复到原始分值尺度后的 MAE）
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rm_train import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    PAIR_VAL,
    POINT_VAL,
    build_prompt,
    load_pair_df,
    normalize_scores,
    read_csv_required,
)


def parse_args() -> argparse.Namespace:
    """解析评估参数。"""
    parser = argparse.ArgumentParser(description="评估 RM：偏好对 + 直接打分 双验证")
    parser.add_argument("--base_model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rm_lora_path", default=str(Path(DEFAULT_OUTPUT_DIR) / "final_lora"))
    parser.add_argument("--pair_val_path", default=PAIR_VAL)
    parser.add_argument("--point_val_path", default=POINT_VAL)
    parser.add_argument("--score_scale", type=float, default=10.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    # 先加载 RM tokenizer（通常跟 rm_lora_path 同目录）
    tokenizer = AutoTokenizer.from_pretrained(args.rm_lora_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 用 base_model + lora 组合恢复可打分的 RM 模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=dtype,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, args.rm_lora_path)
    model.to(args.device)
    model.eval()

    @torch.no_grad()
    def score(question: str, answer: str) -> float:
        encoded = tokenizer(
            build_prompt(question, answer),
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(args.device)
        return float(model(**encoded).logits.float().view(-1).item())

    # 偏好对验证：看 chosen 是否 consistently 高于 rejected
    pair_df = load_pair_df(args.pair_val_path)
    pair_margins = []
    pair_correct = 0
    for row in pair_df.itertuples(index=False):
        chosen_score = score(row.question, row.chosen)
        rejected_score = score(row.question, row.rejected)
        margin = chosen_score - rejected_score
        pair_margins.append(margin)
        pair_correct += int(margin > 0)

    # 直接打分验证：看预测分数与标签分数的偏差
    point_df = read_csv_required(args.point_val_path, ["question", "answer", "final_score"])
    labels = normalize_scores(point_df["final_score"], args.score_scale).tolist()
    preds = [score(row.question, row.answer) for row in point_df.itertuples(index=False)]
    abs_errors = [abs(pred - label) for pred, label in zip(preds, labels)]
    sq_errors = [(pred - label) ** 2 for pred, label in zip(preds, labels)]

    print("Pairwise validation")
    print(f"  samples: {len(pair_df)}")
    print(f"  accuracy: {pair_correct / max(len(pair_df), 1):.6f}")
    print(f"  margin_mean: {mean(pair_margins):.6f}")
    print(f"  margin_min: {min(pair_margins):.6f}")
    print(f"  margin_max: {max(pair_margins):.6f}")

    print("\nPointwise validation")
    print(f"  samples: {len(point_df)}")
    print(f"  mae_normalized: {mean(abs_errors):.6f}")
    print(f"  rmse_normalized: {math.sqrt(mean(sq_errors)):.6f}")
    print(f"  mae_score_scale_{args.score_scale:g}: {mean(abs_errors) * args.score_scale:.6f}")
    print(f"  pred_mean: {mean(preds):.6f}")
    print(f"  label_mean: {mean(labels):.6f}")


if __name__ == "__main__":
    main()
