"""
new_rlhf 第二轮 RM-V2 增量训练脚本

一、脚本目标
1. 在 RM-V1 的基础上继续微调，得到 RM-V2。
2. 使用第二轮成对偏好数据作为主要训练信号。
3. 同时保留直接打分数据作为辅助监督，防止奖励分数尺度漂移。

二、默认输入数据
1. 成对偏好训练集：
   data/03_R2_第二轮RLHF/R2_RM偏好训练集_含回放_混合.csv
2. 成对偏好验证集：
   data/03_R2_第二轮RLHF/R2_RM偏好验证集.csv
3. 直接打分训练集：
   data/03_R2_第二轮RLHF/R2_RM直打训练集_含回放_混合.csv
4. 直接打分验证集：
   data/02_R1_第一轮RLHF/R1_RM直接打分验证集.csv

三、当前数据使用方式
1. 第二轮直接打分训练集已经由外部整理好，脚本不再额外拆分回放数据。
2. 直接打分验证集固定使用第一轮 RLHF 的 R1_RM直接打分验证集.csv，便于观察 RM-V2 是否保持基础分数校准能力。

四、模型初始化
1. 底模：model/sft_v0/merged
2. 继承权重：model/rm_v1/final_lora
3. 输出：model/rm_v2/final_lora

五、损失权重
1. 成对偏好 Bradley-Terry 损失权重 bt_weight = 0.8
2. 直接打分 MSE 损失权重 mse_weight = 0.2
3. 这表示 RM-V2 更重视偏好排序能力。

六、使用方式
直接在 PyCharm 中运行本脚本即可，不需要手动传参数。
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline.paths import DATA_DIR, RM_POINT_VAL_FILE, RM_V1_MODEL_DIR, RM_V2_MODEL_DIR, SFT_MODEL_DIR, ensure_dir
from rm_train import (
    JointRewardCollator,
    JointRewardDataset,
    JointRewardTrainer,
    build_prompt,
    load_pair_df,
    normalize_scores,
    read_csv_required,
)


R2_DIR = DATA_DIR / "03_R2_第二轮RLHF"

PAIR_TRAIN_PATH = R2_DIR / "R2_RM偏好训练集_含回放_混合.csv"
PAIR_VAL_PATH = R2_DIR / "R2_RM偏好验证集.csv"
POINT_TRAIN_PATH = R2_DIR / "R2_RM直打训练集_含回放_混合.csv"
POINT_VAL_PATH = RM_POINT_VAL_FILE

DEFAULT_BASE_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_RM_V1_LORA_PATH = RM_V1_MODEL_DIR / "final_lora"
DEFAULT_OUTPUT_DIR = RM_V2_MODEL_DIR


def parse_args() -> argparse.Namespace:
    """解析参数；默认值已经配置为第二轮 RM-V2 直接运行。"""
    parser = argparse.ArgumentParser(description="Incrementally train RM-V2 from RM-V1.")
    parser.add_argument("--base_model_path", default=str(DEFAULT_BASE_MODEL_PATH))
    parser.add_argument("--rm_v1_lora_path", default=str(DEFAULT_RM_V1_LORA_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pair_train_path", default=str(PAIR_TRAIN_PATH))
    parser.add_argument("--pair_val_path", default=str(PAIR_VAL_PATH))
    parser.add_argument("--point_train_path", default=str(POINT_TRAIN_PATH))
    parser.add_argument("--point_val_path", default=str(POINT_VAL_PATH))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--score_scale", type=float, default=10.0)
    parser.add_argument("--bt_weight", type=float, default=0.8)
    parser.add_argument("--mse_weight", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_pointwise_data(args: argparse.Namespace) -> None:
    """检查 RM-V2 使用的直接打分训练/验证集，不再额外生成新文件。"""
    point_train = read_csv_required(args.point_train_path, ["question", "answer", "final_score"])
    point_val = read_csv_required(args.point_val_path, ["question", "answer", "final_score"])

    print("RM-V2 直接打分数据检查完成：")
    print(f"  直接打分训练集: {len(point_train)} -> {args.point_train_path}")
    print(f"  直接打分验证集: {len(point_val)} -> {args.point_val_path}")


def build_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer, split: str) -> JointRewardDataset:
    """构建联合 RM 数据集。"""
    if split == "train":
        pair_path, point_path = args.pair_train_path, args.point_train_path
    else:
        pair_path, point_path = args.pair_val_path, args.point_val_path

    pair_df = load_pair_df(pair_path)
    point_df = read_csv_required(point_path, ["question", "answer", "final_score"])
    return JointRewardDataset(
        pair_df=pair_df,
        point_df=point_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        score_scale=args.score_scale,
    )


@torch.no_grad()
def evaluate_separately(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """分别计算偏好验证与直接打分验证指标。"""
    model.eval()
    pair_df = load_pair_df(args.pair_val_path)
    point_df = read_csv_required(args.point_val_path, ["question", "answer", "final_score"])

    def score_text(text: str) -> float:
        encoded = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        return float(model(**encoded).logits.float().view(-1).item())

    pair_correct = 0
    for row in pair_df.itertuples(index=False):
        chosen_score = score_text(build_prompt(row.question, row.chosen))
        rejected_score = score_text(build_prompt(row.question, row.rejected))
        pair_correct += int(chosen_score > rejected_score)

    labels = normalize_scores(point_df["final_score"], args.score_scale).tolist()
    preds = [score_text(build_prompt(row.question, row.answer)) for row in point_df.itertuples(index=False)]
    abs_errors = [abs(p - y) for p, y in zip(preds, labels)]
    sq_errors = [(p - y) ** 2 for p, y in zip(preds, labels)]

    return {
        "pair_val_accuracy": pair_correct / max(len(pair_df), 1),
        "point_val_mae": sum(abs_errors) / max(len(abs_errors), 1),
        "point_val_mse": sum(sq_errors) / max(len(sq_errors), 1),
        "point_val_rmse": math.sqrt(sum(sq_errors) / max(len(sq_errors), 1)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(Path(args.output_dir))
    prepare_pointwise_data(args)

    if args.bf16 and args.fp16:
        raise ValueError("--bf16 和 --fp16 不能同时开启。")
    if (not args.bf16) and (not args.fp16) and torch.cuda.is_available():
        args.fp16 = True
        print("[auto] 检测到 CUDA，自动启用 fp16。")

    tokenizer = AutoTokenizer.from_pretrained(
        args.rm_v1_lora_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not Path(args.base_model_path).exists():
        raise FileNotFoundError(f"底模不存在：{args.base_model_path}")
    if not Path(args.rm_v1_lora_path).exists():
        raise FileNotFoundError(f"RM-V1 LoRA 不存在：{args.rm_v1_lora_path}")

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "num_labels": 1,
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
    }
    if args.use_4bit and torch.cuda.is_available():
        model_load_kwargs["device_map"] = {"": 0}

    print(f"加载 RM 底模：{args.base_model_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model_path, **model_load_kwargs)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"继承 RM-V1 LoRA 并继续训练：{args.rm_v1_lora_path}")
    model = PeftModel.from_pretrained(base_model, args.rm_v1_lora_path, is_trainable=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    train_dataset = build_dataset(args, tokenizer, "train")
    eval_dataset = build_dataset(args, tokenizer, "eval")
    print(f"训练样本总数: {len(train_dataset)}; 验证样本总数: {len(eval_dataset)}")
    print(f"联合损失权重: bt_weight={args.bt_weight}; mse_weight={args.mse_weight}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=1.0,
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = JointRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=JointRewardCollator(tokenizer.pad_token_id),
        mse_weight=args.mse_weight,
        bt_weight=args.bt_weight,
    )

    trainer.train()

    final_dir = Path(args.output_dir) / "final_lora"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    device = next(trainer.model.parameters()).device
    metrics = evaluate_separately(trainer.model, tokenizer, args, device)
    print("验证指标：")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"RM-V2 LoRA 已保存到: {final_dir}")


if __name__ == "__main__":
    main()
