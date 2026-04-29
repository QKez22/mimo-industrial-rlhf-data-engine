"""
new_rlhf 第一轮 RM 联合训练脚本（偏好对 + 直接打分）

一、脚本目标
1. 训练奖励模型（Reward Model, RM），同时学习两类监督信号：
   - 成对偏好数据（better_answer vs worse_answer）
   - 直接打分数据（answer + final_score）
2. 训练结束后输出 LoRA 适配器，用于 PPO 阶段给回答打奖励分。

二、默认数据
1. 偏好对训练集：R1_RM偏好训练集_1000对.csv
2. 偏好对验证集：R1_RM偏好验证集_250对.csv
3. 打分训练集：R1_RM直接打分训练集.csv
4. 打分验证集：R1_RM直接打分验证集.csv

三、训练与验证逻辑
1. 训练阶段：两类数据混合训练
   - 偏好对：Bradley-Terry 损失（pairwise）
   - 打分：MSE 损失（pointwise）
2. 验证阶段：
   - 训练中按 eval_loss 做早停选择
   - 训练后额外输出可解释指标：
     pair_val_accuracy / point_val_mae / point_val_rmse 等

四、默认模型路径
1. model_name_or_path 默认指向 sft_v0/merged（已合并的 SFT 模型）
2. output_dir 默认为 model/rm_v1
3. 最终 LoRA 输出为 output_dir/final_lora

五、关键参数（默认）
1. learning_rate=2e-5
2. num_train_epochs=3
3. per_device_train_batch_size=2
4. gradient_accumulation_steps=16
5. warmup_ratio=0.03
6. weight_decay=0.01
7. mse_weight=0.5, bt_weight=0.5
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


# 将 src 目录加入搜索路径，便于直接导入 pipeline.paths
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import (
    LOCAL_MODEL_PATH,
    RM_PAIR_TRAIN_FILE,
    RM_PAIR_VAL_FILE,
    RM_POINT_TRAIN_FILE,
    RM_POINT_VAL_FILE,
    RM_V1_MODEL_DIR,
    SFT_MODEL_DIR,
    ensure_dir,
)


# 默认路径参数
DEFAULT_BASE_MODEL_PATH = str(LOCAL_MODEL_PATH)
DEFAULT_MODEL_PATH = str(SFT_MODEL_DIR / "merged")
DEFAULT_SFT_LORA_PATH = str(SFT_MODEL_DIR / "final_lora")
DEFAULT_OUTPUT_DIR = str(RM_V1_MODEL_DIR)

PAIR_TRAIN = str(RM_PAIR_TRAIN_FILE)
PAIR_VAL = str(RM_PAIR_VAL_FILE)
POINT_TRAIN = str(RM_POINT_TRAIN_FILE)
POINT_VAL = str(RM_POINT_VAL_FILE)


def read_csv_required(path: str, required_columns: List[str]) -> pd.DataFrame:
    """读取 CSV 并校验必需列。

    兼容多种常见编码，降低 Excel 导出导致的乱码风险。
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    encodings_to_try = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error: Optional[Exception] = None
    df: Optional[pd.DataFrame] = None
    used_encoding: Optional[str] = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if df is None:
        raise ValueError(
            f"无法解码 CSV: {path}; 尝试编码={encodings_to_try}; 最后错误={last_error}"
        )
    if used_encoding != "utf-8-sig":
        print(f"[read_csv_required] {Path(path).name} 使用编码: {used_encoding}")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列 {missing}; 实际列={list(df.columns)}")
    return df.dropna(subset=required_columns).reset_index(drop=True)


def load_pair_df(path: str) -> pd.DataFrame:
    """读取偏好对数据并标准化列名。

    原始列:
    - question
    - better_answer
    - worse_answer

    转换后:
    - question
    - chosen
    - rejected
    """
    df = read_csv_required(path, ["question", "better_answer", "worse_answer"])
    df = df.rename(columns={"better_answer": "chosen", "worse_answer": "rejected"})
    return df[["question", "chosen", "rejected"]]


def build_prompt(question: str, answer: str) -> str:
    """统一构造 Qwen Chat 格式输入。"""
    return (
        f"<|im_start|>user\n{str(question).strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{str(answer).strip()}<|im_end|>"
    )


def normalize_scores(scores: pd.Series, scale: float) -> pd.Series:
    """将直接打分标签归一化到 0-1（默认按 10 分制除以 10）。"""
    scores = pd.to_numeric(scores, errors="coerce")
    if scores.isna().any():
        bad_count = int(scores.isna().sum())
        raise ValueError(f"final_score 中有 {bad_count} 条无法转成数值。")
    return scores / scale


class JointRewardDataset(Dataset):
    """将偏好对样本与打分样本合并到同一个 Dataset。"""

    def __init__(
        self,
        pair_df: pd.DataFrame,
        point_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        score_scale: float,
    ) -> None:
        self.items: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 偏好对样本：同一个问题对应 chosen/rejected 两个回答
        for row in pair_df.itertuples(index=False):
            chosen = self._tokenize(build_prompt(row.question, row.chosen))
            rejected = self._tokenize(build_prompt(row.question, row.rejected))
            self.items.append(
                {
                    "sample_type": "pair",
                    "chosen_input_ids": chosen["input_ids"],
                    "chosen_attention_mask": chosen["attention_mask"],
                    "rejected_input_ids": rejected["input_ids"],
                    "rejected_attention_mask": rejected["attention_mask"],
                }
            )

        # 直接打分样本：单条回答 + 归一化标签
        point_scores = normalize_scores(point_df["final_score"], score_scale)
        for (_, row), score in zip(point_df.iterrows(), point_scores):
            encoded = self._tokenize(build_prompt(row["question"], row["answer"]))
            self.items.append(
                {
                    "sample_type": "point",
                    "point_input_ids": encoded["input_ids"],
                    "point_attention_mask": encoded["attention_mask"],
                    "labels": float(score),
                }
            )

    def _tokenize(self, text: str) -> Dict[str, List[int]]:
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


@dataclass
class JointRewardCollator:
    """为混合样本批次做动态 padding。"""

    pad_token_id: int

    def _pad_optional(self, features: List[Dict[str, Any]], ids_key: str, mask_key: str) -> Dict[str, torch.Tensor]:
        selected = [f for f in features if ids_key in f]
        if not selected:
            return {
                ids_key: torch.empty((0, 1), dtype=torch.long),
                mask_key: torch.empty((0, 1), dtype=torch.long),
            }
        input_ids = [torch.tensor(f[ids_key], dtype=torch.long) for f in selected]
        masks = [torch.tensor(f[mask_key], dtype=torch.long) for f in selected]
        return {
            ids_key: pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            mask_key: pad_sequence(masks, batch_first=True, padding_value=0),
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        for ids_key, mask_key in [
            ("chosen_input_ids", "chosen_attention_mask"),
            ("rejected_input_ids", "rejected_attention_mask"),
            ("point_input_ids", "point_attention_mask"),
        ]:
            batch.update(self._pad_optional(features, ids_key, mask_key))

        point_labels = [f["labels"] for f in features if f.get("sample_type") == "point"]
        batch["point_labels"] = torch.tensor(point_labels, dtype=torch.float32)
        return batch


class JointRewardTrainer(Trainer):
    """自定义 Trainer：联合计算 BT 损失与 MSE 损失。"""

    def __init__(self, *args: Any, mse_weight: float = 0.5, bt_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mse_weight = mse_weight
        self.bt_weight = bt_weight

    def _score(self, model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits.float().view(-1)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        losses: List[torch.Tensor] = []
        outputs: Dict[str, torch.Tensor] = {}

        # 偏好对损失：chosen 分数应该高于 rejected
        if inputs["chosen_input_ids"].size(0) > 0:
            chosen_scores = self._score(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
            rejected_scores = self._score(model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"])
            bt_loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
            losses.append(self.bt_weight * bt_loss)
            outputs["bt_loss"] = bt_loss.detach()
            outputs["pair_accuracy"] = (chosen_scores > rejected_scores).float().mean().detach()

        # 打分损失：预测分数接近归一化标签
        if inputs["point_input_ids"].size(0) > 0:
            point_scores = self._score(model, inputs["point_input_ids"], inputs["point_attention_mask"])
            labels = inputs["point_labels"].to(point_scores.device).float()
            mse_loss = F.mse_loss(point_scores, labels)
            losses.append(self.mse_weight * mse_loss)
            outputs["mse_loss"] = mse_loss.detach()

        if not losses:
            raise RuntimeError("当前 batch 没有可用于训练的样本。")

        loss = torch.stack(losses).sum()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


def build_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer, split: str) -> JointRewardDataset:
    """构建训练集或验证集（均包含两类数据）。"""
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
    """训练后分别输出偏好验证和打分验证指标。"""
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


def parse_args() -> argparse.Namespace:
    """解析 RM 训练参数。"""
    parser = argparse.ArgumentParser(description="RM 联合训练：偏好对 + 直接打分")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--sft_lora_path",
        default=None,
        help=f"可选：SFT LoRA 路径。基础模型默认：{DEFAULT_BASE_MODEL_PATH}",
    )
    parser.add_argument(
        "--skip_sft_lora",
        action="store_true",
        help="跳过 SFT LoRA 合并，直接从 model_name_or_path 初始化 RM。",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair_train_path", default=PAIR_TRAIN)
    parser.add_argument("--pair_val_path", default=PAIR_VAL)
    parser.add_argument("--point_train_path", default=POINT_TRAIN)
    parser.add_argument("--point_val_path", default=POINT_VAL)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--score_scale", type=float, default=10.0, help="将 final_score 除以该值做归一化。")
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--bt_weight", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(Path(args.output_dir))

    # 精度参数冲突检查
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 和 --fp16 不能同时开启。")
    if (not args.bf16) and (not args.fp16) and torch.cuda.is_available():
        args.fp16 = True
        print("[auto] 检测到 CUDA，自动启用 fp16。")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_4bit:
        if args.sft_lora_path and not args.skip_sft_lora:
            raise ValueError("使用 --use_4bit 时，建议直接传 merged SFT 模型，不要再叠加 --sft_lora_path。")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(
            f"模型目录不存在: {args.model_name_or_path}\n"
            "请先完成 SFT merged，或传入正确模型路径。"
        )

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "num_labels": 1,
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
    }
    if args.use_4bit and torch.cuda.is_available():
        model_load_kwargs["device_map"] = {"": 0}

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, **model_load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    # 可选：先把 SFT LoRA 合并进基础模型，再作为 RM 初始化起点
    if args.sft_lora_path and not args.skip_sft_lora:
        if not Path(args.sft_lora_path).exists():
            raise FileNotFoundError(f"SFT LoRA adapter 不存在: {args.sft_lora_path}")
        print(f"正在加载并合并 SFT LoRA: {args.sft_lora_path}")
        try:
            model = PeftModel.from_pretrained(model, args.sft_lora_path, is_trainable=False)
            model = model.merge_and_unload()
            model.config.pad_token_id = tokenizer.pad_token_id
        except Exception as exc:
            raise RuntimeError(
                "SFT LoRA 合并失败。建议直接使用 merged SFT 模型作为 --model_name_or_path。"
            ) from exc

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 给 RM 头部叠加 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # 可训练参数转 fp32，降低 AMP 不稳定风险
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    # 训练集和验证集都包含两类数据
    train_dataset = build_dataset(args, tokenizer, "train")
    eval_dataset = build_dataset(args, tokenizer, "eval")
    print(f"训练样本总数: {len(train_dataset)}; 验证样本总数: {len(eval_dataset)}")

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

    # 训练后分别计算偏好验证和打分验证指标
    device = next(trainer.model.parameters()).device
    metrics = evaluate_separately(trainer.model, tokenizer, args, device)
    print("验证指标：")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"RM LoRA 已保存到: {final_dir}")


if __name__ == "__main__":
    main()
