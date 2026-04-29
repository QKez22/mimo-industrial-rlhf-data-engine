"""
new_rlhf 第一轮 SFT 训练脚本

一、脚本目标
1. 本脚本用于 new_rlhf 实验的第一轮监督微调（SFT）。
2. 训练起点始终是基础模型 Qwen1.5-1.8B-Chat。
3. 不是在旧的 SFT / RM / PPO 模型基础上继续训练。
4. 训练结束后的主要结果是“已经合并 LoRA 权重的完整 SFT 模型”。

二、默认数据
1. 训练集：data 中的 R1_SFT训练集_700条.csv
2. 验证集：data 中的 R1_SFT验证集_100条.csv
3. 由于当前项目文件名包含中文，脚本内部会在第一轮数据目录下自动定位这两个文件，
   但逻辑上仍然严格对应这两份数据。

三、默认输出
1. 主输出目录：new_rlhf/model/sft_v0/merged
   这里保存的是“可直接给 RM / PPO 使用”的完整 merged 模型。
2. 备份输出目录：new_rlhf/model/sft_v0/final_lora
   这里保存 LoRA 适配器，仅作为备份或消融实验使用。

四、训练参数摘要
1. 最大长度：512
2. 学习率：2e-5
3. 训练轮数：3 epoch
4. per_device_train_batch_size：1
5. per_device_eval_batch_size：1
6. gradient_accumulation_steps：4
7. warmup_ratio：0.03
8. 每个 epoch 评估一次，并按验证集 loss 选最佳模型
9. 有 CUDA 时使用 fp16，无 CUDA 时使用 fp32

五、训练细节
1. 使用 Qwen Chat 格式构造输入：
   <|im_start|>user ... <|im_end|>
   <|im_start|>assistant ... <|im_end|>
2. 仅 assistant 回复部分参与 loss，user 提问部分标签全部置为 -100。
3. 训练过程采用 LoRA 进行参数高效微调。
4. 训练完成后执行 merge_and_unload，将 LoRA 权重合并回基础模型。

六、结论
1. 你后续默认应该优先使用 merged 目录下的完整 SFT 模型。
2. final_lora 只是训练过程备份，不是主结果。
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


# 项目根目录：.../RLHF-Training/new_rlhf
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

# 基础模型路径：这里明确指定为“原始基础模型”
# 训练时就是从这个模型开始挂载 LoRA，再进行 SFT。
BASE_MODEL_PATH = Path(r"C:\Users\kexiu\.cache\modelscope\hub\models\Qwen\Qwen1.5-1.8B-Chat")

# SFT 总输出目录。
DEFAULT_SFT_ROOT_DIR = MODEL_DIR / "sft_v0"

# 主结果：合并后的完整模型。
DEFAULT_MERGED_OUTPUT_DIR = DEFAULT_SFT_ROOT_DIR / "merged"

# 备份结果：LoRA 适配器。
DEFAULT_LORA_OUTPUT_DIR = DEFAULT_SFT_ROOT_DIR / "final_lora"


# 默认超参数。
MAX_LENGTH = 512
SEED = 42
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.03


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该目录对象。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_r1_sft_file(kind: str) -> Path:
    """自动定位第一轮 SFT 数据文件。

    这里不把中文文件名完全写死，而是根据文件特征搜索：
    1. 训练集：文件名同时包含 R1_SFT 和 700
    2. 验证集：文件名同时包含 R1_SFT 和 100

    这样做更稳，尤其是在终端编码或不同编辑器环境下，能减少中文路径识别问题。
    """
    round1_dirs = sorted(DATA_DIR.glob("02_R1_*"))
    if not round1_dirs:
        raise FileNotFoundError(f"未找到第一轮 RLHF 数据目录：{DATA_DIR}")

    round1_dir = round1_dirs[0]
    pattern = "R1_SFT*700*.csv" if kind == "train" else "R1_SFT*100*.csv"
    matches = sorted(round1_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"未找到 R1 SFT {kind} 文件，搜索目录：{round1_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"R1 SFT {kind} 文件匹配到多个结果：{matches}")
    return matches[0]


DEFAULT_TRAIN_DATA_PATH = resolve_r1_sft_file("train")
DEFAULT_VAL_DATA_PATH = resolve_r1_sft_file("val")


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    默认情况下：
    1. 基础模型就是 BASE_MODEL_PATH
    2. 训练集就是 R1_SFT训练集_700条.csv
    3. 验证集就是 R1_SFT验证集_100条.csv
    4. 主输出就是 model/sft_v0/merged
    """
    parser = argparse.ArgumentParser(description="new_rlhf 第一轮 SFT 训练脚本（主结果为 merged 完整模型）")
    parser.add_argument("--base_model_path", default=str(BASE_MODEL_PATH))
    parser.add_argument("--train_data_path", default=str(DEFAULT_TRAIN_DATA_PATH))
    parser.add_argument("--val_data_path", default=str(DEFAULT_VAL_DATA_PATH))
    parser.add_argument("--sft_root_dir", default=str(DEFAULT_SFT_ROOT_DIR))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip_save_lora",
        action="store_true",
        help="如果只想保留 merged 完整模型，可以加这个参数跳过 final_lora 备份保存。",
    )
    return parser.parse_args()


def read_sft_csv(path: str) -> Dataset:
    """读取 SFT 数据，并检查是否满足训练要求。

    要求：
    1. CSV 至少包含 question / answer 两列
    2. 两列内容不能为空
    3. 清洗后数据不能为空
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    required_cols = {"question", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"SFT CSV 缺少必要列：{sorted(missing)}；文件：{path}")

    df = df[["question", "answer"]].dropna()
    df = df[
        (df["question"].astype(str).str.strip() != "")
        & (df["answer"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"SFT 数据清洗后为空：{path}")

    return Dataset.from_pandas(df)


def preprocess_function(examples, tokenizer: AutoTokenizer, max_length: int):
    """将 question / answer 转成 Qwen 对话格式，并构造 response-only 标签。

    关键逻辑：
    1. user 段只作为输入上下文，不参与 loss
    2. assistant 段才参与 loss
    3. 如果样本在截断后 assistant 完全消失，则跳过该样本
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for question, answer in zip(examples["question"], examples["answer"]):
        user_text = f"<|im_start|>user\n{str(question).strip()}<|im_end|>\n<|im_start|>assistant\n"
        assistant_text = f"{str(answer).strip()}<|im_end|>"
        full_text = user_text + assistant_text

        # 完整样本编码，用于模型真正接收的输入。
        full_enc = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")

        # 单独编码 user，用来确定 assistant 从哪个 token 开始。
        user_enc = tokenizer(user_text, truncation=True, max_length=max_length, padding=False)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = [-100] * len(input_ids)
        user_len = len(user_enc["input_ids"])

        # 只让 assistant 部分参与 loss。
        for i in range(user_len, len(input_ids)):
            if attention_mask[i] == 1:
                labels[i] = input_ids[i]

        # 如果 assistant 完全被截断，这条样本就不参与训练。
        if all(label == -100 for label in labels):
            continue

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def data_collator(features):
    """将样本列表组装成 batch。"""
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    sft_root_dir = ensure_dir(Path(args.sft_root_dir))
    merged_output_dir = ensure_dir(sft_root_dir / "merged")
    lora_output_dir = sft_root_dir / "final_lora"

    print("========== new_rlhf SFT 配置 ==========")
    print(f"基础模型路径: {args.base_model_path}")
    print(f"SFT 训练集:   {args.train_data_path}")
    print(f"SFT 验证集:   {args.val_data_path}")
    print(f"SFT 根目录:   {sft_root_dir}")
    print(f"主输出模型:   {merged_output_dir}")
    print(f"LoRA 备份:    {lora_output_dir}")
    print(f"max_length:   {args.max_length}")
    print(f"seed:         {args.seed}")
    print("训练起点:      基础模型")
    print("训练结果:      merged 完整 SFT 模型")
    print("======================================")

    # tokenizer 明确从基础模型加载，确保词表与模型结构一致。
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型明确从基础模型加载，而不是从旧实验模型继续训练。
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    # LoRA 是训练手段，不是最终主结果。
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # 启用梯度检查点，减小显存压力。
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 保持可训练参数为 fp32，降低 AMP 训练时的数值问题。
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    train_dataset = read_sft_csv(args.train_data_path)
    val_dataset = read_sft_csv(args.val_data_path)

    tokenized_train = train_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val = val_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    if len(tokenized_train) == 0:
        raise ValueError("SFT 训练集在 tokenization 后为空，请检查数据或 max_length。")
    if len(tokenized_val) == 0:
        raise ValueError("SFT 验证集在 tokenization 后为空，请检查数据或 max_length。")

    training_args = TrainingArguments(
        # checkpoint 先写到 sft_v0 根目录；
        # 最终真正给下游使用的是 merged 目录。
        output_dir=str(sft_root_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.0,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 可选保存 LoRA 备份。
    if not args.skip_save_lora:
        ensure_dir(lora_output_dir)
        trainer.model.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))
        print(f"LoRA 备份已保存到: {lora_output_dir}")

    # 主结果：将 LoRA 合并回基础模型，导出完整 merged 模型。
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_output_dir))

    print(f"最终 merged SFT 模型已保存到: {merged_output_dir}")


if __name__ == "__main__":
    main()
