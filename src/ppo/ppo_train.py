"""
new_rlhf PPO 训练脚本（第一轮默认配置）

一、脚本目标
1. 以 SFT 模型作为策略初始权重，进行 PPO 强化学习。
2. 奖励由 RM（base + rm_lora）给出，KL 约束由参考模型提供。
3. 输出每轮模型快照与最终 LoRA，支持三轮迭代范式。

二、默认输入
1. policy_model_path:     model/sft_v0/merged（被优化策略模型）
2. reference_model_path:  model/sft_v0/merged（KL 参考模型，默认与 SFT 一致）
3. reward_base_model_path:model/sft_v0/merged（RM 的 base）
4. rm_lora_path:          model/rm_v1/final_lora
5. ppo_data_path:         data/01_测试集与验证集/PPO全局prompt池_1151题.csv

三、默认训练参数
1. rounds=3，steps_per_round=200（总步数=600）
2. learning_rate=1e-6
3. init_kl_coef=0.05，target_kl=0.1
4. gamma=0.99，lam=0.95
5. batch_size=1，mini_batch_size=1

四、输出与日志
1. 模型输出目录：model/ppo_v1
2. 轮次快照：output_dir/round_1, round_2, round_3
3. 最终结果：output_dir/final_lora
4. 日志目录：logs/ppo/<run_name>/
   - console.log
   - metrics.csv
   - samples.csv
"""

import argparse
import csv
import os
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("WANDB_DISABLED", "true")

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import PPO_LOG_DIR, PPO_PROMPT_FILE, PPO_V1_MODEL_DIR, RM_V1_MODEL_DIR, SFT_MODEL_DIR


DEFAULT_POLICY_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_REFERENCE_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_REWARD_BASE_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_RM_LORA_PATH = RM_V1_MODEL_DIR / "final_lora"
DEFAULT_PPO_DATA_PATH = PPO_PROMPT_FILE
DEFAULT_OUTPUT_DIR = PPO_V1_MODEL_DIR
DEFAULT_LOG_DIR = PPO_LOG_DIR


class Tee:
    """把控制台输出同时写入终端和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def read_csv_required(path: Path, required_columns: List[str]) -> pd.DataFrame:
    """读取 CSV 并校验必需列，兼容常见中文环境编码。"""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            if encoding != "utf-8-sig":
                print(f"[read_csv_required] {path.name} encoding: {encoding}")
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}; columns={df.columns.tolist()}")
    return df


def validate_and_prepare_prompts(csv_path: Path, seed: int, max_prompts: int) -> pd.DataFrame:
    """清洗并打乱 PPO prompt 池。"""
    df = read_csv_required(csv_path, ["question"])
    original_count = len(df)
    df = df.copy()
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"] != ""].drop_duplicates(subset=["question"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid PPO prompts found in {csv_path}")

    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    df = df.iloc[indices].reset_index(drop=True)

    if max_prompts > 0:
        df = df.head(max_prompts).reset_index(drop=True)

    print(f"PPO prompt file: {csv_path}")
    print(f"Raw rows: {original_count}; usable unique prompts: {len(df)}")
    if "source" in df.columns:
        print("Source distribution:")
        print(df["source"].value_counts().to_string())
    if "question_type" in df.columns:
        print("Question type distribution:")
        print(df["question_type"].value_counts().to_string())
    return df


def build_generation_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"


def build_rm_prompt(question: str, answer: str) -> str:
    return (
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer.strip()}<|im_end|>"
    )


class PPODataset(Dataset):
    """将问题列表转换为 PPOTrainer 可用的数据集。"""

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_prompt_length: int):
        self.items = []
        for _, row in df.iterrows():
            question = str(row["question"]).strip()
            prompt = build_generation_prompt(question)
            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_length,
                padding=False,
                return_tensors=None,
            )
            if not encoded["input_ids"]:
                continue
            self.items.append(
                {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "query": prompt,
                    "question": question,
                }
            )

        if not self.items:
            raise ValueError("PPO dataset is empty after tokenization.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def ppo_collator(batch: List[Dict]) -> Dict:
    return {key: [item[key] for item in batch] for key in batch[0]}


def get_args() -> argparse.Namespace:
    """解析 PPO 参数。"""
    parser = argparse.ArgumentParser(description="PPO training with SFT policy and RM rewards.")
    parser.add_argument("--policy_model_path", type=Path, default=DEFAULT_POLICY_MODEL_PATH)
    parser.add_argument("--reference_model_path", type=Path, default=DEFAULT_REFERENCE_MODEL_PATH)
    parser.add_argument("--reward_base_model_path", type=Path, default=DEFAULT_REWARD_BASE_MODEL_PATH)
    parser.add_argument("--rm_lora_path", type=Path, default=DEFAULT_RM_LORA_PATH)
    parser.add_argument("--ppo_data_path", type=Path, default=DEFAULT_PPO_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run_name", default="", help="Default: ppo_YYYYmmdd_HHMMSS.")
    parser.add_argument("--dry_run", action="store_true", help="Only validate paths/data and print settings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps_per_round", type=int, default=200)
    parser.add_argument("--max_prompts", type=int, default=-1, help="Use all prompts when <= 0.")
    parser.add_argument("--save_every_round", action="store_true", default=True)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--init_kl_coef", type=float, default=0.05)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--rm_device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def assert_paths(args: argparse.Namespace) -> None:
    """统一解析相对路径并检查关键输入是否存在。"""
    root = SCRIPT_DIR.parents[2]
    args.policy_model_path = args.policy_model_path if args.policy_model_path.is_absolute() else root / args.policy_model_path
    args.reference_model_path = args.reference_model_path if args.reference_model_path.is_absolute() else root / args.reference_model_path
    args.reward_base_model_path = args.reward_base_model_path if args.reward_base_model_path.is_absolute() else root / args.reward_base_model_path
    args.rm_lora_path = args.rm_lora_path if args.rm_lora_path.is_absolute() else root / args.rm_lora_path
    args.ppo_data_path = args.ppo_data_path if args.ppo_data_path.is_absolute() else root / args.ppo_data_path
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    args.log_dir = args.log_dir if args.log_dir.is_absolute() else root / args.log_dir

    for path, label in (
        (args.policy_model_path, "policy model"),
        (args.reference_model_path, "reference model"),
        (args.reward_base_model_path, "reward base model"),
        (args.rm_lora_path, "RM LoRA"),
        (args.ppo_data_path, "PPO data"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def setup_file_logging(args: argparse.Namespace) -> Dict[str, Path]:
    """创建本次运行日志目录，并接管 stdout/stderr。"""
    run_name = args.run_name.strip() or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    args.run_name = run_name

    run_dir = args.log_dir / run_name
    paths = {
        "console": run_dir / "console.log",
        "metrics": run_dir / "metrics.csv",
        "samples": run_dir / "samples.csv",
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(paths["console"], "a", encoding="utf-8")
    except PermissionError:
        fallback_dir = DEFAULT_LOG_DIR / run_name
        sys.__stdout__.write(f"Cannot write log_dir={args.log_dir}; fallback to {fallback_dir}\n")
        args.log_dir = DEFAULT_LOG_DIR
        fallback_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "console": fallback_dir / "console.log",
            "metrics": fallback_dir / "metrics.csv",
            "samples": fallback_dir / "samples.csv",
        }
        log_file = open(paths["console"], "a", encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"Console log: {paths['console']}")
    print(f"Metrics CSV: {paths['metrics']}")
    print(f"Samples CSV: {paths['samples']}")
    return paths


def load_tokenizer(model_path: Path) -> AutoTokenizer:
    """加载 tokenizer，并设置 left padding（适配 PPO 生成）。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def make_policy_model(args: argparse.Namespace) -> AutoModelForCausalLMWithValueHead:
    """加载被优化的策略模型，并叠加 PPO 阶段 LoRA。"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(args.policy_model_path),
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        peft_config=peft_config,
    )

    model.pretrained_model.config.use_cache = False
    if hasattr(model.pretrained_model, "gradient_checkpointing_enable"):
        model.pretrained_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.pretrained_model, "enable_input_require_grads"):
        model.pretrained_model.enable_input_require_grads()

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def make_reference_model(args: argparse.Namespace) -> AutoModelForCausalLMWithValueHead:
    """加载 KL 参考模型（全程冻结，不做梯度更新）。"""
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(args.reference_model_path),
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.pretrained_model.config.use_cache = False
    return model


@dataclass
class RewardScorer:
    """将 RM 打分封装成批量接口。"""

    model: torch.nn.Module
    tokenizer: AutoTokenizer
    device: torch.device
    max_length: int
    reward_clip: float

    @torch.no_grad()
    def score(self, questions: List[str], answers: List[str]) -> List[float]:
        prompts = [build_rm_prompt(q, a) for q, a in zip(questions, answers)]
        encoded = self.tokenizer(prompts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits.view(-1).detach().float().cpu()
        scores = logits.tolist()
        if self.reward_clip > 0:
            scores = [max(-self.reward_clip, min(self.reward_clip, score)) for score in scores]
        return scores


def load_reward_scorer(args: argparse.Namespace, tokenizer: AutoTokenizer) -> RewardScorer:
    """加载 RM（base + LoRA），用于 PPO 过程奖励计算。"""
    if args.rm_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.rm_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--rm_device cuda was set, but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_rm = AutoModelForSequenceClassification.from_pretrained(
        args.reward_base_model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    base_rm.config.pad_token_id = tokenizer.pad_token_id
    rm_model = PeftModel.from_pretrained(base_rm, args.rm_lora_path, is_trainable=False)
    rm_model.to(device)
    rm_model.eval()
    print(f"RM scorer device: {device}")
    return RewardScorer(rm_model, tokenizer, device, args.max_prompt_length + args.max_new_tokens, args.reward_clip)


def save_ppo_model(trainer: PPOTrainer, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    """保存当前轮次的 PPO LoRA 与 tokenizer。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved PPO model to: {output_dir}")


def scalar_value(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu().item()
    elif hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            if hasattr(value, "mean"):
                value = value.mean().item()
            else:
                return None
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        converted = [scalar_value(item) for item in value]
        converted = [item for item in converted if item is not None]
        return sum(converted) / len(converted) if converted else None

    if isinstance(value, (int, float)):
        return float(value)
    return None


def flatten_scalar_stats(stats: Dict[str, Any]) -> Dict[str, float]:
    """将 PPOTrainer 返回的复杂统计结构扁平化为标量字典。"""
    flattened = {}
    for key, value in stats.items():
        scalar = scalar_value(value)
        if scalar is not None:
            flattened[key] = scalar
    return flattened


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """按需写入 CSV（首次写入时自动补表头）。"""
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = get_args()
    assert_paths(args)
    log_paths = setup_file_logging(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_df = validate_and_prepare_prompts(args.ppo_data_path, args.seed, args.max_prompts)
    total_steps = args.rounds * args.steps_per_round

    print("\nPPO settings:")
    print(f"Policy init: {args.policy_model_path}")
    print(f"KL reference model: {args.reference_model_path}")
    print(f"Reward base model: {args.reward_base_model_path}")
    print(f"RM adapter: {args.rm_lora_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")
    print(f"Rounds: {args.rounds}; steps/round: {args.steps_per_round}; total PPO steps: {total_steps}")
    print(f"KL init coef: {args.init_kl_coef}; lr: {args.learning_rate}; batch_size: {args.batch_size}")

    if args.dry_run:
        print("Dry run finished: data and paths are valid.")
        return

    # tokenizer 跟参考模型保持一致，保证 KL 计算与解码行为一致
    tokenizer = load_tokenizer(args.reference_model_path)
    dataset = PPODataset(prompt_df, tokenizer, args.max_prompt_length)
    reward_scorer = load_reward_scorer(args, tokenizer)
    model = make_policy_model(args)
    ref_model = make_reference_model(args)

    ppo_config = PPOConfig(
        model_name=str(args.policy_model_path),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.init_kl_coef,
        gamma=args.gamma,
        lam=args.lam,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        optimize_cuda_cache=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=ppo_collator,
    )

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 8,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_prompt": False,
    }

    # 按总步数循环执行 PPO 更新，并按轮次保存模型快照
    step = 0
    data_iter = iter(trainer.dataloader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.dataloader)
            batch = next(data_iter)

        step += 1
        round_id = (step - 1) // args.steps_per_round + 1
        query_tensors = [tensor.to(trainer.accelerator.device) for tensor in batch["input_ids"]]
        questions = batch["question"]

        response_tensors = trainer.generate(query_tensors, batch_size=len(query_tensors), **generation_kwargs)
        if isinstance(response_tensors, torch.Tensor):
            response_tensors = list(response_tensors)

        answers = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        answers = [answer.strip() for answer in answers]
        rm_scores = reward_scorer.score(questions, answers)
        rewards = [torch.tensor(score, dtype=torch.float32, device=trainer.accelerator.device) for score in rm_scores]

        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards)
        scalar_stats = flatten_scalar_stats(stats)
        mean_reward = sum(rm_scores) / len(rm_scores)

        metric_row = {
            "step": step,
            "round": round_id,
            "mean_rm_reward": mean_reward,
            "min_rm_reward": min(rm_scores),
            "max_rm_reward": max(rm_scores),
            **scalar_stats,
        }
        append_csv_row(
            log_paths["metrics"],
            metric_row,
            [
                "step",
                "round",
                "mean_rm_reward",
                "min_rm_reward",
                "max_rm_reward",
                "objective/kl",
                "objective/entropy",
                "objective/non_score_reward",
                "objective/rlhf_reward",
                "ppo/loss/policy",
                "ppo/loss/value",
                "ppo/loss/total",
                "ppo/learning_rate",
            ],
        )
        append_csv_row(
            log_paths["samples"],
            {
                "step": step,
                "round": round_id,
                "question": questions[0],
                "answer": answers[0],
                "rm_reward": rm_scores[0],
            },
            ["step", "round", "question", "answer", "rm_reward"],
        )

        if step % args.log_every == 0 or step == 1:
            print(
                f"[round {round_id}/{args.rounds} step {step}/{total_steps}] "
                f"mean_rm_reward={mean_reward:.4f}; sample_q={questions[0][:48]}; sample_a={answers[0][:80]}"
            )

        if step % args.steps_per_round == 0 and args.save_every_round:
            save_ppo_model(trainer, tokenizer, args.output_dir / f"round_{round_id}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_ppo_model(trainer, tokenizer, args.output_dir / "final_lora")
    print("PPO training finished.")


if __name__ == "__main__":
    main()
