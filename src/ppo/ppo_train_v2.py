"""
new_rlhf 第二轮 PPO-V2 优化训练脚本

一、脚本目标
1. 在第一轮 PPO 模型 V1 的基础上继续做 PPO 强化学习优化。
2. 使用已经升级完成的 RM-V2 作为奖励模型，为新生成回答打奖励分。
3. 固定最初的 SFT-V0 模型作为 KL 参考模型，防止策略模型为了追求奖励分而过度偏离原始石化领域表达规范。
4. 本轮最终产出 PPO-V2，用于后续第三轮数据生成或最终效果评估。

二、三类模型角色
1. 策略模型 policy model，被优化者：
   model/ppo_v1/merged
   说明：这里使用 PPO-V1 已合并后的完整模型作为初始权重，PPO-V2 训练会在它上面新增 LoRA 参数并更新。

2. 奖励模型 reward model，裁判员：
   base: model/sft_v0/merged
   lora: model/rm_v2/final_lora
   说明：RM-V2 已经通过 R2 偏好数据增量训练，能更好识别第二轮回答优劣。

3. 参考模型 reference model，安全绳：
   model/sft_v0/merged
   说明：参考模型固定不训练，只用于 PPO 中 KL 散度约束，避免 PPO-V2 偏离 SFT-V0 的基础语言能力和领域格式。

三、默认训练数据
1. PPO prompt 池：
   data/01_测试集与验证集/PPO全局prompt池_1151题.csv
2. 该文件只需要包含 question 列；脚本会自动去重、打乱，然后循环采样训练。

四、默认关键参数
1. rounds = 3
2. steps_per_round = 200
3. 总 PPO 更新步数 = 600
4. learning_rate = 1e-6
5. init_kl_coef = 0.05
6. target_kl = 0.1
7. batch_size = 1
8. mini_batch_size = 1
9. max_new_tokens = 180
10. reward_clip = 5.0

五、输出目录
1. PPO-V2 过程快照：
   model/ppo_v2/round_1
   model/ppo_v2/round_2
   model/ppo_v2/round_3
2. PPO-V2 最终 LoRA：
   model/ppo_v2/final_lora
3. 日志：
   logs/ppo/ppo_v2_时间戳/console.log
   logs/ppo/ppo_v2_时间戳/metrics.csv
   logs/ppo/ppo_v2_时间戳/samples.csv

六、运行方式
直接在 PyCharm 中运行本脚本即可。默认路径和参数已经按第二轮 PPO-V2 实验配置好。
"""

import argparse
import gc
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from trl import PPOConfig, PPOTrainer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline.paths import PPO_LOG_DIR, PPO_PROMPT_FILE, PPO_V1_MODEL_DIR, PPO_V2_MODEL_DIR, RM_V2_MODEL_DIR, SFT_MODEL_DIR
from ppo_train import (
    append_csv_row,
    assert_paths,
    flatten_scalar_stats,
    load_reward_scorer,
    load_tokenizer,
    make_policy_model,
    make_reference_model,
    ppo_collator,
    PPODataset,
    save_ppo_model,
    setup_file_logging,
    validate_and_prepare_prompts,
)


DEFAULT_POLICY_MODEL_PATH = PPO_V1_MODEL_DIR / "merged"
DEFAULT_REFERENCE_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_REWARD_BASE_MODEL_PATH = SFT_MODEL_DIR / "merged"
DEFAULT_RM_LORA_PATH = RM_V2_MODEL_DIR / "final_lora"
DEFAULT_PPO_DATA_PATH = PPO_PROMPT_FILE
DEFAULT_OUTPUT_DIR = PPO_V2_MODEL_DIR
DEFAULT_LOG_DIR = PPO_LOG_DIR


def get_args() -> argparse.Namespace:
    """解析 PPO-V2 参数；默认值已经写死为第二轮实验配置。"""
    parser = argparse.ArgumentParser(description="PPO-V2 training: policy=PPO-V1, reward=RM-V2, reference=SFT-V0.")

    # 三类模型路径：策略模型、参考模型、奖励模型。
    parser.add_argument("--policy_model_path", type=Path, default=DEFAULT_POLICY_MODEL_PATH)
    parser.add_argument("--reference_model_path", type=Path, default=DEFAULT_REFERENCE_MODEL_PATH)
    parser.add_argument("--reward_base_model_path", type=Path, default=DEFAULT_REWARD_BASE_MODEL_PATH)
    parser.add_argument("--rm_lora_path", type=Path, default=DEFAULT_RM_LORA_PATH)

    # 数据与输出路径。
    parser.add_argument("--ppo_data_path", type=Path, default=DEFAULT_PPO_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run_name", default="", help="Default: ppo_v2_YYYYmmdd_HHMMSS.")
    parser.add_argument("--dry_run", action="store_true", help="Only validate paths/data and print settings.")

    # 训练规模。默认沿用第一轮 PPO 的训练强度，保证两轮优化可比较。
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps_per_round", type=int, default=200)
    parser.add_argument("--max_prompts", type=int, default=-1, help="Use all prompts when <= 0.")
    parser.add_argument("--save_every_round", action="store_true", default=True)

    # 生成参数。回答长度不宜过长，避免 RM 奖励被冗余文本干扰。
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # PPO 核心参数。学习率保持较小，因为这是从 PPO-V1 继续优化。
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

    # PPO-V2 新增 LoRA 参数。注意：这是叠加在 PPO-V1 merged 完整模型上的第二轮可训练参数。
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # RM 默认放 CPU，节省显存；如显存足够，可在 PyCharm 参数中改为 --rm_device cuda。
    parser.add_argument("--rm_device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def print_settings(args: argparse.Namespace, total_steps: int) -> None:
    """打印 PPO-V2 实验设置，方便在 PyCharm 控制台中确认角色没有配错。"""
    print("\nPPO-V2 settings:")
    print(f"策略模型 policy init: {args.policy_model_path}")
    print(f"参考模型 KL reference: {args.reference_model_path}")
    print(f"奖励模型 reward base: {args.reward_base_model_path}")
    print(f"奖励模型 RM-V2 LoRA: {args.rm_lora_path}")
    print(f"PPO prompt file: {args.ppo_data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")
    print(f"Rounds: {args.rounds}; steps/round: {args.steps_per_round}; total PPO steps: {total_steps}")
    print(f"KL init coef: {args.init_kl_coef}; target_kl: {args.target_kl}; lr: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}; mini_batch_size: {args.mini_batch_size}")


def release_memory(stage: str) -> None:
    """在大模型分阶段加载之间主动释放缓存，降低 Windows 下底层崩溃概率。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[memory] {stage}: 已执行 Python GC 和 CUDA cache 清理。")


def main() -> None:
    args = get_args()
    assert_paths(args)

    if not args.run_name.strip():
        args.run_name = datetime.now().strftime("ppo_v2_%Y%m%d_%H%M%S")

    log_paths = setup_file_logging(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_df = validate_and_prepare_prompts(args.ppo_data_path, args.seed, args.max_prompts)
    total_steps = args.rounds * args.steps_per_round
    print_settings(args, total_steps)

    if args.dry_run:
        print("Dry run finished: PPO-V2 paths, data, and settings are valid.")
        return

    # tokenizer 使用参考模型 SFT-V0，保证 KL 参考模型与策略模型解码词表一致。
    print("[stage 1/6] 加载 tokenizer，并构建 PPO prompt 数据集。")
    tokenizer = load_tokenizer(args.reference_model_path)
    dataset = PPODataset(prompt_df, tokenizer, args.max_prompt_length)
    release_memory("tokenizer/dataset ready")

    # 先加载被训练的 PPO-V1 策略模型，再加载固定的 SFT-V0 参考模型。
    # 这样能确保 reference 仍然是 SFT-V0，同时避免 CPU 上的 RM 先占住内存导致 policy 加载阶段崩溃。
    print("[stage 2/6] 加载策略模型 policy：PPO-V1 merged，后续会在它上面训练 PPO-V2 LoRA。")
    model = make_policy_model(args)
    release_memory("policy model loaded")

    print("[stage 3/6] 加载参考模型 reference：固定 SFT-V0，用于 KL 约束，不参与训练。")
    ref_model = make_reference_model(args)
    release_memory("reference model loaded")

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

    print("[stage 4/6] 初始化 PPOTrainer。")
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=ppo_collator,
    )
    release_memory("ppo trainer ready")

    # RM-V2 最后加载。RM 默认放 CPU，不改变实验定义，只减少 GPU 压力。
    print("[stage 5/6] 加载奖励模型 reward：SFT-V0 base + RM-V2 LoRA。")
    reward_scorer = load_reward_scorer(args, tokenizer)
    release_memory("reward model loaded")

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

    print("[stage 6/6] 开始 PPO-V2 训练循环。")
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
            "mean_rm_v2_reward": mean_reward,
            "min_rm_v2_reward": min(rm_scores),
            "max_rm_v2_reward": max(rm_scores),
            **scalar_stats,
        }
        append_csv_row(
            log_paths["metrics"],
            metric_row,
            [
                "step",
                "round",
                "mean_rm_v2_reward",
                "min_rm_v2_reward",
                "max_rm_v2_reward",
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
                "rm_v2_reward": rm_scores[0],
            },
            ["step", "round", "question", "answer", "rm_v2_reward"],
        )

        if step % args.log_every == 0 or step == 1:
            print(
                f"[PPO-V2 round {round_id}/{args.rounds} step {step}/{total_steps}] "
                f"mean_rm_v2_reward={mean_reward:.4f}; "
                f"sample_q={questions[0][:48]}; sample_a={answers[0][:80]}"
            )

        if step % args.steps_per_round == 0 and args.save_every_round:
            save_ppo_model(trainer, tokenizer, args.output_dir / f"round_{round_id}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_ppo_model(trainer, tokenizer, args.output_dir / "final_lora")
    print("PPO-V2 training finished.")


if __name__ == "__main__":
    main()
