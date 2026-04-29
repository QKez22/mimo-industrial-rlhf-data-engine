"""
new_rlhf 第一轮 RLHF 一键流程脚本

一、脚本用途
1. 将第一轮 RLHF 所需的所有阶段串成一次执行流程。
2. 解决手动多次运行多个脚本的问题，做到“一个入口自动顺序执行”。

二、默认执行顺序
1. SFT 训练
2. SFT 对比（Base vs SFT）
3. RM 联合训练（偏好对 + 直接打分）
4. RM 评估
5. PPO 优化
6. Base / SFT / PPO 最终效果对比

三、默认产出
1. SFT：
   - model/sft_v0/merged
   - model/sft_v0/final_lora
2. RM：
   - model/rm_v1/final_lora
3. PPO：
   - model/ppo_v1/round_1, round_2, round_3
   - model/ppo_v1/final_lora
4. 对比结果：
   - logs/eval/base_sft_ppo_generations.csv
   - logs/eval/base_sft_ppo_summary.csv

四、参数说明
1. --quick
   快速冒烟模式。会自动把 PPO 轮数与步数缩短，适合先验证流程能否跑通。
2. --skip_xxx
   跳过某个已完成阶段，避免重复训练。
3. --ppo_rounds / --ppo_steps_per_round
   控制 PPO 正式训练规模。
4. --compare_sample_size
   控制最终 Base/SFT/PPO 对比时使用的问题数量。
   默认值为 150，对应局部验证集_150条.csv 的全量验证。

五、适用场景
1. 第一次跑整套第一轮 RLHF
2. 中途某一步失败后，从后续阶段继续跑
3. 做快速冒烟、正式训练、对比导出
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import ROOT_DIR


def run_cmd(cmd: List[str], stage_name: str) -> None:
    """执行一个阶段命令。

    设计原则：
    1. 每个阶段单独打印标题，方便终端中定位问题。
    2. 如果某阶段失败，则立刻停止后续流程，避免脏结果继续传播。
    """
    print("\n" + "=" * 72)
    print(f"[Stage] {stage_name}")
    print("Command:", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    """解析一键流程参数。"""
    parser = argparse.ArgumentParser(description="第一轮 RLHF 一键流程：SFT -> RM -> PPO -> 对比")
    parser.add_argument("--quick", action="store_true", help="快速冒烟模式：缩短 PPO 轮次和步数。")
    parser.add_argument("--skip_sft", action="store_true", help="跳过 SFT 训练阶段。")
    parser.add_argument("--skip_sft_compare", action="store_true", help="跳过 SFT 对比阶段。")
    parser.add_argument("--skip_rm", action="store_true", help="跳过 RM 训练阶段。")
    parser.add_argument("--skip_rm_eval", action="store_true", help="跳过 RM 评估阶段。")
    parser.add_argument("--skip_ppo", action="store_true", help="跳过 PPO 优化阶段。")
    parser.add_argument("--skip_final_compare", action="store_true", help="跳过最终 Base/SFT/PPO 对比阶段。")
    parser.add_argument("--ppo_rounds", type=int, default=3, help="正式运行时的 PPO 轮数。默认 3。")
    parser.add_argument("--ppo_steps_per_round", type=int, default=200, help="正式运行时每轮 PPO 步数。默认 200。")
    parser.add_argument("--compare_sample_size", type=int, default=150, help="最终效果对比时默认使用局部验证集_150条.csv 的全量 150 条。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = sys.executable

    # 各阶段脚本入口
    sft_train = SRC_DIR / "sft" / "sft_train.py"
    sft_compare = SRC_DIR / "sft" / "sft_infer_compare.py"
    rm_train = SRC_DIR / "rm" / "rm_train.py"
    rm_eval = SRC_DIR / "rm" / "rm_eval.py"
    ppo_train = SRC_DIR / "ppo" / "ppo_train.py"
    final_compare = SCRIPT_DIR / "compare_base_sft_ppo.py"

    # quick 模式下自动缩小 PPO 规模，便于快速验证流程。
    ppo_rounds = 1 if args.quick else args.ppo_rounds
    ppo_steps = 20 if args.quick else args.ppo_steps_per_round
    ppo_max_prompts = 128 if args.quick else -1

    if not args.skip_sft:
        run_cmd([python_exe, str(sft_train)], "SFT 训练")
    else:
        print("[Skip] SFT 训练")

    if not args.skip_sft_compare:
        run_cmd([python_exe, str(sft_compare)], "SFT 对比（Base vs SFT）")
    else:
        print("[Skip] SFT 对比")

    if not args.skip_rm:
        run_cmd([python_exe, str(rm_train)], "RM 联合训练（偏好对 + 直接打分）")
    else:
        print("[Skip] RM 训练")

    if not args.skip_rm_eval:
        run_cmd([python_exe, str(rm_eval)], "RM 评估")
    else:
        print("[Skip] RM 评估")

    if not args.skip_ppo:
        run_cmd(
            [
                python_exe,
                str(ppo_train),
                "--rounds",
                str(ppo_rounds),
                "--steps_per_round",
                str(ppo_steps),
                "--max_prompts",
                str(ppo_max_prompts),
            ],
            "PPO 优化",
        )
    else:
        print("[Skip] PPO 优化")

    if not args.skip_final_compare:
        run_cmd(
            [
                python_exe,
                str(final_compare),
                "--sample_size",
                str(args.compare_sample_size),
            ],
            "最终对比（Base / SFT / PPO）",
        )
    else:
        print("[Skip] 最终对比")

    print("\n第一轮 RLHF 流程执行完成。")
    print("请查看 logs/eval 目录下的最终对比结果。")


if __name__ == "__main__":
    main()
