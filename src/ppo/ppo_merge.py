"""
new_rlhf PPO 合并脚本

一、脚本用途
1. 将 PPO 阶段输出的 LoRA 适配器与 SFT merged 底模合并为一个完整模型。
2. 便于后续做稳定推理与自动指标评测，避免在评测阶段再次走 PEFT + 自动卸载链路。

二、默认输入
1. 底模：model/sft_v0/merged
2. PPO LoRA：model/ppo_v1/final_lora

三、默认输出
1. model/ppo_v1/merged

四、说明
1. 该脚本只负责“合并并导出完整模型”，不参与训练。
2. 合并完成后，后续评测脚本可直接传入 --ppo_merged_model_path 使用。
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import PPO_V1_MODEL_DIR, SFT_MODEL_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    """解析 PPO 合并参数。"""
    parser = argparse.ArgumentParser(description="Merge PPO LoRA into a full model for stable inference.")
    parser.add_argument("--base_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_lora_path", default=str(PPO_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--output_dir", default=str(PPO_V1_MODEL_DIR / "merged"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    print(f"加载 PPO 底模：{args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    base_model.to(args.device)
    base_model.eval()

    print(f"加载 PPO LoRA：{args.ppo_lora_path}")
    ppo_model = PeftModel.from_pretrained(base_model, args.ppo_lora_path)
    ppo_model.eval()

    print("开始合并 PPO LoRA。")
    merged_model = ppo_model.merge_and_unload()

    print(f"保存合并后完整模型到：{output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("PPO merged 模型已保存完成。")


if __name__ == "__main__":
    main()
