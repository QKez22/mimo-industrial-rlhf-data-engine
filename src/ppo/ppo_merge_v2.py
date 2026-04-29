"""
new_rlhf PPO-V2 合并导出脚本

一、脚本目标
1. 将 PPO-V2 训练得到的 LoRA 适配器合并到 PPO-V1 完整模型上。
2. 得到可以直接用于推理、评估、第三轮数据生成的完整 PPO-V2 模型。

二、默认输入
1. PPO-V2 的底座：
   model/ppo_v1/merged
   说明：PPO-V2 训练时就是以 PPO-V1 merged 为初始策略模型，因此合并时也必须使用同一个底座。
2. PPO-V2 LoRA：
   model/ppo_v2/final_lora

三、默认输出
1. 合并后的 PPO-V2 完整模型：
   model/ppo_v2/merged

四、运行方式
PPO-V2 训练完成后，直接在 PyCharm 中运行本脚本即可。
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

from pipeline.paths import PPO_V1_MODEL_DIR, PPO_V2_MODEL_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    """解析 PPO-V2 合并参数；默认路径已经按第二轮 PPO 配置好。"""
    parser = argparse.ArgumentParser(description="Merge PPO-V2 LoRA into PPO-V1 merged model.")
    parser.add_argument("--base_model_path", default=str(PPO_V1_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_lora_path", default=str(PPO_V2_MODEL_DIR / "final_lora"))
    parser.add_argument("--output_dir", default=str(PPO_V2_MODEL_DIR / "merged"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    for path, label in [
        (Path(args.base_model_path), "PPO-V1 merged 底座"),
        (Path(args.ppo_lora_path), "PPO-V2 LoRA"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在：{path}")

    print(f"加载 PPO-V2 合并底座：{args.base_model_path}")
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

    print(f"加载 PPO-V2 LoRA：{args.ppo_lora_path}")
    ppo_model = PeftModel.from_pretrained(base_model, args.ppo_lora_path)
    ppo_model.eval()

    print("开始合并 PPO-V2 LoRA。")
    merged_model = ppo_model.merge_and_unload()

    print(f"保存 PPO-V2 merged 完整模型到：{output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("PPO-V2 merged 模型保存完成。")


if __name__ == "__main__":
    main()
