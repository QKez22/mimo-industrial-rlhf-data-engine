"""
new_rlhf R2 PPO-V1 answer generation.

This script reads R2_保留池模板_400条_修正版.csv and generates two answers
for each question with PPO-V1. It handles one question at a time. It does not
ask the model to output tables or column names; the script writes the table
columns itself.

Default output:
- R2_保留池模板_400条_修正版_填充PPO答案_清洁版.xlsx
- R2_保留池模板_400条_修正版_填充PPO答案_清洁版_checkpoint.csv

Why this version is strict:
- The earlier prompt included meta instructions such as "核心约束" and
  "输出格式", and the local model sometimes copied them into answers.
- This version uses a short "material + question + angle" prompt.
- If the generated answer still contains prompt/template leakage, the answer
  is left blank and marked in ppo_generation_note instead of polluting data.
"""

import argparse
import gc
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import PPO_V1_MODEL_DIR, SFT_MODEL_DIR, ensure_dir


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

FORBIDDEN_OUTPUT_MARKERS = [
    "题目模板",
    "prompt",
    "Prompt",
    "我是AI",
    "我是 AI",
    "在此提醒",
    "文档内容",
    "参考文档",
    "需要回答的问题",
    "回答要求",
    "核心约束",
    "输出格式",
    "answer_ppo_a",
    "answer_ppo_b",
    "列标题",
    "制表符",
    "chunk_text",
    "必须遵守",
    "闭卷",
    "不得编造",
    "150字以内",
    "从技术机理",
    "从执行操作",
]


def parse_args() -> argparse.Namespace:
    """Parse arguments. Defaults are suitable for direct PyCharm run."""
    data_root = Path(__file__).resolve().parents[2] / "data"
    r2_csv = next(data_root.rglob("R2_保留池模板_400条_修正版.csv"))
    r2_dir = r2_csv.parent

    parser = argparse.ArgumentParser(description="Generate clean answer_ppo_a/answer_ppo_b for R2.")
    parser.add_argument("--input_csv", default=str(r2_csv))
    parser.add_argument("--output_xlsx", default=str(r2_dir / "R2_保留池模板_400条_修正版_填充PPO答案_清洁版.xlsx"))
    parser.add_argument(
        "--checkpoint_csv",
        default=str(r2_dir / "R2_保留池模板_400条_修正版_填充PPO答案_清洁版_checkpoint.csv"),
    )
    parser.add_argument("--ppo_merged_model_path", default=str(PPO_V1_MODEL_DIR / "merged"))
    parser.add_argument("--sft_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_lora_path", default=str(PPO_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--offload_dir", default=str(r2_dir / "ppo_r2_offload"))
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def angle_instruction(answer_kind: str) -> str:
    """Return the desired answer angle."""
    if answer_kind == "a":
        return "侧重说明原因、作用或概念含义，语言准确完整。"
    if answer_kind == "b":
        return "侧重说明执行做法、管理要求或实施步骤，语言清晰可操作。"
    raise ValueError(f"Unknown answer kind: {answer_kind}")


def build_prompt(question: str, material: str, answer_kind: str, attempt: int) -> str:
    """Build a short one-question prompt with minimal meta language."""
    angle = angle_instruction(answer_kind)
    if attempt == 0:
        user_text = f"""你是石化行业专业知识专家。请根据材料回答问题。

材料：
{material.strip()}

问题：
{question.strip()}

回答角度：
{angle}

请直接给出答案正文，不写说明、表头、题号或自我介绍。"""
    elif attempt == 1:
        user_text = f"""材料：
{material.strip()}

问题：
{question.strip()}

{angle}
直接回答。"""
    else:
        cue = "说明原因或含义。" if answer_kind == "a" else "说明做法或步骤。"
        user_text = f"材料：{material.strip()}\n问题：{question.strip()}\n{cue}"

    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_raw_output(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Generate raw model text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant\n"
    answer = text.split(marker, 1)[1] if marker in text else text
    return answer.replace("<|im_end|>", "").strip()


def trim_to_complete_sentence(text: str, max_chars: int = 150) -> str:
    """Trim to about 150 Chinese characters while preserving complete sentence."""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    cut_at = max(clipped.rfind(mark) for mark in ["。", "；", ";", "！", "？"])
    if cut_at >= 40:
        return clipped[: cut_at + 1].strip()
    return clipped.rstrip("，、,；;：:")


def clean_answer(text: str) -> str:
    """Clean one model answer."""
    text = str(text).strip().strip("`")
    for prefix in ["answer_ppo_a", "answer_ppo_b", "答案", "回答"]:
        for sep in [":", "："]:
            marker = f"{prefix}{sep}"
            if text.lower().startswith(marker.lower()):
                text = text[len(marker) :].strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return trim_to_complete_sentence(text, max_chars=150)


def has_prompt_leak(answer: str) -> bool:
    """Detect prompt/template leakage."""
    return any(marker in str(answer) for marker in FORBIDDEN_OUTPUT_MARKERS)


def generate_clean_answer(model, tokenizer, question: str, material: str, answer_kind: str, max_new_tokens: int) -> str:
    """Generate one clean answer. Return blank if all attempts leak prompt text."""
    for attempt in range(3):
        prompt = build_prompt(question, material, answer_kind, attempt)
        answer = clean_answer(generate_raw_output(model, tokenizer, prompt, max_new_tokens))
        if answer and not has_prompt_leak(answer):
            return answer
    return ""


def load_model(args: argparse.Namespace):
    """Load PPO-V1, preferring merged full model."""
    if Path(args.ppo_merged_model_path).exists():
        print(f"加载 PPO merged 模型：{args.ppo_merged_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.ppo_merged_model_path, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.ppo_merged_model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        model.eval()
        return tokenizer, model

    print("未检测到 PPO merged 模型，回退为 SFT merged + PPO LoRA。")
    offload_dir = ensure_dir(Path(args.offload_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        offload_folder=str(offload_dir),
        offload_state_dict=True,
    )
    model = PeftModel.from_pretrained(base_model, args.ppo_lora_path, offload_folder=str(offload_dir))
    model.eval()
    return tokenizer, model


def load_dataframe(input_csv: str) -> pd.DataFrame:
    """Load R2 revised template."""
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    required_columns = ["question_id", "chunk_id", "chunk_text", "question"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要列：{missing}；文件：{input_csv}")
    for column in ["answer_ppo_a", "answer_ppo_b", "ppo_generation_note"]:
        if column not in df.columns:
            df[column] = ""
    return df


def should_skip(row: pd.Series, skip_existing: bool) -> bool:
    """Optionally skip rows that already have both answers."""
    if not skip_existing:
        return False
    answer_a = str(row.get("answer_ppo_a", "")).strip()
    answer_b = str(row.get("answer_ppo_b", "")).strip()
    return bool(answer_a and answer_b and answer_a.lower() != "nan" and answer_b.lower() != "nan")


def save_checkpoint(df: pd.DataFrame, checkpoint_csv: str) -> None:
    """Save checkpoint CSV."""
    path = Path(checkpoint_csv)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_final_xlsx(df: pd.DataFrame, output_xlsx: str) -> None:
    """Save final Excel."""
    path = Path(output_xlsx)
    ensure_dir(path.parent)
    df.to_excel(path, index=False)


def cleanup_model(model) -> None:
    """Release model memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.input_csv)
    tokenizer, model = load_model(args)

    total = len(df)
    print(f"开始逐题处理第二轮保留池修正版，共 {total} 条。")

    updated = 0
    for idx in range(total):
        row = df.iloc[idx]
        if should_skip(row, args.skip_existing):
            continue

        question = str(row["question"]).strip()
        material = str(row["chunk_text"]).strip()
        if not question or not material:
            print(f"[{idx + 1}/{total}] question 或 chunk_text 为空，跳过。")
            continue

        answer_a = generate_clean_answer(model, tokenizer, question, material, "a", args.max_new_tokens)
        answer_b = generate_clean_answer(model, tokenizer, question, material, "b", args.max_new_tokens)

        df.at[idx, "answer_ppo_a"] = answer_a
        df.at[idx, "answer_ppo_b"] = answer_b
        df.at[idx, "ppo_generation_note"] = "" if answer_a and answer_b else "模板泄漏或空答案，已置空待重跑/人工检查"
        updated += 1

        print(f"[{idx + 1}/{total}] 已生成 answer_ppo_a_len={len(answer_a)} answer_ppo_b_len={len(answer_b)}")

        if updated % args.save_every == 0:
            save_checkpoint(df, args.checkpoint_csv)
            print(f"已保存 checkpoint：{args.checkpoint_csv}")

    save_checkpoint(df, args.checkpoint_csv)
    save_final_xlsx(df, args.output_xlsx)
    cleanup_model(model)

    print(f"最终 Excel 已保存：{args.output_xlsx}")
    print(f"最终 checkpoint CSV 已保存：{args.checkpoint_csv}")


if __name__ == "__main__":
    main()
