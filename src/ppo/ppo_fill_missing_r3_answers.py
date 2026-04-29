"""
new_rlhf R3 PPO-V2 missing-answer refill script.

Purpose:
1. Read the already generated R3 answer workbook.
2. Keep all existing non-empty answers unchanged.
3. Re-generate only blank answer_ppo_a / answer_ppo_b cells.
4. Reuse the exact generation, cleaning, model-loading, and closed-book
   citation-check logic from ppo_generate_r3_answers.py.

Default input:
- data/04_R3_第三轮RLHF/R3_保留池模板_399条_修正版_填充PPO答案_清洁版.xlsx

Default output:
- data/04_R3_第三轮RLHF/R3_保留池模板_399条_修正版_填充PPO答案_清洁版_补全.xlsx

Default checkpoint:
- data/04_R3_第三轮RLHF/R3_保留池模板_399条_修正版_填充PPO答案_清洁版_补全_checkpoint.csv

Notes:
- This script does not regenerate the whole table.
- If a blank answer still fails all retries because of suspected external
  citations or template leakage, it remains blank and ppo_generation_note records
  the reason.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline.paths import DATA_DIR, PPO_V1_MODEL_DIR, PPO_V2_MODEL_DIR
from ppo_generate_r3_answers import (
    build_prompt,
    clean_answer,
    cleanup_model,
    extract_unsupported_references,
    generate_raw_output,
    has_prompt_leak,
    load_model,
    save_checkpoint,
    save_final_xlsx,
)


R3_DIR = DATA_DIR / "04_R3_第三轮RLHF"
DEFAULT_INPUT_XLSX = R3_DIR / "R3_保留池模板_399条_修正版_填充PPO答案_清洁版.xlsx"
DEFAULT_OUTPUT_XLSX = R3_DIR / "R3_保留池模板_399条_修正版_填充PPO答案_清洁版_补全.xlsx"
DEFAULT_CHECKPOINT_CSV = R3_DIR / "R3_保留池模板_399条_修正版_填充PPO答案_清洁版_补全_checkpoint.csv"


def parse_args() -> argparse.Namespace:
    """Parse arguments. Defaults are suitable for direct PyCharm run."""
    parser = argparse.ArgumentParser(description="Fill only blank R3 PPO answer cells.")
    parser.add_argument("--input_xlsx", default=str(DEFAULT_INPUT_XLSX))
    parser.add_argument("--output_xlsx", default=str(DEFAULT_OUTPUT_XLSX))
    parser.add_argument("--checkpoint_csv", default=str(DEFAULT_CHECKPOINT_CSV))
    parser.add_argument("--ppo_merged_model_path", default=str(PPO_V2_MODEL_DIR / "merged"))
    parser.add_argument("--base_model_path", default=str(PPO_V1_MODEL_DIR / "merged"))
    parser.add_argument("--ppo_lora_path", default=str(PPO_V2_MODEL_DIR / "final_lora"))
    parser.add_argument("--offload_dir", default=str(R3_DIR / "ppo_r3_fill_missing_offload"))
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def is_blank(value: object) -> bool:
    """Treat NaN, empty strings, and literal 'nan' as blank."""
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def load_existing_workbook(path: str) -> pd.DataFrame:
    """Load the generated R3 workbook and validate required columns."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    df = pd.read_excel(input_path)
    required_columns = ["question_id", "chunk_id", "chunk_text", "question", "answer_ppo_a", "answer_ppo_b"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input workbook missing required columns: {missing}; file={input_path}")
    if "ppo_generation_note" not in df.columns:
        df["ppo_generation_note"] = ""
    return df


def build_allowed_context(row: pd.Series) -> str:
    """Use the same closed-book context scope as ppo_generate_r3_answers.py."""
    return "\n".join(
        str(row.get(column, ""))
        for column in ["chunk_text", "source", "section_title"]
    )


def refill_one_cell(
    model,
    tokenizer,
    row: pd.Series,
    answer_kind: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Generate one missing answer with a relaxed refill policy.

    Refill policy:
    1. First, keep the same prompt/clean/retry flow as the original R3 script.
    2. If all retries fail only because of unsupported external references,
       directly keep the last non-empty cleaned answer instead of leaving it blank.
    3. Template leakage or truly empty outputs are still rejected.
    """
    question = str(row["question"]).strip()
    material = str(row["chunk_text"]).strip()
    if not question or not material or question.lower() == "nan" or material.lower() == "nan":
        return "", "question 或 chunk_text 为空"
    allowed_context = build_allowed_context(row)

    last_nonempty_answer = ""
    last_reject_reason = "空答案"
    for attempt in range(3):
        prompt = build_prompt(question, material, answer_kind, attempt)
        answer = clean_answer(generate_raw_output(model, tokenizer, prompt, max_new_tokens))
        if answer:
            last_nonempty_answer = answer
        if not answer:
            last_reject_reason = "空答案"
            continue
        if has_prompt_leak(answer):
            last_reject_reason = "模板泄漏"
            continue
        unsupported_references = extract_unsupported_references(answer, allowed_context)
        if unsupported_references:
            preview = "、".join(unsupported_references[:4])
            last_reject_reason = f"疑似外部引用：{preview}"
            continue
        return answer, ""

    if last_nonempty_answer and last_reject_reason.startswith("疑似外部引用"):
        return last_nonempty_answer, "多次外部引用，已直接保留最后一次回答"

    return "", last_reject_reason


def main() -> None:
    args = parse_args()
    df = load_existing_workbook(args.input_xlsx)

    blank_a = int(df["answer_ppo_a"].apply(is_blank).sum())
    blank_b = int(df["answer_ppo_b"].apply(is_blank).sum())
    print(f"读取 R3 已生成结果：{args.input_xlsx}")
    print(f"待补全 answer_ppo_a 空白数：{blank_a}")
    print(f"待补全 answer_ppo_b 空白数：{blank_b}")

    if blank_a == 0 and blank_b == 0:
        print("没有空白答案需要补全，直接保存一份输出文件。")
        save_final_xlsx(df, args.output_xlsx)
        return

    tokenizer, model = load_model(args)
    updated_cells = 0
    total_rows = len(df)

    for idx in range(total_rows):
        row = df.iloc[idx]
        missing_a = is_blank(row.get("answer_ppo_a", ""))
        missing_b = is_blank(row.get("answer_ppo_b", ""))
        if not missing_a and not missing_b:
            continue

        reject_notes: list[str] = []
        if missing_a:
            answer_a, reject_reason_a = refill_one_cell(model, tokenizer, row, "a", args.max_new_tokens)
            df.at[idx, "answer_ppo_a"] = answer_a
            updated_cells += 1
            if reject_reason_a:
                reject_notes.append(f"answer_ppo_a {reject_reason_a}")

        if missing_b:
            answer_b, reject_reason_b = refill_one_cell(model, tokenizer, row, "b", args.max_new_tokens)
            df.at[idx, "answer_ppo_b"] = answer_b
            updated_cells += 1
            if reject_reason_b:
                reject_notes.append(f"answer_ppo_b {reject_reason_b}")

        still_blank_a = is_blank(df.at[idx, "answer_ppo_a"])
        still_blank_b = is_blank(df.at[idx, "answer_ppo_b"])
        df.at[idx, "ppo_generation_note"] = "；".join(reject_notes) if (still_blank_a or still_blank_b) else ""

        print(
            f"[{idx + 1}/{total_rows}] "
            f"answer_ppo_a_len={0 if is_blank(df.at[idx, 'answer_ppo_a']) else len(str(df.at[idx, 'answer_ppo_a']))} "
            f"answer_ppo_b_len={0 if is_blank(df.at[idx, 'answer_ppo_b']) else len(str(df.at[idx, 'answer_ppo_b']))} "
            f"note={df.at[idx, 'ppo_generation_note']}"
        )

        if updated_cells % args.save_every == 0:
            save_checkpoint(df, args.checkpoint_csv)
            print(f"已保存补全 checkpoint：{args.checkpoint_csv}")

    save_checkpoint(df, args.checkpoint_csv)
    save_final_xlsx(df, args.output_xlsx)
    cleanup_model(model)

    final_blank_a = int(df["answer_ppo_a"].apply(is_blank).sum())
    final_blank_b = int(df["answer_ppo_b"].apply(is_blank).sum())
    print(f"补全完成，输出 Excel：{args.output_xlsx}")
    print(f"补全 checkpoint：{args.checkpoint_csv}")
    print(f"剩余 answer_ppo_a 空白数：{final_blank_a}")
    print(f"剩余 answer_ppo_b 空白数：{final_blank_b}")


if __name__ == "__main__":
    main()
