"""
new_rlhf RM 人工抽检脚本

一、脚本用途
1. 对已经训练完成的 RM 做人工 sanity check，避免只看总指标却忽略个体样本异常。
2. 抽样检查成对偏好数据中，RM 是否稳定地给 chosen 高于 rejected 的分数。
3. 额外构造“模型生成回答 vs 明显较差模板回答”的对照样本，帮助人工判断 RM 的偏好方向是否合理。

二、默认输入
1. RM 底模：model/sft_v0/merged
2. RM LoRA：model/rm_v1/final_lora
3. 成对偏好数据：默认使用 R1_RM偏好验证集_250对.csv
4. 生成问答来源：默认使用 R1_RM直接打分验证集.csv 中的问题字段

三、默认输出
1. 成对偏好抽检结果：
   logs/rm/rm_pair_manual_check.csv
2. 生成回答抽检结果：
   logs/rm/rm_generated_manual_check.csv

四、使用说明
1. 该脚本不是训练脚本，而是训练完成后的人工复核工具。
2. 输出 CSV 里预留了 human_check 和 notes 两列，方便你手工填写检查结论。
3. 如果只想检查成对偏好结果，可加 --skip_generation 跳过第二部分。
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.paths import RM_LOG_DIR, RM_V1_MODEL_DIR, SFT_MODEL_DIR, ensure_dir
from rm_train import PAIR_TRAIN, PAIR_VAL, POINT_VAL, build_prompt, read_csv_required


BAD_ANSWERS = [
    "这个问题需要结合实际情况综合判断。",
    "建议按照相关制度执行，并做好记录。",
    "目前无法准确判断，建议进一步核实。",
    "首先加强管理，其次提高意识，最后持续改进。",
    "设备管理应遵循安全、稳定、经济的原则。",
]


def parse_args() -> argparse.Namespace:
    """解析人工抽检参数。"""
    parser = argparse.ArgumentParser(description="RM 人工抽检：偏好对样本 + 生成回答样本")
    parser.add_argument("--base_model_path", default=str(SFT_MODEL_DIR / "merged"))
    parser.add_argument("--rm_lora_path", default=str(RM_V1_MODEL_DIR / "final_lora"))
    parser.add_argument("--pair_source", choices=["val", "train"], default="val")
    parser.add_argument("--pair_n", type=int, default=20)
    parser.add_argument("--generated_n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--pair_output", default=str(RM_LOG_DIR / "rm_pair_manual_check.csv"))
    parser.add_argument("--generated_output", default=str(RM_LOG_DIR / "rm_generated_manual_check.csv"))
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rm(args: argparse.Namespace):
    """加载 RM 打分器。"""
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.rm_lora_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=dtype,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, args.rm_lora_path)
    model.to(args.device)
    model.eval()
    return tokenizer, model


def load_sft_generator(args: argparse.Namespace):
    """加载 SFT 模型，用来生成抽检答案。"""
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(args.device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_answer(tokenizer, model, device: str, question: str, answer: str, max_length: int) -> float:
    """给单条回答打分。"""
    encoded = tokenizer(
        build_prompt(question, answer),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return float(model(**encoded).logits.float().view(-1).item())


def build_generation_prompt(question: str) -> str:
    """统一生成阶段 prompt。"""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_answer(tokenizer, model, device: str, question: str, max_new_tokens: int) -> str:
    """生成单条回答。"""
    encoded = tokenizer(build_generation_prompt(question), return_tensors="pt").to(device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    marker = "<|im_start|>assistant\n"
    answer = text.split(marker, 1)[1] if marker in text else text
    return answer.replace("<|im_end|>", "").strip()


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出抽检结果 CSV。"""
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_check(args: argparse.Namespace, rm_tokenizer, rm_model) -> None:
    """检查成对偏好样本中，RM 是否更偏向 chosen。"""
    pair_path = PAIR_VAL if args.pair_source == "val" else PAIR_TRAIN
    pair_df = read_csv_required(pair_path, ["question", "chosen", "rejected"])
    sample_df = pair_df.sample(n=min(args.pair_n, len(pair_df)), random_state=args.seed)

    rows = []
    for idx, row in sample_df.reset_index(drop=True).iterrows():
        chosen_score = score_answer(rm_tokenizer, rm_model, args.device, row["question"], row["chosen"], args.max_length)
        rejected_score = score_answer(
            rm_tokenizer, rm_model, args.device, row["question"], row["rejected"], args.max_length
        )
        rows.append(
            {
                "id": idx + 1,
                "source": args.pair_source,
                "question": row["question"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "margin": chosen_score - rejected_score,
                "model_prefers_chosen": chosen_score > rejected_score,
                "human_check": "",
                "notes": "",
            }
        )

    write_csv(
        args.pair_output,
        rows,
        [
            "id",
            "source",
            "question",
            "chosen",
            "rejected",
            "chosen_score",
            "rejected_score",
            "margin",
            "model_prefers_chosen",
            "human_check",
            "notes",
        ],
    )
    print(f"Pair manual check saved to: {args.pair_output}")


def generated_check(args: argparse.Namespace, rm_tokenizer, rm_model) -> None:
    """构造生成回答与差回答的对照样本，方便人工看 RM 偏好。"""
    sft_tokenizer, sft_model = load_sft_generator(args)
    question_df = read_csv_required(POINT_VAL, ["question", "answer", "final_score"])
    questions = question_df["question"].dropna().drop_duplicates().tolist()
    random.Random(args.seed).shuffle(questions)
    questions = questions[: args.generated_n]

    rows = []
    row_id = 1
    for question in questions:
        generated = generate_answer(sft_tokenizer, sft_model, args.device, question, args.max_new_tokens)
        candidates = [("sft_generated", generated)]
        candidates.extend((f"bad_template_{i + 1}", answer) for i, answer in enumerate(BAD_ANSWERS))

        for answer_type, answer in candidates:
            rows.append(
                {
                    "id": row_id,
                    "question": question,
                    "answer_type": answer_type,
                    "answer": answer,
                    "reward_score": score_answer(rm_tokenizer, rm_model, args.device, question, answer, args.max_length),
                    "human_check": "",
                    "notes": "",
                }
            )
            row_id += 1

    write_csv(
        args.generated_output,
        rows,
        ["id", "question", "answer_type", "answer", "reward_score", "human_check", "notes"],
    )
    print(f"Generated-answer manual check saved to: {args.generated_output}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rm_tokenizer, rm_model = load_rm(args)
    pair_check(args, rm_tokenizer, rm_model)

    if not args.skip_generation:
        generated_check(args, rm_tokenizer, rm_model)


if __name__ == "__main__":
    main()
