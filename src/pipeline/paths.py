"""
new_rlhf 全局路径配置

一、用途
1. 统一管理 new_rlhf 实验中的数据、日志、模型目录路径。
2. 避免各脚本重复硬编码路径，降低迁移和改名时的维护成本。

二、目录约定
1. ROOT_DIR: new_rlhf 根目录
2. DATA_DIR: 数据目录（包含 R1/R2/R3 及测试与验证数据）
3. LOGS_DIR: 日志目录
4. MODEL_DIR: 模型目录

三、当前默认文件
1. SFT 第一轮：R1_SFT训练集_700条.csv / R1_SFT验证集_100条.csv
2. RM 第一轮：
   - 成对偏好训练/验证：R1_RM偏好训练集_1000对.csv / R1_RM偏好验证集_250对.csv
   - 直接打分训练/验证：R1_RM直接打分训练集.csv / R1_RM直接打分验证集.csv
3. PPO prompt：PPO全局prompt池_1151题.csv
4. 过程验证集：局部验证集_150条.csv
5. 最终测试集：测试集_500条.csv

四、注意事项
1. 这里的中文文件名与目录名为当前实验真实结构，请勿随意修改。
2. 如果后续目录有调整，优先改本文件，不建议在业务脚本里直接改硬编码路径。
"""

from pathlib import Path


# 项目根目录：.../RLHF-Training/new_rlhf
ROOT_DIR = Path(__file__).resolve().parents[2]

# 三大基础目录
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "model"

# 基础模型本地缓存路径（SFT 第一轮起点）
LOCAL_MODEL_PATH = Path(r"C:\Users\kexiu\.cache\modelscope\hub\models\Qwen\Qwen1.5-1.8B-Chat")

# 数据路径：SFT
SFT_TRAIN_FILE = DATA_DIR / "02_R1_第一轮RLHF" / "R1_SFT训练集_700条.csv"
SFT_VAL_FILE = DATA_DIR / "02_R1_第一轮RLHF" / "R1_SFT验证集_100条.csv"

# 数据路径：RM（偏好对 + 直接打分）
RM_PAIR_TRAIN_FILE = DATA_DIR / "02_R1_第一轮RLHF" / "R1_RM偏好训练集_1000对.csv"
RM_PAIR_VAL_FILE = DATA_DIR / "02_R1_第一轮RLHF" / "R1_RM偏好验证集_250对.csv"
RM_POINT_TRAIN_FILE = DATA_DIR / "02_R1_第一轮RLHF" / "R1_RM直接打分训练集.csv"
RM_POINT_VAL_FILE = DATA_DIR / "02_R1_第一轮RLHF" / "R1_RM直接打分验证集.csv"

# 数据路径：PPO 与评估
PPO_PROMPT_FILE = DATA_DIR / "01_测试集与验证集" / "PPO全局prompt池_1151题.csv"
LOCAL_VAL_FILE = DATA_DIR / "01_测试集与验证集" / "局部验证集_150条.csv"
TEST_FILE = DATA_DIR / "01_测试集与验证集" / "测试集_500条.csv"

# 模型输出目录
SFT_MODEL_DIR = MODEL_DIR / "sft_v0"
RM_V1_MODEL_DIR = MODEL_DIR / "rm_v1"
RM_V2_MODEL_DIR = MODEL_DIR / "rm_v2"
RM_V3_MODEL_DIR = MODEL_DIR / "rm_v3"
PPO_V1_MODEL_DIR = MODEL_DIR / "ppo_v1"
PPO_V2_MODEL_DIR = MODEL_DIR / "ppo_v2"
PPO_V3_MODEL_DIR = MODEL_DIR / "ppo_v3"
EVAL_RESULTS_DIR = MODEL_DIR / "eval_results"

# 日志目录
SFT_LOG_DIR = LOGS_DIR / "sft"
RM_LOG_DIR = LOGS_DIR / "rm"
PPO_LOG_DIR = LOGS_DIR / "ppo"
EVAL_LOG_DIR = LOGS_DIR / "eval"


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该目录对象。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
