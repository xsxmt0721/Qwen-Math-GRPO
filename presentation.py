"""
presentation.py - 加载训练完成的模型，导入测试集前 x 条数据，在终端打印数据和模型反馈。

输出格式: {"message": "", "completion": "", "answer": "", "true_answer": ""}
"""

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.dataloader import DataLoader
from utils.utils import extract_boxed_answer


# ═══════════════════════════════════════════════════════════════
# 工具函数: 加载 .env
# ═══════════════════════════════════════════════════════════════

def _load_env(env_path: Path) -> dict:
    """从 .env 文件解析键值对（与 train.py / test.py 保持一致）。"""
    env = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


# ═══════════════════════════════════════════════════════════════
# 可调参数
# ═══════════════════════════════════════════════════════════════

class presentation_config:
    num_samples: int = 5            # 展示前 x 条测试数据
    max_new_tokens: int = 512       # 模型生成的最大 token 数
    max_prompt_length: int = 1024   # 输入 prompt 的最大 token 数
    temperature: float = 1.0        # 生成温度
    seed: int = 42


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. 加载 .env 配置 ──
    env = _load_env(Path(__file__).parent / ".env")

    DATA_DIR: str = env.get("DATA_DIR", "/data")
    SPLIT_DIR: str = env.get("DATA_SPLIT_DIR", "/workspace/logs/data_split")
    MODEL_PATH: str = env.get("BASEMODEL_DIR", "")
    GRPO_PATH: str = env.get("GRPO_SAVE", "")
    CHECKPOINT_DIR: str = env.get("CHECKPOINT_DIR", "checkpoint-200")
    GRPO_SAVE: str = os.path.join(GRPO_PATH, CHECKPOINT_DIR)

    cfg = presentation_config()

    print("=" * 60)
    print("GRPO 模型展示程序")
    print(f"  基座模型    : {MODEL_PATH}")
    print(f"  GRPO LoRA   : {GRPO_SAVE}")
    print(f"  数据目录    : {DATA_DIR}")
    print(f"  划分目录    : {SPLIT_DIR}")
    print(f"  展示条数    : {cfg.num_samples}")
    print("=" * 60)

    # ── 2. 加载测试集 ──
    print("\n[1/3] 加载测试数据 ...")
    test_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="TEST",
        batch_size=1,
        shuffle=False,
        transform=True,
        transform_all_ans=False,
        include_answer=False,
        seed=cfg.seed,
    )
    total = test_loader.num_data
    n_show = min(cfg.num_samples, total)
    print(f"  测试集共 {total} 条，将展示前 {n_show} 条")

    if n_show == 0:
        print("  测试集为空，退出。")
        return

    # ── 3. 加载模型 ──
    print("\n[2/3] 加载模型 ...")

    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else "cpu"

    quant_config = None
    if use_cuda:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    # 加载 GRPO 训练后的 LoRA 权重
    if GRPO_SAVE and Path(GRPO_SAVE).exists():
        model = PeftModel.from_pretrained(model, GRPO_SAVE)
        print(f"  已加载 GRPO LoRA: {GRPO_SAVE}")
    else:
        print(f"  ⚠ 未找到 GRPO checkpoint ({GRPO_SAVE})，将使用基座模型")

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    print(f"  模型设备: {device}")

    # ── 4. 逐条生成并打印 ──
    print(f"\n[3/3] 开始生成，共 {n_show} 条 ...\n")

    for i in range(n_show):
        item = test_loader.data[i]          # {"messages": [...]}
        true_answer = test_loader.ans[i]    # ground truth final_answer

        # 构建 prompt（system + user + generation prompt）
        messages = item.get("messages", [])
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize & generate
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )

        # 仅提取模型新生成的部分作为 completion
        completion_ids = generated[0][input_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # 从 completion 中提取 \boxed{...} 答案
        answer = extract_boxed_answer(completion)

        # 按指定格式打印
        result = {
            "message": prompt_text,
            "completion": completion,
            "answer": answer,
            "true_answer": true_answer,
        }

        print(f"--- 第 {i + 1}/{n_show} 条 ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

    print("展示完成。")


if __name__ == "__main__":
    main()
