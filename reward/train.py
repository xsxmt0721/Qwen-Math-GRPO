from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

from reward.dataloader import BatchLoader, DataLoader
from reward.utils import build_reward_prompt, load_model


class train_config:
    device = "cuda"         # "cuda" 或 "cpu" 或 "auto"

    batch_size = 4
    sub_batch_size = 16
    min_skip_steps = 1
    max_skip_steps = 3

    neg_distribution_step = [0.25, 0.25, 0.25, 0.25]
    neg_distribution_error = [0.2, 0.2, 0.2, 0.2, 0.2]
    neg_distribution_num = 3
    latex_keywords = ["\\frac", "\\cdot", "\\times", "\\sqrt", 
                      "\\sum", "\\prod", "\\int"]

    epochs = 3

    seed = 42
    shuffle = True


def _load_env(env_path: Path) -> dict:
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


if __name__ == "__main__":
    # 加载环境变量
    env = _load_env(Path(__file__).parent.parent / ".env")
    
    # 导入数据
    loader = DataLoader(filter_think=True)
    train_set, val_set, meta = loader.load_prm_sft_datasets()
    
    train_loader = BatchLoader(train_set, config=train_config)
    val_loader = BatchLoader(val_set, config=train_config)

    # 导入模型
    model, tokenizer = load_model(model_path=env["PRM_DIR"], device=train_config.device)

    max_token_length = 0
    # 训练循环
    for epoch in range(train_config.epochs):
        train_batches = range(train_loader.batch_generator())
        train_bar = tqdm(train_batches)
        for batch_idx in train_bar:
            train_loader.sub_batch_generator(idx=batch_idx, epoch=epoch)
            while batch := train_loader.forward():
                if not batch["input"]:
                    continue
                else:
                    batch = build_reward_prompt(batch)
                    for i, sample in enumerate(batch):
                        text = tokenizer.apply_chat_template(
                            sample['messages'], 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        tokens = tokenizer.encode(text)
                        token_count = len(tokens)
                        max_token_length = max(max_token_length, token_count)
        break
    print(f"最大 token 长度: {max_token_length}") 