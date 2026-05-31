from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward.dataloader import BatchLoader, DataLoader
from reward.utils import build_reward_prompt


class train_config:
    log_dir = "/workspace/logs/prm_logs/v1"
    
    device = "cuda"         # "cuda" 或 "cpu" 或 "auto"

    batch_size = 4
    sub_batch_size = 4
    min_skip_steps = 1
    max_skip_steps = 3

    neg_distribution_step = [0.4, 0.2, 0.2, 0.2]
    neg_distribution_error = [0.2, 0.2, 0.2, 0.2, 0.2]
    neg_distribution_num = 3
    latex_keywords_basic = [
        "\\frac", "\\sqrt", "\\cdot", "\\times", "\\div", "\\pm", "\\mp", 
        "\\sum", "\\prod", "\\int", "\\oint", "\\partial", "\\nabla",
        "\\infty", "\\exp", "\\ln", "\\log", "\\sin", "\\cos", "\\tan",
        "\\cot", "\\sec", "\\csc", "\\arcsin", "\\arccos", "\\arctan",
        "\\sinh", "\\cosh", "\\tanh", "\\coth",
        "\\hat", "\\bar", "\\tilde", "\\vec", "\\dot", "\\ddot", 
        "\\overline", "\\underline", "\\widehat", "\\widetilde",
        "\\left", "\\right", "\\langle", "\\rangle", 
        "\\lfloor", "\\rfloor", "\\lceil", "\\rceil",
        "\\forall", "\\exists", "\\in", "\\notin", "\\subset", "\\subseteq", 
        "\\cup", "\\cap", "\\setminus", "\\rightarrow", "\\Rightarrow", 
        "\\implies", "\\iff", "\\neq", "\\leq", "\\geq", "\\approx"
    ]

    epochs = 3

    learning_rate = 2e-4
    weight_decay = 0.0
    max_grad_norm = 1.0
    grad_accum_steps = 1
    max_length = 1024

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    log_step = 10

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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _validate_sft_batch(items: list[dict]) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("Empty SFT batch after build_reward_prompt")
    sample = items[0]
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("SFT sample missing messages list")
    roles = [m.get("role") for m in messages]
    if roles[-1] != "assistant":
        raise ValueError("Last message must be assistant for SFT labels")


def _build_inputs_and_labels(
    items: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> list[dict]:
    features: list[dict] = []
    for item in items:
        messages = item.get("messages")
        if not messages or messages[-1].get("role") != "assistant":
            continue
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(prompt_ids) > len(full_ids):
            prompt_ids = full_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        input_ids = full_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        features.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }
        )
    return features


def _pad_batch(features: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids = []
    labels = []
    attention_mask = []
    for feat in features:
        pad_len = max_len - len(feat["input_ids"])
        input_ids.append(feat["input_ids"] + [pad_token_id] * pad_len)
        labels.append(feat["labels"] + [-100] * pad_len)
        attention_mask.append(feat["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _save_epoch_metrics(log_dir: str, epoch: int, payload: dict) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = Path(log_dir) / f"epoch_{epoch:03d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_train_config(log_dir: str) -> None:
    payload = {
        key: value
        for key, value in train_config.__dict__.items()
        if not key.startswith("__")
    }
    path = Path(log_dir) / "train_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_epoch_metrics(log_dir: str, history: list[dict]) -> None:
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train_loss")
    ax.plot(epochs, val_losses, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()

    fig.tight_layout()
    output_path = Path(log_dir) / "metrics.png"
    fig.savefig(output_path)
    plt.close(fig)


def _plot_train_losses(log_dir: str, losses: list[float]) -> None:
    if not losses:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, label="train_loss")
    ax.set_xlabel("log_step")
    ax.set_ylabel("loss")
    ax.legend()

    fig.tight_layout()
    output_path = Path(log_dir) / "train_loss.png"
    fig.savefig(output_path)
    plt.close(fig)


def _get_gpu_mem_pct() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return (used / total) * 100.0


if __name__ == "__main__":
    # 加载环境变量
    env = _load_env(Path(__file__).parent.parent / ".env")

    _set_seed(train_config.seed)
    
    # 导入数据
    loader = DataLoader(filter_think=True)
    train_set, val_set, meta = loader.load_prm_sft_datasets()
    
    train_loader = BatchLoader(train_set, config=train_config)
    val_loader = BatchLoader(val_set, config=train_config)

    # 导入模型
    compute_dtype = _get_compute_dtype()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(env["PRM_DIR"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        env["PRM_DIR"],
        device_map=train_config.device,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    os.makedirs(train_config.log_dir, exist_ok=True)
    _save_train_config(train_config.log_dir)
    history: list[dict] = []
    train_log_losses: list[float] = []
    best_val_loss = float("inf")
    save_path = env["PRM_SAVE"]

    # 训练循环
    loss_sum = 0.0 
    for epoch in range(train_config.epochs):
        train_loss_sum = 0.0
        train_loss_count = 0
        log_loss_sum = 0.0
        log_loss_count = 0
        log_losses_epoch: list[float] = []
        step_in_epoch = 0
        train_batches = range(train_loader.batch_generator())
        train_bar = tqdm(train_batches)
        for batch_idx in train_bar:
            train_loader.sub_batch_generator(idx=batch_idx, epoch=epoch)
            while batch := train_loader.forward():
                if not batch["input"]:
                    continue
                batch = build_reward_prompt(batch)
                _validate_sft_batch(batch)
                features = _build_inputs_and_labels(
                    batch,
                    tokenizer,
                    train_config.max_length,
                )
                if not features:
                    continue

                model.train()
                packed = _pad_batch(features, tokenizer.pad_token_id)
                device = next(model.parameters()).device
                packed = {k: v.to(device) for k, v in packed.items()}
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=compute_dtype,
                    enabled=torch.cuda.is_available(),
                ):
                    outputs = model(**packed)
                    loss = outputs.loss / train_config.grad_accum_steps
                    loss_value = float(outputs.loss)
                    train_loss_sum += loss_value
                    train_loss_count += 1
                    loss_sum += float(loss)
                    loss_avg = train_loss_sum / max(train_loss_count, 1)
                    mem_pct = _get_gpu_mem_pct()
                    train_bar.set_description(
                        f"Epoch {epoch + 1} Loss: {loss_avg:.4f} GPU: {mem_pct:.1f}%"
                    )
                loss.backward()
                step_in_epoch += 1
                log_loss_sum += loss_value
                log_loss_count += 1
                if log_loss_count >= train_config.log_step:
                    log_losses_epoch.append(log_loss_sum / log_loss_count)
                    log_loss_sum = 0.0
                    log_loss_count = 0

                if step_in_epoch % train_config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

        if step_in_epoch % train_config.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        if log_loss_count > 0:
            log_losses_epoch.append(log_loss_sum / log_loss_count)

        # 验证循环
        model.eval()
        val_losses = []
        val_loss_sum = 0.0
        val_loss_count = 0
        val_batches = range(val_loader.batch_generator())
        val_bar = tqdm(val_batches)
        for batch_idx in val_bar:
            val_loader.sub_batch_generator(idx=batch_idx, epoch=epoch)
            while batch := val_loader.forward():
                if not batch["input"]:
                    continue
                batch = build_reward_prompt(batch)
                features = _build_inputs_and_labels(
                    batch,
                    tokenizer,
                    train_config.max_length,
                )
                if not features:
                    continue
                packed = _pad_batch(features, tokenizer.pad_token_id)
                device = next(model.parameters()).device
                packed = {k: v.to(device) for k, v in packed.items()}
                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=compute_dtype,
                        enabled=torch.cuda.is_available(),
                    ):
                        outputs = model(**packed)
                        val_loss_value = float(outputs.loss)
                        val_losses.append(val_loss_value)
                        val_loss_sum += val_loss_value
                        val_loss_count += 1
                        val_loss_avg = val_loss_sum / max(val_loss_count, 1)
                        val_bar.set_description(f"Epoch {epoch + 1} Val Loss: {val_loss_avg:.4f}")

        if val_losses:
            avg_val_loss = val_loss_sum / max(val_loss_count, 1)
            print(f"Epoch {epoch + 1}: val_loss={avg_val_loss:.4f}")

        avg_train_loss = train_loss_sum / max(train_loss_count, 1)
        epoch_payload = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss if val_losses else 0.0,
        }
        _save_epoch_metrics(
            train_config.log_dir,
            epoch + 1,
            {
                "epoch": epoch + 1,
                "log_step": train_config.log_step,
                "losses": log_losses_epoch,
            },
        )
        history.append(epoch_payload)
        train_log_losses.extend(log_losses_epoch)

        current_val_loss = epoch_payload["val_loss"]
        if val_losses and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved best model at epoch {epoch + 1}: val_loss={best_val_loss:.4f}")

    # 保存微调权重
    if history:
        best_epoch = min(history, key=lambda item: item["val_loss"])
        best_epoch_idx = best_epoch["epoch"]
        print(f"Best epoch: {best_epoch_idx} val_loss={best_epoch['val_loss']:.4f}")

    summary_path = Path(train_config.log_dir) / "metrics_summary.json"
    summary_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        _plot_epoch_metrics(train_config.log_dir, history)
    _plot_train_losses(train_config.log_dir, train_log_losses)