import json
import os
import re
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.dataloader import DataLoader
from utils.utils import extract_boxed_answer

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

os.environ["WANDB_MODE"] = "offline"

env = _load_env(Path(__file__).parent.parent / ".env")

DATA_DIR: str = env.get("DATA_DIR", "/data")
SPLIT_DIR: str = env.get("DATA_SPLIT_DIR", "/data_split")

MODEL_PATH: str = env.get("BASEMODEL_DIR", "")
SAVE_DIR: str = env.get("REF_SAVE", "")
SAVE_PATH = os.path.join(SAVE_DIR, "checkpoint-138")

class eval_config:
    log_dir = "/workspace/logs/ref_logs/ref-1.5b"
    
    cuda_use: bool = True
    eval_batch_size: int = 16
    max_length = 1024
    max_new_tokens = 256

    seed = 42
    shuffle: bool = False
    transform: bool = True
    transform_all_ans: bool = False


def _get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

if __name__ == "__main__":
    val_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="SFT_VAL",
        batch_size=eval_config.eval_batch_size,
        shuffle=eval_config.shuffle,
        transform=eval_config.transform,
        transform_all_ans=eval_config.transform_all_ans,
        include_answer=False,
        seed=eval_config.seed,
    )
    
    val_dataset = Dataset.from_list(val_loader.data)
    val_answers = val_loader.ans

    if not MODEL_PATH or not SAVE_PATH:
        raise ValueError("Missing BASEMODEL_DIR or REF_SAVE in .env")

    compute_dtype = _get_compute_dtype()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if eval_config.cuda_use and torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, SAVE_PATH)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    def _build_prompt(example: dict) -> str:
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            messages = messages[0]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    output_dir = Path(eval_config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"

    prompts_all = [_build_prompt(val_dataset[i]) for i in range(len(val_dataset))]

    flush_every = 25
    buffer = []
    correct = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(val_dataset), eval_config.eval_batch_size)):
            end = min(start + eval_config.eval_batch_size, len(val_dataset))
            prompts = prompts_all[start:end]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=eval_config.max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=eval_config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for offset, pred_text in enumerate(pred_texts):
                idx = start + offset
                pred_ans = extract_boxed_answer(pred_text) or pred_text.strip()
                gold_text = val_answers[idx] if idx < len(val_answers) else ""
                gold_ans = extract_boxed_answer(gold_text) or str(gold_text).strip()
                is_correct = pred_ans == gold_ans
                if is_correct:
                    correct += 1
                total += 1
                record = {
                    "index": idx,
                    "prediction": pred_text,
                    "predicted_answer": pred_ans,
                    "gold_answer": gold_ans,
                    "is_correct": is_correct,
                }
                buffer.append(json.dumps(record, ensure_ascii=True))
            if len(buffer) >= flush_every:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")

    accuracy = correct / total if total else 0.0
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Saved results to: {output_path}")
    print(f"Saved summary to: {summary_path}")