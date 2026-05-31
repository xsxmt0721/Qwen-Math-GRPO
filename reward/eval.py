from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward.dataloader import BatchLoader, DataLoader
from reward.utils import build_reward_prompt

eval_dir = "/workspace/logs/prm_logs/v1"
max_new_tokens = 1

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


def _get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _build_prompt_label_pairs(
    items: list[dict],
    tokenizer: AutoTokenizer,
) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in items:
        messages = item.get("messages")
        if not messages or messages[-1].get("role") != "assistant":
            continue
        label_text = str(messages[-1].get("content", "")).strip()
        if not label_text:
            continue
        label_char = label_text[-1]
        if label_char not in ("0", "1"):
            continue
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        pairs.append((prompt, int(label_char)))
    return pairs


def _update_confusion(stats: dict[str, int], pred: int, label: int) -> None:
    if pred == 1 and label == 1:
        stats["tp"] += 1
    elif pred == 1 and label == 0:
        stats["fp"] += 1
    elif pred == 0 and label == 1:
        stats["fn"] += 1
    else:
        stats["tn"] += 1


def _compute_metrics(stats: dict[str, int]) -> dict[str, float]:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def _load_train_config(log_dir: str) -> dict:
    path = Path(log_dir) / "train_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
	env = _load_env(Path(__file__).parent.parent / ".env")
	base_model_path = env.get("PRM_DIR", "")
	lora_path = env.get("PRM_SAVE", "")
	if not base_model_path or not lora_path:
		raise ValueError("Missing PRM_DIR or PRM_SAVE in .env")

	train_config_payload = _load_train_config(eval_dir)
	if not train_config_payload:
		raise ValueError("train_config.json not found in eval_dir")

	eval_config = SimpleNamespace(**train_config_payload)
	eval_config.log_dir = eval_dir

	loader = DataLoader(filter_think=True)
	_, val_set, _ = loader.load_prm_sft_datasets()
	val_loader = BatchLoader(val_set, config=eval_config)

	compute_dtype = _get_compute_dtype()
	quant_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=compute_dtype,
	)

	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
	tokenizer.padding_side = "left"
	model = AutoModelForCausalLM.from_pretrained(
		base_model_path,
		device_map=eval_config.device,
		quantization_config=quant_config,
		torch_dtype=compute_dtype,
		trust_remote_code=True,
	)
	model = PeftModel.from_pretrained(model, lora_path)
	model.eval()

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model.config.pad_token_id = tokenizer.pad_token_id

	stats = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
	total = 0
	batches = tqdm(range(val_loader.batch_generator()))
	for batch_idx in batches:
		val_loader.sub_batch_generator(idx=batch_idx, epoch=0)
		while batch := val_loader.forward():
			if not batch["input"]:
				continue
			items = build_reward_prompt(batch)
			pairs = _build_prompt_label_pairs(items, tokenizer)
			if not pairs:
				continue
			prompts = [prompt for prompt, _ in pairs]
			labels = [label for _, label in pairs]

			encoded = tokenizer(
				prompts,
				padding=True,
				truncation=True,
				max_length=eval_config.max_length,
				return_tensors="pt",
			)
			device = next(model.parameters()).device
			encoded = {k: v.to(device) for k, v in encoded.items()}

			with torch.no_grad():
				outputs = model.generate(
					**encoded,
					max_new_tokens=max_new_tokens,
					do_sample=False,
					temperature=1.0,
                    top_p=1.0,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
				)

			decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
			for text, label in zip(decoded, labels):
				if not text:
					continue
				pred_char = text[-1]
				if pred_char not in ("0", "1"):
					continue
				pred = int(pred_char)
				_update_confusion(stats, pred, label)
				total += 1

	metrics = _compute_metrics(stats)
	payload = {"total": total, **metrics}
	os.makedirs(eval_config.log_dir, exist_ok=True)
	output_path = Path(eval_config.log_dir) / "eval_metrics.json"
	output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(payload, ensure_ascii=False, indent=2))
