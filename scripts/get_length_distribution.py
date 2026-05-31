# 这个脚本可以返回当前数据集的实际token长度分布，用于合理地设置训练时的max_length参数。

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from reward.dataloader import BatchLoader, DataLoader
from reward.utils import build_reward_prompt


class length_config:
	batch_size = 4
	sub_batch_size = 4
	max_length = 4096
	seed = 42
	shuffle = True
	buckets = [256, 512, 1024, 1536, 2048]


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
			}
		)
	return features


def _summarize_lengths(lengths: list[int]) -> list[tuple[int, float]]:
	buckets = length_config.buckets
	total = max(len(lengths), 1)
	results = []
	for bound in buckets:
		count = sum(1 for value in lengths if value < bound)
		results.append((bound, count / total))
	return results


def _collect_lengths(loader: BatchLoader, tokenizer: AutoTokenizer, max_length: int) -> list[int]:
	lengths: list[int] = []
	batches = range(loader.batch_generator())
	for batch_idx in tqdm(batches):
		loader.sub_batch_generator(idx=batch_idx, epoch=0)
		while batch := loader.forward():
			if not batch["input"]:
				continue
			items = build_reward_prompt(batch)
			features = _build_inputs_and_labels(items, tokenizer, max_length)
			for feat in features:
				lengths.append(len(feat["input_ids"]))
	return lengths


if __name__ == "__main__":
	env = _load_env(Path(__file__).parent.parent / ".env")
	_set_seed(length_config.seed)

	loader = DataLoader(filter_think=True)
	train_set, val_set, _ = loader.load_prm_sft_datasets()

	train_loader = BatchLoader(train_set, config=length_config)
	val_loader = BatchLoader(val_set, config=length_config)

	tokenizer = AutoTokenizer.from_pretrained(env["PRM_DIR"], trust_remote_code=True)

	train_lengths = _collect_lengths(train_loader, tokenizer, length_config.max_length)
	val_lengths = _collect_lengths(val_loader, tokenizer, length_config.max_length)

	for name, lengths in ("train", train_lengths), ("val", val_lengths):
		summary = _summarize_lengths(lengths)
		print(f"{name} samples: {len(lengths)}")
		for bound, ratio in summary:
			print(f"  <{bound}: {ratio:.4%}")
