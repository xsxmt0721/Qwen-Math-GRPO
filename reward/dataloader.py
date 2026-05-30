from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from datasets import Dataset

from reward.utils import build_negative_samples


def _load_env(env_path: Path) -> Dict[str, str]:
	env: Dict[str, str] = {}
	if not env_path.exists():
		return env
	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if "#" in value:
			value = value.split("#", 1)[0].rstrip()
		value = value.strip().strip("\"").strip("'")
		env[key] = value
	return env


def _load_json_list(path: Path) -> List[dict]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Expected list in {path}")
	return data


def _load_index_list(path: Path) -> List[int]:
	data = _load_json_list(path)
	indices: List[int] = []
	for item in data:
		try:
			indices.append(int(item))
		except (TypeError, ValueError):
			raise ValueError(f"Invalid index in {path}: {item}")
	return indices


def _select_first_data_file(file_order: List[str]) -> str:
	for fname in file_order:
		if fname != "test.json":
			return fname
	raise ValueError("file_order contains only test.json")


def _calc_offsets(file_order: List[str], file_lengths: Dict[str, int]) -> Dict[str, int]:
	offsets: Dict[str, int] = {}
	current = 0
	for fname in file_order:
		offsets[fname] = current
		current += int(file_lengths.get(fname, 0))
	return offsets


class DataLoader:
	"""Load PRM SFT datasets based on SPLIT_META."""

	def __init__(self, repo_root: Path | None = None, filter_think: bool = False) -> None:
		self.repo_root = repo_root or Path(__file__).resolve().parents[1]
		self.split_root = self.repo_root / "logs" / "data_split"
		self.meta_path = self.split_root / "SPLIT_META.json"
		self.filter_think = filter_think
		self.env = _load_env(self.repo_root / ".env")
		data_dir_value = self.env.get("DATA_DIR")
		if not data_dir_value:
			raise ValueError("DATA_DIR is missing in .env")
		self.data_dir = Path(os.path.expandvars(os.path.expanduser(data_dir_value))).resolve()

	def _load_split_meta(self) -> dict:
		if not self.meta_path.exists():
			raise FileNotFoundError(f"SPLIT_META.json not found: {self.meta_path}")
		meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
		if "datasets" not in meta:
			raise ValueError("SPLIT_META.json missing datasets")
		return meta

	def _to_local(self, dataset_name: str, data_file: str, offset: int, length: int, indices: List[int]) -> List[int]:
		local = []
		for idx in indices:
			if idx < offset or idx >= offset + length:
				raise ValueError(
					f"Index {idx} out of range for {dataset_name}/{data_file}"
				)
			local.append(idx - offset)
		return local

	def _apply_filter_think(self, blocks: List[str]) -> List[str]:
		if not self.filter_think:
			return blocks
		try:
			idx = blocks.index("</think>")
		except ValueError:
			return blocks
		return blocks[idx + 1 :]

	@staticmethod
	def _merge_brackets(blocks: List[str]) -> List[str]:
		merged: List[str] = []
		i = 0
		while i < len(blocks):
			block = blocks[i]
			if block == "\\[":
				if i + 2 < len(blocks) and blocks[i + 2] == "\\]":
					merged.append(f"\\[\n{blocks[i + 1]}\n\\]")
					i += 3
					continue
				if i + 1 < len(blocks):
					merged.append(f"\\[\n{blocks[i + 1]}")
					i += 2
					continue
				merged.append(block)
				i += 1
				continue
			if block == "\\]":
				if merged:
					merged[-1] = f"{merged[-1]}\n\\]"
				else:
					merged.append(block)
				i += 1
				continue
			merged.append(block)
			i += 1
		return merged

	def load_prm_sft_datasets(self) -> Tuple[Dataset, Dataset, dict]:
		meta = self._load_split_meta()
		train_rows: List[dict] = []
		val_rows: List[dict] = []
		meta_summary = {"datasets": {}}

		for dataset_name, dataset_meta in tqdm(meta["datasets"].items(), desc="Loading datasets"):
			file_order = dataset_meta.get("file_order")
			file_lengths = dataset_meta.get("file_lengths")
			if not isinstance(file_order, list) or not isinstance(file_lengths, dict):
				raise ValueError(f"Invalid meta for dataset {dataset_name}")

			data_file = _select_first_data_file(file_order)
			offsets = _calc_offsets(file_order, file_lengths)
			offset = offsets.get(data_file, 0)
			length = int(file_lengths.get(data_file, 0))

			dataset_path = self.data_dir / dataset_name / "processed" / data_file
			rows = _load_json_list(dataset_path)
			if len(rows) != length:
				raise ValueError(
					f"Length mismatch for {dataset_name}/{data_file}: meta={length}, file={len(rows)}"
				)

			train_idx_path = self.split_root / "SFT_TRAIN" / f"{dataset_name}.json"
			val_idx_path = self.split_root / "SFT_VAL" / f"{dataset_name}.json"
			train_indices = _load_index_list(train_idx_path)
			val_indices = _load_index_list(val_idx_path)

			train_local = self._to_local(dataset_name, data_file, offset, length, train_indices)
			val_local = self._to_local(dataset_name, data_file, offset, length, val_indices)

			train_rows.extend([rows[i] for i in train_local])
			val_rows.extend([rows[i] for i in val_local])

			meta_summary["datasets"][dataset_name] = {
				"train": len(train_local),
				"val": len(val_local),
				"source_file": data_file,
			}

		meta_summary["train_total"] = len(train_rows)
		meta_summary["val_total"] = len(val_rows)

		rng = random.Random(42)
		rng.shuffle(train_rows)
		rng.shuffle(val_rows)

		def normalize_rows(rows: List[dict]) -> List[dict]:
			normalized: List[dict] = []
			for item in rows:
				question = item.get("question", "")
				answers = item.get("answer") or []
				split_answers = []
				for answer in answers:
					blocks = [block for block in str(answer).split("\n") if block.strip()]
					blocks = [block.strip() for block in blocks]
					blocks = self._apply_filter_think(blocks)
					blocks = self._merge_brackets(blocks)
					split_answers.append(blocks)
				normalized.append({"question": question, "answer": split_answers})
			return normalized

		train_dataset = Dataset.from_list(normalize_rows(train_rows))
		val_dataset = Dataset.from_list(normalize_rows(val_rows))

		return train_dataset, val_dataset, meta_summary



class BatchLoader:
	"""Create batch generators from PRM SFT datasets."""

	def __init__(self, datasets, config=None) -> None:
		self.datasets = datasets
		self.config = config
		self.batch_size = int(getattr(config, "batch_size", 1)) if config else 1
		self.sub_batch_size = int(getattr(config, "sub_batch_size", 1)) if config else 1
		self.batch_index: List[List[int]] = []
		self.current_batch: List[dict] = []
		seed = getattr(config, "seed", 42) if config else 42
		self.rng = random.Random(seed)

	def _require_config(self) -> None:
		if self.config is None:
			raise ValueError("BatchLoader config is required")

	def batch_generator(self) -> int:
		self.batch_index = []
		indices = list(range(len(self.datasets)))
		if getattr(self.config, "shuffle", False):
			self.rng.shuffle(indices)
		for start in range(0, len(indices), self.batch_size):
			self.batch_index.append(indices[start : start + self.batch_size])
		return len(self.batch_index)

	def sub_batch_generator(self, idx: int, epoch: int = 0) -> int:
		self._require_config()
		shuffle = getattr(self.config, "shuffle", False)
		if not self.batch_index:
			self.batch_generator()
		if idx < 0 or idx >= len(self.batch_index):
			self.current_batch = []
			return 0

		batch_rows = self.datasets.select(self.batch_index[idx])
		pos_samples: List[dict] = []
		pos_meta: List[dict] = []
		batch_answers: List[List[str]] = []

		for row in batch_rows:
			question = row.get("question", "")
			answers = row.get("answer") or []
			if not answers:
				continue
			answer_idx = epoch % len(answers)
			selected = answers[answer_idx]
			steps = [str(step).strip() for step in selected if str(step).strip()]
			if not steps:
				continue
			batch_answers.append(steps)
			batch_answer_idx = len(batch_answers) - 1

			former = ""
			for step_idx, step in enumerate(steps):
				pos_samples.append(
					{
						"idx": step_idx,
						"question": question,
						"former": former,
						"next": step,
						"label": 1,
					}
				)
				pos_meta.append(
					{
						"answer_steps": steps,
						"step_idx": step_idx,
						"batch_idx": batch_answer_idx,
					}
				)
				former = f"{former}\n{step}" if former else step

		neg_samples = build_negative_samples(pos_samples, pos_meta, batch_answers, self.config, self.rng)
		all_samples = pos_samples + neg_samples
		if shuffle:
			self.rng.shuffle(all_samples)
		self.current_batch = all_samples
		return len(self.current_batch)

	def forward(self) -> dict:
		if not self.current_batch:
			return None
		count = min(self.sub_batch_size, len(self.current_batch))
		samples = [self.current_batch.pop(0) for _ in range(count)]
		inputs = [json.dumps(sample, ensure_ascii=False) for sample in samples]
		labels = [sample["label"] for sample in samples]
		return {"input": inputs, "label": labels}

	def __len__(self) -> int:
		return len(self.batch_index)


if __name__ == "__main__":
	loader = DataLoader(filter_think=True)
	train_set, val_set, dataset_meta = loader.load_prm_sft_datasets()
	print(json.dumps(dataset_meta, ensure_ascii=False, indent=2))
	print(json.dumps(train_set.select(range(min(3, len(train_set)))).to_list(), ensure_ascii=False, indent=2))
