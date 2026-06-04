from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

from utils.utils import build_prompt


def _load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _load_index_list(path: Path) -> List[int]:
	raw = _load_json(path)
	if not isinstance(raw, list):
		raise ValueError(f"Expected index list in {path}")
	indices: List[int] = []
	for item in raw:
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


def _calc_offset(file_order: List[str], file_lengths: Dict[str, int], target: str) -> int:
	offset = 0
	for fname in file_order:
		if fname == target:
			return offset
		offset += int(file_lengths.get(fname, 0))
	return offset


class DataLoader:
	def __init__(
		self,
		data_dir: str,
		split_dir: str,
		data_type: str,
		batch_size: int,
		shuffle: bool = True,
		transform: bool = True,
		transform_all_ans: bool = False,
		seed: int = 42,
	) -> None:
		self.data_dir = Path(data_dir)
		self.split_dir = Path(split_dir)
		self.data_type = data_type
		self.batch_size = int(batch_size)
		self.seed = seed
		self.shuffle = shuffle
		self.transform_all_ans = transform_all_ans

		self.raw_data: List[dict] = []
		self.data: List[dict] = []
		self.ans: List[str] = []
		self.questions: List[str] = []
		self.num_data = 0
		self.num_batches = 0

		self._load_data()
		if transform:
			self.transform()
		else:
			self.data = list(self.raw_data)
			self.questions = [str(item.get("question", "")) for item in self.raw_data]
			self.ans = [str(item.get("final_answer", "")) for item in self.raw_data]
			self.num_data = len(self.data)
			self.num_batches = ceil(self.num_data / self.batch_size) if self.batch_size > 0 else 0

	def _load_data(self) -> None:
		meta_path = self.split_dir / "SPLIT_META.json"
		meta = _load_json(meta_path)
		datasets = meta.get("datasets")
		if not isinstance(datasets, dict):
			raise ValueError("SPLIT_META.json missing datasets")

		merged: List[dict] = []
		for dataset_name, dataset_meta in datasets.items():
			file_order = dataset_meta.get("file_order")
			file_lengths = dataset_meta.get("file_lengths")
			if not isinstance(file_order, list) or not isinstance(file_lengths, dict):
				raise ValueError(f"Invalid meta for dataset {dataset_name}")

			data_file = _select_first_data_file(file_order)
			offset = _calc_offset(file_order, file_lengths, data_file)

			index_path = self.split_dir / self.data_type / f"{dataset_name}.json"
			indices = _load_index_list(index_path)

			data_path = self.data_dir / dataset_name / "processed" / data_file
			rows = _load_json(data_path)
			if not isinstance(rows, list):
				raise ValueError(f"Expected list in {data_path}")

			for idx in indices:
				local_idx = idx - offset
				if local_idx < 0 or local_idx >= len(rows):
					raise ValueError(
						f"Index {idx} out of range for {dataset_name}/{data_file}"
					)
				merged.append(rows[local_idx])

		self.raw_data = merged

	def transform(self) -> None:
		num_data, data_list, questions, final_answers = build_prompt(
			self.raw_data,
			shuffle=self.shuffle,
			all_ans=self.transform_all_ans,
			seed=self.seed,
		)
		self.data = data_list
		self.questions = questions
		self.ans = final_answers
		self.num_data = num_data
		self.num_batches = ceil(self.num_data / self.batch_size) if self.batch_size > 0 else 0

	def __getitem__(self, idx: int) -> Any:
		if idx < 0 or idx >= self.num_batches:
			raise IndexError("Batch index out of range")
		start = idx * self.batch_size
		end = min((idx + 1) * self.batch_size, self.num_data)
		batch_data = self.data[start:end]
		batch_ans = self.ans[start:end]
		return {
			"data": batch_data,
			"ans": batch_ans,
			"num": len(batch_data),
		}

	def __len__(self) -> int:
		return self.num_data

	def __batches__(self) -> int:
		return self.num_batches
