import argparse
import json
import os
import random


def load_env(env_path: str) -> dict:
	env = {}
	with open(env_path, "r", encoding="utf-8") as f:
		for line in f:
			raw = line.strip()
			if not raw or raw.startswith("#"):
				continue
			if "#" in raw:
				raw = raw.split("#", 1)[0].strip()
			if "=" not in raw:
				continue
			key, value = raw.split("=", 1)
			env[key.strip()] = value.strip().strip("\"").strip("'")
	return env


def load_json_list(path: str) -> list:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Expected list in {path}")
	return data


def write_index_list(path: str, indices: list) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump([str(i) for i in indices], f, ensure_ascii=False, indent=2)


def validate_indices(indices: list, total: int, label: str) -> list:
	clean = []
	for item in indices:
		try:
			value = int(item)
		except (TypeError, ValueError):
			raise ValueError(f"Invalid index in {label}: {item}")
		if value < 0 or value >= total:
			raise ValueError(f"Index out of range in {label}: {value}")
		clean.append(value)
	if len(set(clean)) != len(clean):
		raise ValueError(f"Duplicate indices found in {label}")
	return sorted(clean)


def main() -> None:
	parser = argparse.ArgumentParser(description="Split datasets into GRPO/SFT train/val and test")
	parser.add_argument(
		"--data",
		nargs="+",
		action="append",
		required=True,
		help="Dataset names under DATA_DIR (can pass multiple)",
	)
	parser.add_argument(
		"--keep-test",
		action="store_true",
		help="Keep existing test split if present under DATA_SPLIT_DIR/TEST",
	)
	args = parser.parse_args()

	dataset_names = [name for group in args.data for name in group]
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	env_path = os.path.join(repo_root, ".env")
	if not os.path.isfile(env_path):
		raise SystemExit(f".env not found at {env_path}")

	env = load_env(env_path)
	data_dir = env.get("DATA_DIR")
	split_dir = env.get("DATA_SPLIT_DIR")
	if not data_dir or not split_dir:
		raise SystemExit("DATA_DIR and DATA_SPLIT_DIR must be set in .env")

	grpo_train_rate = float(env.get("GRPO_TRAIN_RATE", "0"))
	grpo_val_rate = float(env.get("GRPO_VAL_RATE", "0"))
	sft_train_rate = float(env.get("SFT_TRAIN_RATE", "0"))
	sft_val_rate = float(env.get("SFT_VAL_RATE", "0"))
	test_rate = float(env.get("TEST_RATE", "0"))

	os.makedirs(split_dir, exist_ok=True)
	split_subdirs = {
		"GRPO_TRAIN": os.path.join(split_dir, "GRPO_TRAIN"),
		"GRPO_VAL": os.path.join(split_dir, "GRPO_VAL"),
		"SFT_TRAIN": os.path.join(split_dir, "SFT_TRAIN"),
		"SFT_VAL": os.path.join(split_dir, "SFT_VAL"),
		"TEST": os.path.join(split_dir, "TEST"),
	}
	for path in split_subdirs.values():
		os.makedirs(path, exist_ok=True)

	meta = {
		"rates": {
			"grpo_train": grpo_train_rate,
			"grpo_val": grpo_val_rate,
			"sft_train": sft_train_rate,
			"sft_val": sft_val_rate,
			"test": test_rate,
		},
		"datasets": {},
		"keep_test": args.keep_test,
		"seed": 42,
	}

	rng = random.Random(42)

	for dataset_name in dataset_names:
		dataset_dir = os.path.join(data_dir, dataset_name)
		processed_dir = os.path.join(dataset_dir, "processed")
		if not os.path.isdir(processed_dir):
			continue

		json_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
		if not json_files:
			continue
		json_files.sort()

		file_lengths = {}
		total = 0
		for fname in json_files:
			path = os.path.join(processed_dir, fname)
			data = load_json_list(path)
			file_lengths[fname] = len(data)
			total += len(data)

		offsets = {}
		current = 0
		for fname in json_files:
			offsets[fname] = current
			current += file_lengths[fname]

		test_indices = []
		test_source = None
		existing_test_path = os.path.join(split_subdirs["TEST"], f"{dataset_name}.json")
		if args.keep_test and os.path.isfile(existing_test_path):
			existing = load_json_list(existing_test_path)
			test_indices = validate_indices(existing, total, f"TEST/{dataset_name}.json")
			test_source = "keep-test"
		elif "test.json" in json_files:
			start = offsets["test.json"]
			count = file_lengths["test.json"]
			test_indices = list(range(start, start + count))
			test_source = "test.json"
		else:
			all_indices = list(range(total))
			rng.shuffle(all_indices)
			test_count = int(total * test_rate)
			if total > 0 and test_rate > 0 and test_count == 0:
				test_count = 1
			test_indices = sorted(all_indices[:test_count])
			test_source = "rate"

		test_set = set(test_indices)
		remaining = [idx for idx in range(total) if idx not in test_set]
		rng.shuffle(remaining)

		remaining_count = len(remaining)
		grpo_train_count = int(remaining_count * grpo_train_rate)
		grpo_val_count = int(remaining_count * grpo_val_rate)
		sft_train_count = int(remaining_count * sft_train_rate)
		sft_val_count = remaining_count - grpo_train_count - grpo_val_count - sft_train_count

		grpo_train = remaining[:grpo_train_count]
		grpo_val = remaining[grpo_train_count : grpo_train_count + grpo_val_count]
		sft_train = remaining[
			grpo_train_count + grpo_val_count : grpo_train_count + grpo_val_count + sft_train_count
		]
		sft_val = remaining[
			grpo_train_count + grpo_val_count + sft_train_count :
		]

		write_index_list(os.path.join(split_subdirs["TEST"], f"{dataset_name}.json"), test_indices)
		write_index_list(os.path.join(split_subdirs["GRPO_TRAIN"], f"{dataset_name}.json"), grpo_train)
		write_index_list(os.path.join(split_subdirs["GRPO_VAL"], f"{dataset_name}.json"), grpo_val)
		write_index_list(os.path.join(split_subdirs["SFT_TRAIN"], f"{dataset_name}.json"), sft_train)
		write_index_list(os.path.join(split_subdirs["SFT_VAL"], f"{dataset_name}.json"), sft_val)

		meta["datasets"][dataset_name] = {
			"total": total,
			"test": len(test_indices),
			"remaining": remaining_count,
			"grpo_train": len(grpo_train),
			"grpo_val": len(grpo_val),
			"sft_train": len(sft_train),
			"sft_val": len(sft_val),
			"test_source": test_source,
			"file_order": json_files,
			"file_lengths": file_lengths,
		}

	meta_path = os.path.join(split_dir, "SPLIT_META.json")
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	main()
