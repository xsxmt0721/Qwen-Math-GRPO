from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reward.model import load_prm_model_and_tokenizer


def _resolve_pretrained_dir(value: str) -> Path:
	if not value:
		raise ValueError("Missing --pretrained-dir")
	return Path(value).expanduser().resolve()


def main() -> None:
	parser = argparse.ArgumentParser(description="Load PRM model weights and report metadata")
	parser.add_argument(
		"--pretrained-dir",
		required=True,
		help="Local model directory on host, e.g. ~/MathRL/Models/PRM",
	)
	parser.add_argument(
		"--model-id",
		default="Qwen/Qwen2.5-1.5B",
		help="Hugging Face model id for tokenizer/model config",
	)
	parser.add_argument(
		"--torch-dtype",
		choices=["float16", "bf16", "float32"],
		default="float16",
		help="Torch dtype for model weights to reduce VRAM usage",
	)
	args = parser.parse_args()
	if not torch.cuda.is_available():
		raise SystemExit("CUDA is not available. Please check GPU driver/CUDA setup.")

	model_id = args.model_id
	pretrained_dir = _resolve_pretrained_dir(args.pretrained_dir)
	if args.torch_dtype == "bf16":
		model_dtype = torch.bfloat16
	elif args.torch_dtype == "float32":
		model_dtype = torch.float32
	else:
		model_dtype = torch.float16
	model, tokenizer = load_prm_model_and_tokenizer(
		pretrained_dir=str(pretrained_dir),
		hf_model_id=model_id,
		device_map="cuda",
		torch_dtype=model_dtype,
	)
	key_files = [
		"config.json",
		"tokenizer.json",
		"model.safetensors",
		"pytorch_model.bin",
	]
	available = [name for name in key_files if (pretrained_dir / name).exists()]
	result = {
		"status": "ok",
		"model_id": model_id,
		"pretrained_dir": str(pretrained_dir),
		"files_found": available,
		"tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
		"param_count": sum(p.numel() for p in model.parameters()),
	}
	print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
	main()
