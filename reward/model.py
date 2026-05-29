"""PRM model loader with local pretrained cache."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_HF_MODEL_ID = "Qwen/Qwen2.5-1.5B"


def _is_valid_pretrained_dir(path: Path) -> bool:
	if not path.exists() or not path.is_dir():
		return False
	required_any = [
		"model.safetensors",
		"pytorch_model.bin",
	]
	if not any((path / name).exists() for name in required_any):
		return False
	return (path / "config.json").exists() and (path / "tokenizer.json").exists()


def _ensure_pretrained_weights(
	pretrained_dir: Path,
	hf_model_id: str,
	trust_remote_code: bool = True,
) -> None:
	if _is_valid_pretrained_dir(pretrained_dir):
		return

	pretrained_dir.mkdir(parents=True, exist_ok=True)
	tokenizer = AutoTokenizer.from_pretrained(
		hf_model_id,
		trust_remote_code=trust_remote_code,
	)
	model = AutoModelForCausalLM.from_pretrained(
		hf_model_id,
		trust_remote_code=trust_remote_code,
	)
	tokenizer.save_pretrained(pretrained_dir)
	model.save_pretrained(pretrained_dir, safe_serialization=True)


def load_prm_model_and_tokenizer(
	pretrained_dir: str,
	hf_model_id: str = DEFAULT_HF_MODEL_ID,
	device_map: Optional[str] = "auto",
	torch_dtype: Optional[torch.dtype] = None,
	trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
	"""Load PRM model/tokenizer and ensure a local pretrained cache exists."""
	if not pretrained_dir:
		raise ValueError("Missing pretrained_dir")
	pretrained_dir_path = Path(pretrained_dir).expanduser().resolve()
	model_id = hf_model_id or DEFAULT_HF_MODEL_ID

	_ensure_pretrained_weights(
		pretrained_dir=pretrained_dir_path,
		hf_model_id=model_id,
		trust_remote_code=trust_remote_code,
	)

	tokenizer = AutoTokenizer.from_pretrained(
		pretrained_dir_path,
		local_files_only=True,
		trust_remote_code=trust_remote_code,
	)
	model = AutoModelForCausalLM.from_pretrained(
		pretrained_dir_path,
		local_files_only=True,
		trust_remote_code=trust_remote_code,
		device_map=device_map,
		torch_dtype=torch_dtype,
	)
	model.config.use_cache = False

	return model, tokenizer


__all__ = ["load_prm_model_and_tokenizer"]
