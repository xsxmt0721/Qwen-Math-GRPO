from __future__ import annotations

import json
import random
import re
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _weighted_choice(rng: random.Random, weights: Sequence[float]) -> int:
	if not weights:
		return 0
	total = sum(float(w) for w in weights)
	if total <= 0:
		return 0
	threshold = rng.random() * total
	acc = 0.0
	for idx, weight in enumerate(weights):
		acc += float(weight)
		if threshold <= acc:
			return idx
	return len(weights) - 1


def _pick_other_step(rng: random.Random, pool: List[str]) -> str:
	if not pool:
		return ""
	return rng.choice(pool)


def _collect_brace_letters(text: str) -> List[str]:
	letters = re.findall(r"\{([A-Za-z])\}", text)
	return list(dict.fromkeys(letters))


def _get_math_spans(text: str) -> List[tuple[int, int]]:
	pattern = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\[(?:.|\n)*?\\\]", re.DOTALL)
	return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _replace_digit(text: str, rng: random.Random, span: tuple[int, int]) -> str:
	start, end = span
	segment = text[start:end]
	digits = [i for i, ch in enumerate(segment) if ch.isdigit()]
	if not digits:
		return text
	pos = rng.choice(digits)
	old = segment[pos]
	new_digit = rng.choice([d for d in "0123456789" if d != old])
	segment = segment[:pos] + new_digit + segment[pos + 1 :]
	return text[:start] + segment + text[end:]


def _replace_letter(text: str, rng: random.Random, span: tuple[int, int], letter_pool: List[str]) -> str:
	if len(letter_pool) < 2:
		return text
	start, end = span
	segment = text[start:end]
	letters = [i for i, ch in enumerate(segment) if ch.isalpha() and ch in letter_pool]
	if not letters:
		return text
	pos = rng.choice(letters)
	old = segment[pos]
	choices = [ch for ch in letter_pool if ch != old]
	if not choices:
		return text
	new_letter = rng.choice(choices)
	segment = segment[:pos] + new_letter + segment[pos + 1 :]
	return text[:start] + segment + text[end:]


def _replace_operator(text: str, rng: random.Random, span: tuple[int, int]) -> str:
	operators = ["^", "=", ">", "<", "(", ")", "+", "-", "*", "/"]
	start, end = span
	segment = text[start:end]
	positions = [i for i, ch in enumerate(segment) if ch in operators]
	if not positions:
		return text
	pos = rng.choice(positions)
	old = segment[pos]
	choices = [op for op in operators if op != old]
	new_op = rng.choice(choices)
	segment = segment[:pos] + new_op + segment[pos + 1 :]
	return text[:start] + segment + text[end:]


def _apply_format_error(text: str, rng: random.Random, latex_keywords: Sequence[str]) -> str:
	if not text:
		return text
	remove_chars = ["{", "}", "\\", "$", "[", "]", "_"]
	positions = [i for i, ch in enumerate(text) if ch in remove_chars]
	if positions:
		pos = rng.choice(positions)
		return text[:pos] + text[pos + 1 :]
	if latex_keywords:
		keyword = rng.choice(list(latex_keywords))
		if keyword and keyword in text:
			return text.replace(keyword, "", 1)
	return text


def _apply_math_error(text: str, rng: random.Random, mode: str) -> str:
	spans = _get_math_spans(text)
	if not spans:
		return text
	span = rng.choice(spans)
	if mode == "number":
		return _replace_digit(text, rng, span)
	if mode == "letter":
		letter_pool = _collect_brace_letters(text)
		return _replace_letter(text, rng, span, letter_pool)
	if mode == "operator":
		return _replace_operator(text, rng, span)
	return text


def _build_step_error(
	rng: random.Random,
	pos_next: str,
	answer_steps: List[str],
	step_idx: int,
	former: str,
	other_pool: List[str],
	config,
) -> str:
	step_weights = list(getattr(config, "neg_distribution_step", [1.0, 0.0, 0.0, 0.0]))
	choice = _weighted_choice(rng, step_weights)
	if choice == 0:
		return pos_next
	if choice == 1:
		min_skip = int(getattr(config, "min_skip_steps", 1))
		max_skip = int(getattr(config, "max_skip_steps", 1))
		offset = rng.randint(min_skip, max_skip)
		start = step_idx + offset
		end = min(step_idx + max_skip, len(answer_steps) - 1)
		if start > len(answer_steps) - 1:
			return ""
		return rng.choice(answer_steps[start : end + 1])
	if choice == 2:
		return _pick_other_step(rng, other_pool)
	former_steps = [step for step in former.split("\n") if step]
	if not former_steps:
		return ""
	return rng.choice(former_steps)


def _build_error_variant(text: str, rng: random.Random, config) -> str:
	error_weights = list(getattr(config, "neg_distribution_error", [1.0, 0.0, 0.0, 0.0, 0.0]))
	latex_keywords = getattr(config, "latex_keywords", [])
	iterations = int(getattr(config, "neg_distribution_num", 1))
	current = text
	for _ in range(iterations):
		choice = _weighted_choice(rng, error_weights)
		if choice == 0:
			continue
		if choice == 1:
			current = _apply_format_error(current, rng, latex_keywords)
			continue
		if choice == 2:
			current = _apply_math_error(current, rng, "number")
			continue
		if choice == 3:
			current = _apply_math_error(current, rng, "letter")
			continue
		if choice == 4:
			current = _apply_math_error(current, rng, "operator")
			continue
	return current


def build_negative_samples(
	pos_samples: List[dict],
	pos_meta: List[dict],
	batch_answers: List[List[str]],
	config,
	rng: random.Random,
) -> List[dict]:
	neg_samples: List[dict] = []
	for sample, meta in zip(pos_samples, pos_meta):
		pos_next = sample["next"]
		former = sample["former"]
		answer_steps = meta["answer_steps"]
		step_idx = int(meta["step_idx"])
		batch_idx = int(meta["batch_idx"])
		other_pool = [
			step
			for idx, steps in enumerate(batch_answers)
			if idx != batch_idx
			for step in steps
		]

		attempts = 0
		neg_next = pos_next
		while neg_next == pos_next:
			attempts += 1
			step_variant = _build_step_error(
				rng,
				pos_next,
				answer_steps,
				step_idx,
				former,
				other_pool,
				config,
			)
			neg_next = _build_error_variant(step_variant, rng, config)
			if attempts >= 10:
				if neg_next == pos_next:
					neg_next = ""
				break

		neg_samples.append(
			{
				"idx": sample["idx"],
				"question": sample["question"],
				"former": sample["former"],
				"next": neg_next,
				"label": 0,
			}
		)
	return neg_samples

def build_reward_prompt(
	batch: dict,
) -> dict:
	inputs = batch.get("input", [])
	labels = batch.get("label", [])
	if not inputs:
		return []
	if len(labels) != len(inputs):
		labels = list(labels)[: len(inputs)]

	system_prompt = (
		"# Description\n"
        "You are a reward model that judges whether the next step is correct. "
		"Given a math question, the former steps, and a candidate next step, "
		"output 1 if the next step is correct and consistent; otherwise output 0. "
		"# Constraints\n"
        "Only output a single digit: 0 or 1."
	)

	items: List[dict] = []
	for item, label in zip(inputs, labels):
		if isinstance(item, str):
			try:
				payload = json.loads(item)
			except Exception:
				payload = {}
		else:
			payload = item if isinstance(item, dict) else {}

		question = str(payload.get("question", ""))
		former = str(payload.get("former", ""))
		next_step = str(payload.get("next", ""))

		user_prompt = (
			"Question:\n"
			f"{question}\n\n"
			"Former steps:\n"
			f"{former}\n\n"
			"Next step:\n"
			f"{next_step}"
		)
		items.append(
			{
				"messages": [
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
					{"role": "assistant", "content": str(label)},
				]
			}
		)

	return items


def load_model(
		model_path: str, 
		device: str = "auto"
		) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    print(f"Successfully loaded model with device_map: {model.hf_device_map}")
    return model, tokenizer