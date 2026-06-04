from __future__ import annotations

import random
import re
from typing import List, Tuple, Any

import torch


_SYSTEM_PROMPT = (
	"# Task\n"
    "You are a helpful math assistant. "
	"Solve the problem that is presented. \n"
	"# Constraints\n"
    "- You must show your reasoning process step by step. "
    "- Put the final answer in \\boxed{...}."
	"\n"
	"# Question:\n"
)


def _normalize_answers(raw_answers) -> List[str]:
	if raw_answers is None:
		return []
	if isinstance(raw_answers, list):
		return [str(ans) for ans in raw_answers]
	return [str(raw_answers)]


def build_prompt(
	data: List[dict],
	shuffle: bool,
	all_ans: bool,
	seed: int
) -> Tuple[int, List[dict], List[str], List[str]]:
	samples: List[dict] = []
	questions_list: List[str] = []
	final_answers: List[str] = []

	for item in data:
		question = str(item.get("question", ""))
		answers = _normalize_answers(item.get("answer"))
		final_answer = str(item.get("final_answer", ""))

		if not answers:
			answers = [""]

		if len(answers) == 1 or not all_ans:
			answers = [answers[0]]

		for answer in answers:
			samples.append(
				{
					"messages": [
						{"role": "system", "content": _SYSTEM_PROMPT},
						{"role": "user", "content": question},
						{"role": "assistant", "content": str(answer)},
					]
				}
			)
			questions_list.append(question)
			final_answers.append(final_answer)

	if shuffle and samples:
		indices = list(range(len(samples)))
		random.seed(seed)
		random.shuffle(indices)
		samples = [samples[i] for i in indices]
		questions_list = [questions_list[i] for i in indices]
		final_answers = [final_answers[i] for i in indices]

	return len(samples), samples, questions_list, final_answers

def extract_boxed_answer(text: str) -> str:
    if not text:
        return ""
    # 支持 \boxed{...} 或 \box{...}
    pattern = r"\\box(?:ed)?\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return ""
    last = matches[-1].end()
    stack = 1
    i = last
    start = last
    while i < len(text):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i].strip()
        i += 1
    return ""


def extract_question(prompt: Any) -> str:
    if isinstance(prompt, dict):
        messages = prompt.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", "")).strip()
        return str(prompt.get("content", "")).strip()
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        return str(prompt).strip()
    text = str(prompt or "")
    marker = "# Question:\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()

def split_steps(text: str) -> List[str]:
    blocks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    if not blocks:
        return []
    merged: List[str] = []
    idx = 0
    while idx < len(blocks):
        item = blocks[idx]
        if item == "\\[":
            if idx + 1 < len(blocks):
                content = blocks[idx + 1]
                if idx + 2 < len(blocks) and blocks[idx + 2] == "\\]":
                    merged.append(f"\\[\n{content}\n\\]")
                    idx += 3
                    continue
                merged.append(f"\\[\n{content}")
                idx += 2
                continue
            merged.append(item)
            idx += 1
            continue
        if item == "\\]":
            if merged:
                merged[-1] = f"{merged[-1]}\n\\]"
            else:
                merged.append(item)
            idx += 1
            continue
        merged.append(item)
        idx += 1
    return merged

def build_prm_prompt(question: str, former: str, next_step: str, tokenizer) -> str:
    system_prompt = (
        "# Description\n"
        "You are a reward model that judges whether the next step is correct. "
        "Given a math question, the former steps, and a candidate next step, "
        "output 1 if the next step is correct and consistent; otherwise output 0. "
        "# Constraints\n"
        "Only output a single digit: 0 or 1."
    )
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Former steps:\n"
        f"{former}\n\n"
        "Next step:\n"
        f"{next_step}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """返回两个 tensor 的公共前缀长度（沿最后一维逐元素比较）。

    input_ids 通常 shape 为 (1, seq_len)，tolist() 会得到嵌套列表，
    因此需要 view(-1) 展平后再比较。
    """
    if a.numel() == 0 or b.numel() == 0:
        return 0
    a_list: list = a.view(-1).tolist()
    b_list: list = b.view(-1).tolist()
    limit = min(len(a_list), len(b_list))
    idx = 0
    while idx < limit and a_list[idx] == b_list[idx]:
        idx += 1
    return idx


def slice_past(past, length: int):
    if past is None:
        return None
    sliced = []
    for layer in past:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            key, value = layer[0], layer[1]
            if key is None or value is None:
                sliced.append(layer)
                continue
            key = key[:, :, :length, :]
            value = value[:, :, :length, :]
            if len(layer) > 2:
                sliced.append((key, value, *layer[2:]))
            else:
                sliced.append((key, value))
        else:
            sliced.append(layer)
    return tuple(sliced)