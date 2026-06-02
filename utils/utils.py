from __future__ import annotations

import random
from typing import List, Tuple


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
) -> Tuple[int, List[dict], List[str]]:
	samples: List[dict] = []
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
			final_answers.append(final_answer)

	if shuffle and samples:
		indices = list(range(len(samples)))
		random.seed(seed)
		random.shuffle(indices)
		samples = [samples[i] for i in indices]
		final_answers = [final_answers[i] for i in indices]

	return len(samples), samples, final_answers
