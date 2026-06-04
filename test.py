"""
Evaluation module for GRPO-trained math reasoning models.

Evaluates on multiple metrics per source dataset:
- @1-Accuracy: 1st generation answer correctness
- @3-Accuracy: at least 1 of 3 generations correct
- @1-Steps-Accuracy: PRM step score for 1st generation
- @3-Steps-Avg-Accuracy: average PRM step score over 3 generations
- Average generation length
"""

from __future__ import annotations

import json
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.dataloader import DataLoader
from utils.utils import (
    build_prm_prompt,
    common_prefix_len,
    extract_boxed_answer,
    slice_past,
    split_steps,
)
from train import _load_env, _predict_step_label


# ═══════════════════════════════════════════════════════════════════════
# Helper: JSON I/O
# ═══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# Test Configuration
# ═══════════════════════════════════════════════════════════════════════

class test_config:
    # ── Paths ──
    log_dir: str = "/workspace/logs/grpo_logs/grpo-1.5b"

    # ── Data ──
    eval_batch_size: int = 4           # prompts per generation batch
    max_test_size: int = 300           # max total test samples (use large default)
    max_length: int = 1024             # prompt + completion max tokens
    max_completion_length: int = 512   # max new tokens per generation

    # ── Generation ──
    num_generations: int = 3           # K completions per prompt
    temperature: float = 1.0

    # ── PRM steps scoring ──
    prm_num_wrong_stop: int = 1
    prm_use_prefix_kv_cache: bool = True

    seed: int = 42
    transform: bool = True
    transform_all_ans: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Scoring functions
# ═══════════════════════════════════════════════════════════════════════

def _compute_result_scores(
    completions: List[str],
    answers: List[str],
) -> List[float]:
    """Binary answer‑match scores (0/1) for a flat list of completions."""
    scores: List[float] = []
    for completion, answer in zip(completions, answers):
        extracted = extract_boxed_answer(str(completion))
        score = 1.0 if extracted.strip() == str(answer).strip() and extracted else 0.0
        scores.append(score)
    return scores


def _compute_steps_scores(
    completions: List[str],
    questions: List[str],
    prm_model,
    prm_tokenizer,
    device: torch.device,
    max_length: int,
    num_wrong_stop: int = 1,
    use_prefix_kv_cache: bool = True,
) -> List[float]:
    """Compute PRM step scores (0..1) for a flat list of completions.

    Returns one score per completion.
    """
    prm_model.eval()
    scores: List[float] = []
    for completion, question in zip(completions, questions):
        steps = split_steps(str(completion))
        total = len(steps)
        if total == 0:
            scores.append(0.0)
            continue

        former = ""
        correct = 0
        wrong = 0
        prev_input_ids = None
        prev_past = None

        for step in steps:
            prompt_text = build_prm_prompt(question, former, step, prm_tokenizer)
            label, prev_input_ids, prev_past = _predict_step_label(
                prompt_text,
                prm_model,
                prm_tokenizer,
                device,
                max_length,
                prev_input_ids,
                prev_past,
                use_prefix_kv_cache,
            )
            if label == 1:
                correct += 1
            else:
                wrong += 1
                if num_wrong_stop > 0 and wrong >= num_wrong_stop:
                    break
            former = f"{former}\n{step}" if former else step

        scores.append(1.0 if correct == total else correct / total)
    return scores


# ═══════════════════════════════════════════════════════════════════════
# Dataset grouping helper
# ═══════════════════════════════════════════════════════════════════════

def _build_source_boundaries(
    datasets_meta: Dict[str, Any],
    actual_total: int,
) -> List[Tuple[str, int, int]]:
    """Return [(source_name, start_idx, end_idx), ...] for the merged test data.

    The DataLoader (with shuffle=False) appends datasets in dict‑insertion
    order.  We use the ``test`` field from SPLIT_META to determine sizes.
    Boundaries are bounded by the actual loaded data size.
    """
    boundaries: List[Tuple[str, int, int]] = []
    cursor = 0
    for src_name, src_meta in datasets_meta.items():
        size = int(src_meta.get("test", 0))
        end = min(cursor + size, actual_total)
        if end > cursor:
            boundaries.append((src_name, cursor, end))
            cursor = end
        if cursor >= actual_total:
            break
    return boundaries


# ═══════════════════════════════════════════════════════════════════════
# Main evaluation logic
# ═══════════════════════════════════════════════════════════════════════

def evaluate() -> None:
    # ── Load environment ──
    env = _load_env(Path(__file__).parent / ".env")
    DATA_DIR: str = env.get("DATA_DIR", "/data")
    SPLIT_DIR: str = env.get("DATA_SPLIT_DIR", "/workspace/logs/data_split")
    MODEL_PATH: str = env.get("BASEMODEL_DIR", "")
    GRPO_SAVE: str = env.get("GRPO_SAVE", "")
    PRM_DIR: str = env.get("PRM_DIR", "")
    PRM_SAVE: str = env.get("PRM_SAVE", "")

    print("=" * 60)
    print("GRPO Evaluation")
    print(f"  Base model : {MODEL_PATH}")
    print(f"  GRPO LoRA  : {GRPO_SAVE}")
    print(f"  PRM        : {PRM_DIR}")
    print(f"  Log dir    : {test_config.log_dir}")
    print("=" * 60)

    os.makedirs(test_config.log_dir, exist_ok=True)

    # ── Load SPLIT_META ──
    meta = _load_json(Path(SPLIT_DIR) / "SPLIT_META.json")
    datasets_meta: Dict[str, Any] = meta.get("datasets", {})
    print(f"\nDatasets in meta: {list(datasets_meta.keys())}")

    # ── Load test data via DataLoader (shuffle=False → ordered by source) ──
    print("\n[1/5] Loading test data ...")
    test_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="TEST",
        batch_size=test_config.eval_batch_size,
        shuffle=False,
        transform=test_config.transform,
        transform_all_ans=test_config.transform_all_ans,
        include_answer=False,
        seed=test_config.seed,
    )
    print(f"  Total test samples loaded: {test_loader.num_data}")

    actual_total = test_loader.num_data
    print(f"  Effective test samples: {actual_total}")

    # ── Compute per‑source boundaries ──
    source_boundaries = _build_source_boundaries(datasets_meta, actual_total)
    print("\n  Source boundaries:")
    for name, start, end in source_boundaries:
        print(f"    {name}: [{start}, {end})  ({end - start} samples)")

    # ── Prepare per‑source slices, then truncate each independently ──
    source_data: Dict[str, Dict[str, List[Any]]] = {}
    for src_name, start, end in source_boundaries:
        raw_size = end - start
        take = min(raw_size, test_config.max_test_size)
        if take > 0:
            source_data[src_name] = {
                "data": test_loader.data[start:start + take],
                "questions": test_loader.questions[start:start + take],
                "answers": test_loader.ans[start:start + take],
            }
            if take < raw_size:
                print(f"    {src_name}: truncated from {raw_size} to {take} samples")

    # ── Quantisation config ──
    compute_dtype = torch.bfloat16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # ── Load policy model (base + merge GRPO LoRA) ──
    print("\n[2/5] Loading policy model ...")
    policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    if GRPO_SAVE and Path(GRPO_SAVE).exists():
        policy_model = PeftModel.from_pretrained(policy_model, GRPO_SAVE)
        print(f"  Loaded GRPO LoRA from: {GRPO_SAVE}")
    else:
        print(f"  WARNING: GRPO_SAVE not found ({GRPO_SAVE}), using base model only")

    policy_model.eval()
    policy_device = next(policy_model.parameters()).device
    print(f"  Policy model device: {policy_device}")

    # ── Load PRM model ──
    print("\n[3/5] Loading PRM model ...")
    prm_tokenizer = AutoTokenizer.from_pretrained(PRM_DIR, trust_remote_code=True)
    prm_model = AutoModelForCausalLM.from_pretrained(
        PRM_DIR,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    if PRM_SAVE and Path(PRM_SAVE).exists():
        prm_model = PeftModel.from_pretrained(prm_model, PRM_SAVE)
        print(f"  Loaded PRM LoRA from: {PRM_SAVE}")

    if prm_tokenizer.pad_token is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
    prm_model.config.pad_token_id = prm_tokenizer.pad_token_id
    prm_model.eval()
    prm_device = next(prm_model.parameters()).device
    print(f"  PRM model device: {prm_device}")

    # ── Evaluate each source ──
    print("\n[4/5] Running evaluation ...")
    K = test_config.num_generations
    results: Dict[str, Any] = {"config": {k: v for k, v in test_config.__dict__.items() if not k.startswith("__")}}

    for src_name, sdata in source_data.items():
        print(f"\n--- Evaluating: {src_name} ---")
        src_questions = sdata["questions"]
        src_answers = sdata["answers"]
        src_data = sdata["data"]
        n_samples = len(src_questions)
        print(f"  Samples: {n_samples}")

        # Accumulators
        all_first_result_scores: List[float] = []    # for @1-Accuracy
        all_first_steps_scores: List[float] = []     # for @1-Steps-Accuracy
        all_3result_flags: List[float] = []           # for @3-Accuracy (1 if any of 3 correct)
        all_3steps_avg: List[float] = []              # for @3-Steps-Avg-Accuracy
        all_gen_lengths: List[int] = []               # token counts of generated part

        batch_size = test_config.eval_batch_size
        num_batches = ceil(n_samples / batch_size) if batch_size > 0 else 0

        for batch_idx in tqdm(range(num_batches), desc=f"  {src_name}", unit="batch"):
            b_start = batch_idx * batch_size
            b_end = min((batch_idx + 1) * batch_size, n_samples)

            batch_questions = src_questions[b_start:b_end]
            batch_answers = src_answers[b_start:b_end]
            batch_data = src_data[b_start:b_end]

            # ── Build prompts ──
            prompts: List[str] = []
            for item in batch_data:
                prompt_text = policy_tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_text)

            # ── Tokenize ──
            encoded = policy_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=test_config.max_length,
            )
            input_ids = encoded["input_ids"].to(policy_device)
            attention_mask = encoded["attention_mask"].to(policy_device)
            # Per‑sample actual prompt lengths (no padding)
            prompt_lens = attention_mask.sum(dim=1).tolist()  # list of int, len = batch_size

            # ── Generate K completions per prompt ──
            with torch.no_grad():
                gen_outputs = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=test_config.max_completion_length,
                    do_sample=True,
                    temperature=test_config.temperature,
                    num_return_sequences=K,
                    pad_token_id=policy_tokenizer.pad_token_id,
                    eos_token_id=policy_tokenizer.eos_token_id,
                )

            # gen_outputs shape: (batch_size * K, seq_len)
            # Order: [s0_g0, s0_g1, s0_g2, s1_g0, s1_g1, s1_g2, ...]
            batch_size_actual = b_end - b_start

            # ── Decode completions (only newly generated tokens) ──
            all_completions: List[str] = []
            for s in range(batch_size_actual):
                actual_prompt_len = int(prompt_lens[s])
                for k in range(K):
                    i = s * K + k
                    new_tokens = gen_outputs[i, actual_prompt_len:]
                    comp_text = policy_tokenizer.decode(new_tokens, skip_special_tokens=True)
                    all_completions.append(comp_text)
                    all_gen_lengths.append(int(new_tokens.numel()))

            # ── Result scores (0/1) for all K*N completions ──
            # Replicate answers K times to match interleaved completions
            flat_answers: List[str] = []
            for ans in batch_answers:
                flat_answers.extend([ans] * K)
            flat_result_scores = _compute_result_scores(all_completions, flat_answers)

            # ── Per‑sample metrics ──
            for s in range(batch_size_actual):
                base = s * K
                # @1-Accuracy & @1-Steps: use first completion only
                all_first_result_scores.append(flat_result_scores[base])

                # @3-Accuracy: at least one correct
                three_results = flat_result_scores[base:base + K]
                all_3result_flags.append(1.0 if any(r > 0.5 for r in three_results) else 0.0)

            # ── Steps scores (PRM) for first completions only ──
            first_completions = [all_completions[s * K] for s in range(batch_size_actual)]
            first_steps = _compute_steps_scores(
                first_completions,
                batch_questions,
                prm_model,
                prm_tokenizer,
                prm_device,
                test_config.max_length,
                num_wrong_stop=test_config.prm_num_wrong_stop,
                use_prefix_kv_cache=test_config.prm_use_prefix_kv_cache,
            )
            all_first_steps_scores.extend(first_steps)

            # ── Steps scores (PRM) for all K completions → average per sample ──
            flat_questions: List[str] = []
            for q in batch_questions:
                flat_questions.extend([q] * K)
            all_steps = _compute_steps_scores(
                all_completions,
                flat_questions,
                prm_model,
                prm_tokenizer,
                prm_device,
                test_config.max_length,
                num_wrong_stop=test_config.prm_num_wrong_stop,
                use_prefix_kv_cache=test_config.prm_use_prefix_kv_cache,
            )
            for s in range(batch_size_actual):
                base = s * K
                avg_steps = sum(all_steps[base:base + K]) / K
                all_3steps_avg.append(avg_steps)

        # ── Aggregate metrics ──
        n = len(all_first_result_scores)
        metrics = {
            "@1-Accuracy": sum(all_first_result_scores) / n if n else 0.0,
            "@3-Accuracy": sum(all_3result_flags) / n if n else 0.0,
            "@1-Steps-Accuracy": sum(all_first_steps_scores) / n if n else 0.0,
            "@3-Steps-Avg-Accuracy": sum(all_3steps_avg) / n if n else 0.0,
            "avg_generation_length": sum(all_gen_lengths) / len(all_gen_lengths) if all_gen_lengths else 0.0,
        }

        results[src_name] = metrics
        print(f"  Results for {src_name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    # ── Save results ──
    print(f"\n[5/5] Saving results to {test_config.log_dir} ...")
    result_path = Path(test_config.log_dir) / "test_results.json"
    _save_json(result_path, results)
    print(f"  Results saved to: {result_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
