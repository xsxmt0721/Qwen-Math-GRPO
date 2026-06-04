# GRPO Trainer
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import matplotlib.pyplot as plt

from utils.dataloader import DataLoader
from utils.utils import *

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

os.environ["WANDB_MODE"] = "offline"

env = _load_env(Path(__file__).parent / ".env")

# 数据集路径
DATA_DIR: str = env.get("DATA_DIR", "/data")
# 数据集划分路径
SPLIT_DIR: str = env.get("DATA_SPLIT_DIR", "/data_split")

# GRPO & REF模型基座路径
MODEL_PATH: str = env.get("BASEMODEL_DIR", "")
# GRPO 模型保存路径
SAVE_PATH: str = env.get("GRPO_SAVE", "")
# 参考模型微调权重保存路径
REF_DIR: str = env.get("REF_SAVE", "")
REF_SAVE: str = os.path.join(REF_DIR, "checkpoint-138")
# PRM 模型基座路径
PRM_DIR: str = env.get("PRM_DIR", "")
# PRM 微调权重保存路径
PRM_SAVE: str = env.get("PRM_SAVE", "")

class train_config:
    log_dir = "/workspace/logs/grpo_logs/grpo-1.5b-result-most"

    # ── Data ──
    batch_size: int = 4
    eval_batch_size: int = 4
    train_data_total: int = 2400
    eval_data_total: int = 500

    # ── Reward ──
    reward_mode: str = "full"  # "full"=PRM过程分+结果分加权 / "result_only"=仅答案匹配，快速验证
    reward_alpha: float = 0.9
    reward_beta: float = 0.1
    reward_num_wrong_stop: int = 1
    reward_use_prefix_kv_cache: bool = True

    # ── GRPO ──
    num_generations: int = 4          # 每个 prompt 采样 K 条 completion（group size）
    max_completion_length: int = 512  # 单条 completion 最大 token 数
    kl_beta: float = 0.04             # KL 散度惩罚系数
    temperature: float = 1.0          # 生成温度（越大越多样）
    # top_p: float = 0.95             # nucleus sampling

    # ── Training ──
    epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 4
    max_length: int = 1024            # prompt + completion 最大 token 数

    # ── LoRA ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ── Logging ──
    log_step: int = 10
    eval_step: int = 200
    save_step: int = 200

    seed = 42
    shuffle: bool = True
    transform: bool = True
    transform_all_ans: bool = False

def _predict_step_label(
    prompt_text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    prev_input_ids: Optional[torch.Tensor],
    prev_past,
    use_prefix_kv_cache: bool,
) -> tuple[int, torch.Tensor, Any]:
    encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    past = None
    if use_prefix_kv_cache and prev_input_ids is not None and prev_past is not None:
        prefix_len = common_prefix_len(prev_input_ids, input_ids)
        if prefix_len >= input_ids.size(1):
            prefix_len = max(input_ids.size(1) - 1, 0)
        if prefix_len > 0:
            past = slice_past(prev_past, prefix_len)
            input_ids = input_ids[:, prefix_len:]
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True,
        )
    logits = outputs.logits[:, -1, :]
    pred_id = int(torch.argmax(logits, dim=-1).item())
    pred_text = tokenizer.decode([pred_id]).strip()
    pred_char = pred_text[-1] if pred_text else ""
    label = 1 if pred_char == "1" else 0
    return label, encoded["input_ids"], outputs.past_key_values


def reward_func(
    completions: List[str],
    mode: str = "full",
    alpha: float = 0.6,
    beta: float = 0.4,
    num_wrong_stop: int = 1,
    use_prefix_kv_cache: bool = True,
    **kwargs,
) -> List[float]:
    # print(f"Reward function called with {len(completions)} completions (mode={mode})")
    answers: Sequence[str] = kwargs.get("answers", [])

    # ── result_only 模式：仅用答案匹配，无需 PRM ──
    if mode == "result_only":
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            answer = answers[idx] if idx < len(answers) else ""
            extracted = extract_boxed_answer(str(completion))
            result_score = 1.0 if extracted.strip() == str(answer).strip() and extracted else 0.0
            _write_reward_detail(kwargs, result_score, 0.0, result_score)
            rewards.append(float(result_score))
        return rewards

    # ── full 模式：PRM 过程分 + 结果分加权 ──
    questions: Sequence[str] = kwargs.get("question", [])
    prm_model = kwargs.get("prm_model")
    prm_tokenizer = kwargs.get("prm_tokenizer")
    max_length = int(kwargs.get("max_length", 2048))

    rewards: List[float] = []
    if prm_model is None or prm_tokenizer is None:
        raise ValueError("reward_func requires prm_model and prm_tokenizer in kwargs for full mode")

    device = next(prm_model.parameters()).device
    prm_model.eval()

    for idx, completion in enumerate(completions):
        answer = answers[idx] if idx < len(answers) else ""
        question = questions[idx] if idx < len(questions) else ""

        extracted = extract_boxed_answer(str(completion))
        result_score = 1.0 if extracted.strip() == str(answer).strip() and extracted else 0.0

        steps = split_steps(str(completion))
        total_steps = len(steps)
        if total_steps == 0:
            steps_score = 0.0
            reward = alpha * result_score + beta * steps_score
            _write_reward_detail(kwargs, result_score, steps_score, reward)
            rewards.append(float(reward))
            continue

        former = ""
        correct_steps = 0
        wrong_steps = 0
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
                correct_steps += 1
            else:
                wrong_steps += 1
                if num_wrong_stop > 0 and wrong_steps >= num_wrong_stop:
                    break
            former = f"{former}\n{step}" if former else step

        if correct_steps == total_steps:
            steps_score = 1.0
        else:
            steps_score = correct_steps / total_steps

        reward = alpha * result_score + beta * steps_score
        _write_reward_detail(kwargs, result_score, steps_score, reward)
        rewards.append(float(reward))
    return rewards


def _write_reward_detail(kwargs: dict, result_score: float, steps_score: float, composite: float) -> None:
    """侧信道：将分项奖励写入共享容器，供 MetricsCallback 消费"""
    detail = kwargs.get("_reward_detail")
    if detail is not None:
        detail["result_scores"].append(result_score)
        detail["steps_scores"].append(steps_score)
        detail["composite_scores"].append(composite)


def _save_train_config(log_dir: str) -> None:
    """将 train_config 保存为 JSON，便于日后溯源复现"""
    payload = {
        key: value
        for key, value in train_config.__dict__.items()
        if not key.startswith("__")
    }
    os.makedirs(log_dir, exist_ok=True)
    path = Path(log_dir) / "train_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Train config saved to: {path}")


class MetricsCallback(TrainerCallback):
    """收集训练过程中的 loss 和分项 reward，在训练结束时生成折线图和 JSON。

    TRL GRPOTrainer 的 log 格式:
      - 训练步: {"loss": ..., "reward": ..., "kl": ..., ...}
      - 评估步: {"eval_loss": ..., "eval_reward": ..., ...}

    通过侧信道 _reward_detail 收集 result_score / steps_score / composite 的逐条数据，
    在每个 eval_step 处对 train 和 eval 分别聚合求平均。
    """

    def __init__(self, log_dir: str, reward_detail: Optional[dict] = None):
        self.log_dir = Path(log_dir)
        self._reward_detail = reward_detail

        # ── 训练 loss ──
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self._last_train_step = 0

        # ── 当前 eval 区间内的训练奖励累积器 ──
        self._train_result_acc: List[float] = []
        self._train_steps_acc: List[float] = []
        self._train_composite_acc: List[float] = []

        # ── 每个 eval_step 的快照 ──
        self.eval_steps: List[int] = []
        self.eval_result_rewards: List[float] = []
        self.eval_steps_rewards: List[float] = []
        self.eval_composite_rewards: List[float] = []
        self.train_result_rewards: List[float] = []
        self.train_steps_rewards: List[float] = []
        self.train_composite_rewards: List[float] = []

    # ── 辅助：从侧信道排空数据 ──
    def _drain_detail(self) -> tuple:
        if self._reward_detail is None:
            return [], [], []
        result = self._reward_detail.get("result_scores", [])
        steps = self._reward_detail.get("steps_scores", [])
        composite = self._reward_detail.get("composite_scores", [])
        # 取出后清空
        out_result = list(result)
        out_steps = list(steps)
        out_composite = list(composite)
        result.clear()
        steps.clear()
        composite.clear()
        return out_result, out_steps, out_composite

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)

        is_eval = "eval_loss" in logs

        # 排空侧信道
        result_scores, steps_scores, composite_scores = self._drain_detail()

        if is_eval:
            # ── 评估步：记录 eval 数据 + 快照当前区间训练数据 ──
            self.eval_steps.append(step)
            self.eval_losses.append(float(logs["eval_loss"]))

            # eval 平均
            self.eval_result_rewards.append(
                sum(result_scores) / len(result_scores) if result_scores else 0.0
            )
            self.eval_steps_rewards.append(
                sum(steps_scores) / len(steps_scores) if steps_scores else 0.0
            )
            self.eval_composite_rewards.append(
                sum(composite_scores) / len(composite_scores) if composite_scores else 0.0
            )

            # train 平均（当前 eval 区间）
            self.train_result_rewards.append(
                sum(self._train_result_acc) / len(self._train_result_acc) if self._train_result_acc else 0.0
            )
            self.train_steps_rewards.append(
                sum(self._train_steps_acc) / len(self._train_steps_acc) if self._train_steps_acc else 0.0
            )
            self.train_composite_rewards.append(
                sum(self._train_composite_acc) / len(self._train_composite_acc) if self._train_composite_acc else 0.0
            )

            # 重置训练累积器
            self._train_result_acc.clear()
            self._train_steps_acc.clear()
            self._train_composite_acc.clear()

        elif "loss" in logs:
            # ── 训练步：累积 loss 和 reward ──
            self.train_losses.append(float(logs["loss"]))

            # 将本 log_step 内的奖励累积到区间容器
            self._train_result_acc.extend(result_scores)
            self._train_steps_acc.extend(steps_scores)
            self._train_composite_acc.extend(composite_scores)

    def on_train_end(self, args, state, control, **kwargs):
        self._save_metrics()

    def _save_metrics(self):
        os.makedirs(self.log_dir, exist_ok=True)

        reward_groups = [
            ("composite", "Composite Reward (α·result + β·steps)",
             self.train_composite_rewards, self.eval_composite_rewards),
            ("result", "Result Reward (\\boxed{} match)",
             self.train_result_rewards, self.eval_result_rewards),
            ("steps", "Steps Reward (PRM correctness)",
             self.train_steps_rewards, self.eval_steps_rewards),
        ]

        for key, title, train_rw, eval_rw in reward_groups:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            if self.eval_steps and train_rw:
                ax.plot(self.eval_steps, train_rw, label="train_reward",
                        color="#1f77b4", linewidth=1.2, marker="s", markersize=4)
            if self.eval_steps and eval_rw:
                ax.plot(self.eval_steps, eval_rw, label="eval_reward",
                        color="#ff7f0e", linewidth=2.0, marker="o", markersize=5)
            ax.set_xlabel("step")
            ax.set_ylabel("reward")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = self.log_dir / f"reward_{key}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Reward curve ({key}) saved to: {path}")

        # ── loss 折线图 ──
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if self.train_losses:
            train_steps = list(range(1, len(self.train_losses) + 1))
            ax.plot(train_steps, self.train_losses, label="train_loss",
                    color="#1f77b4", linewidth=1.2)
        if self.eval_steps:
            ax.plot(self.eval_steps, self.eval_losses, label="eval_loss",
                    color="#ff7f0e", linewidth=2.0, marker="o", markersize=4)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("GRPO Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        loss_path = self.log_dir / "loss_curve.png"
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to: {loss_path}")

        # ── 汇总 JSON ──
        summary = {
            "train_losses": [round(l, 6) for l in self.train_losses],
            "eval_steps": self.eval_steps,
            "eval_losses": [round(l, 6) for l in self.eval_losses],
            "train_composite_rewards": [round(r, 6) for r in self.train_composite_rewards],
            "eval_composite_rewards": [round(r, 6) for r in self.eval_composite_rewards],
            "train_result_rewards": [round(r, 6) for r in self.train_result_rewards],
            "eval_result_rewards": [round(r, 6) for r in self.eval_result_rewards],
            "train_steps_rewards": [round(r, 6) for r in self.train_steps_rewards],
            "eval_steps_rewards": [round(r, 6) for r in self.eval_steps_rewards],
        }
        summary_path = self.log_dir / "metrics_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Metrics summary saved to: {summary_path}")


if __name__ == "__main__":
    # 导入数据
    print("Model Save Dir:", SAVE_PATH)
    print("Log Save Dir:", train_config.log_dir)
    train_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="GRPO_TRAIN",
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
        transform=train_config.transform,
        transform_all_ans=train_config.transform_all_ans,
        seed=train_config.seed,
    )
    val_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="GRPO_VAL",
        batch_size=train_config.eval_batch_size,
        shuffle=train_config.shuffle,
        transform=train_config.transform,
        transform_all_ans=train_config.transform_all_ans,
        seed=train_config.seed,
    )
    
    train_dataset = Dataset.from_list(train_loader.data)
    train_dataset = train_dataset.add_column("answers", train_loader.ans)
    train_dataset = train_dataset.add_column("question", train_loader.questions)
    val_dataset = Dataset.from_list(val_loader.data)
    val_dataset = val_dataset.add_column("answers", val_loader.ans)
    val_dataset = val_dataset.add_column("question", val_loader.questions)

    # ── 根据 train_config 截断数据集大小──
    train_total = min(train_config.train_data_total, len(train_dataset))
    eval_total = min(train_config.eval_data_total, len(val_dataset))
    if train_total < len(train_dataset):
        train_dataset = train_dataset.select(range(train_total))
        print(f"Train dataset truncated to {train_total} samples (from {len(train_loader.data)})")
    if eval_total < len(val_dataset):
        val_dataset = val_dataset.select(range(eval_total))
        print(f"Eval dataset truncated to {eval_total} samples (from {len(val_loader.data)})")

    # ── 公共量化配置（PRM 和 Policy 模型共用）──
    compute_dtype = torch.bfloat16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # ── PRM 模型（仅在 full 模式下需要）──
    prm_model = None
    prm_tokenizer = None
    if train_config.reward_mode == "full":
        prm_tokenizer = AutoTokenizer.from_pretrained(PRM_DIR, trust_remote_code=True)
        prm_model = AutoModelForCausalLM.from_pretrained(
            PRM_DIR,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )

        # 加载 PRM 微调后的 LoRA 权重
        if PRM_SAVE and Path(PRM_SAVE).exists():
            prm_model = PeftModel.from_pretrained(prm_model, PRM_SAVE)
            print(f"Loaded PRM LoRA weights from: {PRM_SAVE}")

        if prm_tokenizer.pad_token is None:
            prm_tokenizer.pad_token = prm_tokenizer.eos_token
        prm_model.config.pad_token_id = prm_tokenizer.pad_token_id
        prm_model.eval()

        print(f"PRM model loaded. Device: {next(prm_model.parameters()).device}")
    else:
        print("PRM model skipped (reward_mode=result_only)")

    # 加载 tokenizer（策略模型和 processing_class 共用）
    # GRPOTrainer 内部会自动复制 policy_model 作为冻结参考模型，无需手动加载 ref_model
    policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # 预生成 "prompt" 列：用 tokenizer 将 messages 转为纯文本 prompt（GRPOTrainer 需要此列）
    def _build_prompt_text(example: dict) -> dict:
        example["prompt"] = policy_tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        return example
    train_dataset = train_dataset.map(_build_prompt_text)
    val_dataset = val_dataset.map(_build_prompt_text)
    # 移除 messages
    train_dataset = train_dataset.remove_columns(["messages"])
    val_dataset = val_dataset.remove_columns(["messages"])

    # 加载 GRPO 策略模型（以 SFT checkpoint 为起点，合并后应用新 LoRA 进行 GRPO 训练）
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    # 加载 SFT 微调后的 LoRA checkpoint 作为初始策略
    if REF_SAVE and Path(REF_SAVE).exists():
        policy_model = PeftModel.from_pretrained(policy_model, REF_SAVE)
        # 将 SFT 学到的 LoRA 权重合并到基座中
        policy_model = policy_model.merge_and_unload()
        print(f"Merged SFT LoRA into GRPO policy model from: {REF_SAVE}")
    else:
        print("Warning: REF_SAVE path not found, starting GRPO from base model")

    # 在 SFT 合并后的基座上应用新的 LoRA adapter 用于 GRPO 训练
    policy_model = prepare_model_for_kbit_training(policy_model)
    # prepare_model_for_kbit_training 内部会自动启用 gradient checkpointing，
    # GRPOConfig 中 gradient_checkpointing=True 交由 TRL 统一管理。
    # 生成阶段的 KV cache 问题通过下方的 generate monkey-patch 解决。
    grpo_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    policy_model = get_peft_model(policy_model, grpo_lora_config)

    # GRPO 训练配置
    policy_model.train()

    print(f"GRPO policy model loaded. Device: {next(policy_model.parameters()).device}")

    # 设置 GRPO 训练配置
    grpo_config = GRPOConfig(
        output_dir=SAVE_PATH,
        logging_dir=train_config.log_dir,
        report_to="tensorboard",

        # ── GRPO 核心参数 ──
        num_generations=train_config.num_generations,
        max_completion_length=train_config.max_completion_length,
        beta=train_config.kl_beta,
        temperature=train_config.temperature,

        # ── 训练参数 ──
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.grad_accum_steps,
        per_device_eval_batch_size=train_config.eval_batch_size,
        weight_decay=train_config.weight_decay,
        max_grad_norm=train_config.max_grad_norm,

        # ── 精度 & 显存 ──
        bf16=True,
        # 训练时启用 gradient checkpointing 以节省显存（反向传播时 activations 不常驻），
        # 生成阶段的 KV cache 通过 monkey-patch generate 来临时关闭 checkpointing 解决。
        gradient_checkpointing=True,

        # ── 日志 & 保存 ──
        logging_strategy="steps",
        logging_steps=train_config.log_step,
        eval_strategy="steps",
        eval_steps=train_config.eval_step,
        save_strategy="steps",
        save_steps=train_config.save_step,
        save_total_limit=3,

        # ── 其他 ──
        seed=train_config.seed,
        remove_unused_columns=False,   # 保留 "answers" / "question" 列传给 reward_func
    )

    # 创建共享的奖励侧信道容器
    _reward_detail: dict = {"result_scores": [], "steps_scores": [], "composite_scores": []}

    # 保存训练配置，便于未来溯源复现
    _save_train_config(train_config.log_dir)

    # 注册指标收集回调（训练过程中收集，训练结束时自动保存图表和 JSON）
    metrics_callback = MetricsCallback(train_config.log_dir, reward_detail=_reward_detail)

    # 用闭包将 PRM 等额外参数绑定到 reward_func（trl 0.15.x 无 reward_kwargs，且要求 __name__）
    def reward_func_bound(completions, **kwargs):
        return reward_func(
            completions,
            mode=train_config.reward_mode,
            prm_model=prm_model,
            prm_tokenizer=prm_tokenizer,
            alpha=train_config.reward_alpha,
            beta=train_config.reward_beta,
            num_wrong_stop=train_config.reward_num_wrong_stop,
            use_prefix_kv_cache=train_config.reward_use_prefix_kv_cache,
            max_length=train_config.max_length,
            _reward_detail=_reward_detail,
            **kwargs,
        )

    # 初始化 GRPO Trainer（ref_model 由 trainer 内部自动从 policy_model 复制）
    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[reward_func_bound],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=policy_tokenizer,   # 用于 apply_chat_template 构建 prompt
    )

    trainer.add_callback(metrics_callback)

    # ── Monkey-patch generate：训练时保留 gradient checkpointing 节省显存，
    #     但生成时临时关闭以恢复 KV cache，避免 O(n²) 生成开销。
    _original_generate = trainer.model.generate

    def _generate_with_cache(*args, **kwargs):
        was_gc = getattr(trainer.model, "is_gradient_checkpointing", False)
        if was_gc:
            trainer.model.gradient_checkpointing_disable()
        try:
            return _original_generate(*args, **kwargs)
        finally:
            if was_gc:
                trainer.model.gradient_checkpointing_enable()

    trainer.model.generate = _generate_with_cache

    # 开始训练
    trainer.train()

    # 保存 GRPO 训练好的 LoRA 权重
    trainer.save_model(SAVE_PATH)
    print(f"GRPO training completed. Model saved to: {SAVE_PATH}")