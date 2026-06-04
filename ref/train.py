import torch
import os
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

from utils.dataloader import DataLoader

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

env = _load_env(Path(__file__).parent.parent / ".env")

DATA_DIR: str = env.get("DATA_DIR", "/data")
SPLIT_DIR: str = env.get("DATA_SPLIT_DIR", "/data_split")

MODEL_PATH: str = env.get("BASEMODEL_DIR", "")
SAVE_PATH: str = env.get("REF_SAVE", "")

class train_config:
    log_dir = "/workspace/logs/ref_logs/ref-1.5b"
    
    cuda_use: bool = True
    batch_size: int = 2
    eval_batch_size: int = 1
    epochs: int = 3
    learning_rate = 2e-4
    weight_decay = 0.0
    max_grad_norm = 1.0
    grad_accum_steps = 4
    max_length = 1024

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    log_step = 5

    seed = 42
    shuffle: bool = True
    transform: bool = True
    transform_all_ans: bool = False


if __name__ == "__main__":
    # 导入数据
    train_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="SFT_TRAIN",
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
        transform=train_config.transform,
        transform_all_ans=train_config.transform_all_ans,
        seed=train_config.seed,
    )
    val_loader = DataLoader(
        data_dir=DATA_DIR,
        split_dir=SPLIT_DIR,
        data_type="SFT_VAL",
        batch_size=train_config.eval_batch_size,
        shuffle=train_config.shuffle,
        transform=train_config.transform,
        transform_all_ans=train_config.transform_all_ans,
        seed=train_config.seed,
    )
    
    train_dataset = Dataset.from_list(train_loader.data)
    train_answers = train_loader.ans
    val_dataset = Dataset.from_list(val_loader.data)
    val_answers = val_loader.ans

    # 加载模型
    compute_dtype = torch.bfloat16 # A5000 支持 BF16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_args = SFTConfig(
        output_dir=SAVE_PATH,
        logging_dir=train_config.log_dir,
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=train_config.log_step,
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.grad_accum_steps,
        per_device_eval_batch_size=train_config.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=train_config.max_length,
        packing=True,
        dataset_kwargs={"add_special_tokens": False},
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()