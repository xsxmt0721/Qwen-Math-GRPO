import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 定义容器内的模型路径
# 注意：这里直接指向挂载后的绝对路径
model_path = "/models/qwen2.5-1.5b-prm"

def load_model():
    print(f"正在从 {model_path} 加载模型...")
    
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 3. 加载模型
    # 使用 device_map="auto" 自动分配显存，torch_dtype 建议设为 torch.bfloat16 (如果 GPU 支持)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    print(f"模型加载成功！device_map: {model.hf_device_map}")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
    # 简单测试一下
    print("模型配置:", model.config)