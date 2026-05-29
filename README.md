# 改良 GRPO 算法微调模型：数学推理任务

## 训练数据集导入

```
cd Core
conda create -n DataLoader python=3.12
conda activate DataLoader
python DataLoader/GMS8K.py --save-dir ~/MathRL/Data/GSM8K --force
python DataLoader/MATH.py --save-dir ~/MathRL/Data/MATH --force
python DataLoader/CMATH.py --save-dir ~/MathRL/Data/CMATH --force
python DataLoader/DEEPMATH-103K.py --save-dir ~/MathRL/Data/DEEPMATH-103K --force
```

## 初始化 PRM 模型和预训练权重导入
```
python -m DataLoader.load_prm --pretrained-dir ~/MathRL/Models/prm/pretrained --model-id Qwen/Qwen2.5-1.5B --torch-dtype bf16
```

## 数据划分

- 进入容器
```
docker exec -it mathrl /bin/bash
```
- 运行以下脚本
```
python scripts/data_split.py --data GSM8K MATH DEEPMATH-103K --keep-test
```