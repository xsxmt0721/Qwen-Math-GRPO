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

## 获取模型文件和权重
```
cd MathRL/Models
# 过程奖励模型
mkdir qwen2.5-1.5b-prm
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir qwen2.5-1.5b-prm
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

## 训练奖励模型
```
python -m reward.train
```