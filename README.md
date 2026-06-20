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
# 基准模型和训练模型的初始模型
mkdir qwen2.5-7b
hf download Qwen/Qwen2.5-7B-Instruct --local-dir qwen2.5-7b
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

- 运行以下脚本可以查看当前的数据集在传入模型时实际的序列长度分布
```
python -m scripts.get_length_distribution
```

## 训练奖励模型并评估
```
python -m reward.train
python -m reward.eval
```
## 训练基准参考模型并评估
```
python -m ref.train
python -m ref.eval
```

## 启动 GRPO 训练
```
python -m train
```

## 测试
```
python -m test
```

## 展示真实输出
```
python -m presentation
```