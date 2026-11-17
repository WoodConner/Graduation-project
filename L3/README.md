# L3: Action Recognition Module (ST-GCN)

## 毕设L3模块 - 完整实现

按照毕设计划书完整实现G1/G2/G3对比训练框架。

### 文件说明

- **stgcn.py**: 9层ST-GCN网络（完整架构）
- **dataset.py**: H36M预处理（根中心化、T=120统一、Z-score归一化）
- **gen_l2_poses.py**: L2推理脚本（生成L2输出3D序列用于G2/G3）
- **train.py**: G1/G2/G3对比训练框架

### 快速开始

#### 方式1: 用真值3D直接运行（G1基准）

```bash
cd /home/wood/long\ task/FMPose3D_PathA
PYTHONPATH=. python L3/train.py --group G1 --seeds 2 --epochs 80 --batch_size 32
```

预期输出：
- G1 Top-1 acc ~92-95%（真值3D天花板）
- results/ 目录下保存checkpoint和混淆矩阵

#### 方式2: 用L2推理输出运行（完整G1/G2/G3）

需要先生成L2输出序列（推理全H36M耗时~1小时）：

```bash
cd /home/wood/long\ task/FMPose3D_PathA

# Step 1: 生成L2输出 (可选，可用真值替代)
PYTHONPATH=. python L3/gen_l2_poses.py \\
    --ckpt_path checkpoint/261108_2203_17/CFM_10_4811_best.pth \\
    --out_path L3/l2_poses.npz \\
    --steps 5

# Step 2: 运行G1/G2/G3对比
PYTHONPATH=. python L3/train.py \\
    --l2_poses_path L3/l2_poses.npz \\
    --group all --seeds 2 --epochs 80
```

### 输出格式

L3/results/目录:
- `summary.json`: 所有组的最佳Top-1精度
- `G1_s0_best.pth`, `G2_s0_best.pth`, `G3_s0_best.pth`: 每组seed的最佳checkpoint
- `G1_s0_conf.npy`: 混淆矩阵
- `G1_s0_hist.json`: 每epoch的训练历史

### 核心设计

**预处理（严格按论文）**:
- 根关节零中心化：消除虚假空间偏置
- 均匀重采样T=120帧：保留完整动作周期
- Z-score归一化：统计量仅从训练集计算
- 跨subject划分：S1/5/6/7/8训练，S9/11测试

**训练配方（Recipe A）**:
- 优化器：SGD + Nesterov momentum 0.9
- 学习率：初始0.01，MultiStep衰减（epoch 10, 50各×0.1）
- 权重衰减：5e-4
- Dropout: 0.5
- Batch size: 32

**数据增强（克制策略）**:
- 随机y轴旋转 ±30度
- 随机缩放 0.9~1.1
- 左右翻转 p=0.5
- 仅训练集应用，验证集不应用

**G1/G2/G3定义**:
- G1：真值3D训练+真值3D测试（天花板）
- G2：L2输出训练+L2输出测试（真实部署）
- G3：50% GT + 50% L2混合训练，L2输出测试（容错补偿）

### 判决规则

```
Delta = A_G1 - A_G2 (百分比差)

if Delta <= 3%:
    L2定稿 -> 进入L4
elif Delta <= 5%:
    L2勉强可用
    if G3能缩小gap >50%:
        用G3补偿策略定稿
    else:
        L2需优化
else:
    L2需回炉，优先优化下肢关节误差
```

### 混淆矩阵分析

训练完成后自动输出：
- 逐类Top-1精度
- Top-5混淆对（哪两个动作最容易混淆）
- 标记精度<70%的动作（通常是下肢关键动作如Sitting/SittingDown）

### 性能参考

- 模型参数：~3.7M（标准ST-GCN）
- 推理延迟：~50ms/frame（GPU）
- G1（真值）预期精度：92-95%
- G2（L2输出）预期精度：85-90%（取决于L2 P1）
- G3（混合）预期精度：88-92%（取决于混合策略有效性）

### 常见问题

Q: 如何只跑G1不跑L2推理？
A: `python L3/train.py --group G1` 直接使用真值3D

Q: 推理时间太长怎么办？
A: 用真值3D替代（实际上数据泄漏会让G2精度虚高，但快速验证模型）

Q: 如何改变假设数量？
A: gen_l2_poses.py中 `euler_sample(num_hyp=8)` 参数

### 实验建议

1. 先跑G1验证数据管道无误
2. 再跑G2看L2误差的真实影响
3. 如果Gap>5%，用混淆矩阵定位问题类别
4. 根据下肢关节误差（Sitting类）反馈给L2优化方向

