# 📊 最终实验结果

本目录包含毕业设计的所有关键实验结果和模型文件。

## 📁 文件列表

### L2: 3D姿态估计
- `L2_pathA_best.pth` (51.6 MB)
  - Path A最优模型（Epoch 10）
  - MPJPE P1 = 48.11mm
  - MPJPE P2 = 38.68mm
  - 训练配置：lr=1e-3, batch=512, epochs=10

- `L2_test_v8.npz` (108 MB)
  - Human3.6M测试集（S9/S11）的L2输出
  - 59个动作序列，每个序列包含120-2357帧
  - 坐标系：world坐标系，root-relative
  - Procrustes disparity vs GT: 0.0029

### L3: 动作识别
- `L3_G1_best.pth` (696 KB)
  - G1组最优模型（GT数据训练和测试）
  - 测试精度：41.81% ± 2.11%
  - 架构：9层ST-GCN，通道[48, 96, 192]

- `L3_G2_training.log` (2.2 KB)
  - G2组训练日志（GT训练，L2测试）
  - 最终测试精度：42.37%
  - 性能差异 Δ = G1 - G2 = -0.56%
  - **判决：L2输出完全够用**

### L4: 规范性评分
- `L4_gt_vs_l2_final.npz` (18 KB)
  - GT vs L2评分对比完整结果
  - 包含15类动作的逐样本评分
  - 平均绝对偏差：1.17分
  - Spearman相关系数 ρ：0.981
  - p-value：< 0.0001

## 🔬 实验复现

### L2推理
```bash
cd '/home/wood/long task/FMPose3D_PathA'

# 加载模型
model = CFM(...)
model.load_state_dict(torch.load('results_final/L2_pathA_best.pth'))

# 生成测试集输出
PYTHONPATH=. python L3/gen_l2_poses_fixed.py \
  --checkpoint results_final/L2_pathA_best.pth \
  --test_only --steps 5 --num_hyp 4 \
  --out_path L3/l2_poses_v8.npz
```

### L3动作识别
```bash
cd L3

# G1: GT数据评估
python train.py --group G1 --eval_only \
  --checkpoint ../results_final/L3_G1_best.pth

# G2: L2数据评估
python train.py --group G2 --eval_only \
  --checkpoint ../results_final/L3_G1_best.pth \
  --l2_poses_path ../results_final/L2_test_v8.npz
```

### L4评分对比
```bash
cd L4

# 创建软链接
ln -sf ../results_final/L2_test_v8.npz L3/l2_poses_full.npz

# 运行对比实验
python test_gt_vs_l2.py

# 加载结果
import numpy as np
result = np.load('results_final/L4_gt_vs_l2_final.npz', allow_pickle=True)
print(f"平均偏差: {result['mean_deviation']:.2f} 分")
print(f"Spearman ρ: {result['spearman_rho']:.3f}")
```

## 📈 关键指标汇总

### L2性能
| 指标 | 数值 |
|------|------|
| P1 (MPJPE) | 48.11 mm |
| P2 (P-MPJPE) | 38.68 mm |
| Procrustes disparity | 0.0029 |
| 参数量 | ~10M |

### L3性能
| 组别 | 测试数据 | 精度 | Δ (vs G1) |
|------|----------|------|-----------|
| G1 | GT | 41.81% ± 2.11% | - |
| G2 | L2 (v8) | 42.37% | -0.56% |

### L4性能
| 指标 | 数值 |
|------|------|
| 平均绝对偏差 | 1.17 分 |
| 中位数偏差 | 0.90 分 |
| 最大偏差 | 4.58 分 |
| Spearman ρ | 0.981 |
| p-value | < 0.0001 |

## 🎯 结论

1. **L2达标**: P1=48.11mm接近baseline 47.3mm
2. **L2→L3接口通畅**: Procrustes=0.0029（完美对齐），Δ=-0.56%（性能等价）
3. **L2→L4接口通畅**: 偏差1.17分（可忽略），ρ=0.981（排名一致）
4. **系统完整性**: 首次实现L2→L3→L4完整链路的端到端验证

---

**注意**: 由于文件大小限制，模型文件(.pth)和数据文件(.npz)未上传到GitHub。如需复现实验，请联系作者获取完整数据集和模型。

**联系方式**: wood@example.com
