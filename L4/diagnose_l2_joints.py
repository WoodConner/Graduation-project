"""
诊断L2输出的关节顺序问题
检查L2和GT的关节对应关系
"""
import numpy as np
import sys
from pathlib import Path

# H36M 17关节顺序
H36M_17_JOINTS = [
    'Hip',           # 0
    'RHip',          # 1
    'RKnee',         # 2
    'RAnkle',        # 3
    'LHip',          # 4
    'LKnee',         # 5
    'LAnkle',        # 6
    'Spine',         # 7
    'Thorax',        # 8
    'Neck/Nose',     # 9
    'Head',          # 10
    'LShoulder',     # 11
    'LElbow',        # 12
    'LWrist',        # 13
    'RShoulder',     # 14
    'RElbow',        # 15
    'RWrist',        # 16
]

print("="*70)
print("诊断L2关节顺序问题")
print("="*70)

# 加载GT数据
GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
gt_data = np.load('/home/wood/long task/FMPose3D_PathA/dataset/data_3d_h36m.npz', allow_pickle=True)
gt_positions = gt_data['positions_3d'].item()

# 选择一个Walking序列
walking_actions = [(k, v) for k, v in gt_positions['S9'].items() if 'Walking' in k]
gt_seq_32 = walking_actions[0][1]  # (T, 32, 3)
gt_seq = gt_seq_32[:, GT_32_TO_17, :]  # (T, 17, 3)
gt_seq = gt_seq - gt_seq[:, 0:1, :]  # Hip归零

print(f"GT序列形状: {gt_seq.shape}")

# 加载L2数据
l2_data = np.load('/home/wood/long task/FMPose3D_PathA/L3/l2_poses_full.npz', allow_pickle=True)
l2_seqs = l2_data['val_seqs']
l2_labels = l2_data['val_labels']

# 找到对应的Walking序列（标签13）
walking_idx = np.where(l2_labels == 13)[0][0]
l2_seq = l2_seqs[walking_idx]
l2_seq = l2_seq - l2_seq[:, 0:1, :]  # Hip归零

print(f"\n序列长度: GT={len(gt_seq)}, L2={len(l2_seq)}")
print(f"GT形状: {gt_seq.shape}, L2形状: {l2_seq.shape}")
print(f"选择Walking动作进行分析\n")

# 取第一帧进行分析
frame_idx = min(100, len(gt_seq)-1, len(l2_seq)-1)
gt_frame = gt_seq[frame_idx]  # (17, 3)
l2_frame = l2_seq[frame_idx]  # (17, 3)

print("【关节位置对比】(第100帧)")
print(f"{'关节':<15} {'GT (x,y,z)':<30} {'L2 (x,y,z)':<30} {'欧氏距离'}")
print("-"*70)

distances = []
for i, joint_name in enumerate(H36M_17_JOINTS):
    gt_pos = gt_frame[i]
    l2_pos = l2_frame[i]
    dist = np.linalg.norm(gt_pos - l2_pos)
    distances.append(dist)
    print(f"{joint_name:<15} ({gt_pos[0]:6.1f},{gt_pos[1]:6.1f},{gt_pos[2]:6.1f})  ({l2_pos[0]:6.1f},{l2_pos[1]:6.1f},{l2_pos[2]:6.1f})  {dist:6.1f}mm")

print(f"\n平均偏差: {np.mean(distances):.1f}mm")
print(f"中位数偏差: {np.median(distances):.1f}mm")

# 计算骨长
print(f"\n{'='*70}")
print("【骨长对比】")
print(f"{'='*70}")

bone_pairs = [
    ('RHip-RKnee', 1, 2),
    ('RKnee-RAnkle', 2, 3),
    ('LHip-LKnee', 4, 5),
    ('LKnee-LAnkle', 5, 6),
    ('LShoulder-LElbow', 11, 12),
    ('LElbow-LWrist', 12, 13),
    ('RShoulder-RElbow', 14, 15),
    ('RElbow-RWrist', 15, 16),
]

print(f"{'骨骼':<20} {'GT长度':<12} {'L2长度':<12} {'比率':<10}")
print("-"*70)

for bone_name, j1, j2 in bone_pairs:
    gt_len = np.linalg.norm(gt_frame[j1] - gt_frame[j2])
    l2_len = np.linalg.norm(l2_frame[j1] - l2_frame[j2])
    ratio = l2_len / (gt_len + 1e-8)
    print(f"{bone_name:<20} {gt_len:<12.1f} {l2_len:<12.1f} {ratio:<10.2f}")

# 检查左右对称性
print(f"\n{'='*70}")
print("【左右对称性检查】")
print(f"{'='*70}")

symmetric_pairs = [
    ('Hip', 1, 4),  # RHip vs LHip
    ('Knee', 2, 5),
    ('Ankle', 3, 6),
    ('Shoulder', 14, 11),
    ('Elbow', 15, 12),
    ('Wrist', 16, 13),
]

print(f"{'关节对':<15} {'GT对称性':<15} {'L2对称性':<15}")
print("-"*70)

for pair_name, r_idx, l_idx in symmetric_pairs:
    # GT对称性：左右关节到hip的距离应该相近
    gt_r_dist = np.linalg.norm(gt_frame[r_idx] - gt_frame[0])
    gt_l_dist = np.linalg.norm(gt_frame[l_idx] - gt_frame[0])
    gt_sym = abs(gt_r_dist - gt_l_dist)
    
    l2_r_dist = np.linalg.norm(l2_frame[r_idx] - l2_frame[0])
    l2_l_dist = np.linalg.norm(l2_frame[l_idx] - l2_frame[0])
    l2_sym = abs(l2_r_dist - l2_l_dist)
    
    print(f"{pair_name:<15} {gt_sym:<15.1f} {l2_sym:<15.1f}")

# 尝试检测是否有关节交换
print(f"\n{'='*70}")
print("【关节交换检测】")
print(f"{'='*70}")
print("检查L2的每个关节与GT的哪个关节最接近...")

l2_to_gt_mapping = []
for i in range(17):
    l2_joint = l2_frame[i]
    min_dist = float('inf')
    best_match = -1
    for j in range(17):
        gt_joint = gt_frame[j]
        dist = np.linalg.norm(l2_joint - gt_joint)
        if dist < min_dist:
            min_dist = dist
            best_match = j
    l2_to_gt_mapping.append((i, best_match, min_dist))
    if i != best_match:
        print(f"⚠️  L2[{i}]({H36M_17_JOINTS[i]}) -> GT[{best_match}]({H36M_17_JOINTS[best_match]}) dist={min_dist:.1f}mm")

# 检查是否有系统性的镜像或旋转
print(f"\n{'='*70}")
print("【坐标系检查】")
print(f"{'='*70}")

# 计算GT和L2的主轴方向
gt_spine = gt_frame[10] - gt_frame[0]  # Head - Hip
l2_spine = l2_frame[10] - l2_frame[0]

gt_right = gt_frame[1] - gt_frame[0]  # RHip - Hip
l2_right = l2_frame[1] - l2_frame[0]

print(f"GT脊柱方向: {gt_spine}")
print(f"L2脊柱方向: {l2_spine}")
print(f"GT右侧方向: {gt_right}")
print(f"L2右侧方向: {l2_right}")

# 检查是否有镜像
spine_dot = np.dot(gt_spine / (np.linalg.norm(gt_spine) + 1e-8), 
                   l2_spine / (np.linalg.norm(l2_spine) + 1e-8))
right_dot = np.dot(gt_right / (np.linalg.norm(gt_right) + 1e-8),
                   l2_right / (np.linalg.norm(l2_right) + 1e-8))

print(f"\n脊柱方向余弦相似度: {spine_dot:.3f} (1.0=完全一致, -1.0=反向)")
print(f"右侧方向余弦相似度: {right_dot:.3f}")

if spine_dot < -0.5:
    print("⚠️  检测到上下翻转！")
if right_dot < -0.5:
    print("⚠️  检测到左右镜像！")

print(f"\n{'='*70}")
print("诊断完成")
print(f"{'='*70}")
