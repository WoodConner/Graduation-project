"""
对比修复前后的L2输出质量
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from test_gt_vs_l2 import load_gt_test, compute_score_differences

print("="*70)
print("L2修复效果对比")
print("="*70)

# 加载GT数据
gt_seqs, gt_labels, gt_action_names = load_gt_test()
print(f"\n加载了{len(gt_seqs)}个GT测试序列\n")

# 对比1: 修复前的L2
print("【修复前】l2_poses_full.npz")
try:
    l2_old = np.load('/home/wood/long task/FMPose3D_PathA/L3/l2_poses_full.npz', allow_pickle=True)
    l2_old_seqs = l2_old['val_seqs']
    l2_old_labels = l2_old['val_labels']
    
    results_old = compute_score_differences(gt_seqs, l2_old_seqs, gt_labels, l2_old_labels, gt_action_names)
    print(f"平均绝对偏差: {results_old['mean_delta']:.2f}分")
    print(f"Spearman ρ: {results_old['spearman_rho']:.3f}")
    print(f"最大偏差: {results_old['max_delta']:.2f}分\n")
except FileNotFoundError:
    print("未找到，跳过\n")

# 对比2: 修复后的L2  
print("【修复后】l2_poses_test_fixed.npz")
try:
    l2_new = np.load('/home/wood/long task/FMPose3D_PathA/L3/l2_poses_test_fixed.npz', allow_pickle=True)
    l2_new_seqs = l2_new['val_seqs']
    l2_new_labels = l2_new['val_labels']
    
    results_new = compute_score_differences(gt_seqs, l2_new_seqs, gt_labels, l2_new_labels, gt_action_names)
    print(f"平均绝对偏差: {results_new['mean_delta']:.2f}分")
    print(f"Spearman ρ: {results_new['spearman_rho']:.3f}")
    print(f"最大偏差: {results_new['max_delta']:.2f}分\n")
    
    # 改进量
    if 'results_old' in locals():
        print("="*70)
        print("改进效果")
        print("="*70)
        print(f"平均偏差: {results_old['mean_delta']:.2f} → {results_new['mean_delta']:.2f} (改善 {results_old['mean_delta']-results_new['mean_delta']:.2f}分)")
        print(f"Spearman ρ: {results_old['spearman_rho']:.3f} → {results_new['spearman_rho']:.3f} (改善 {results_new['spearman_rho']-results_old['spearman_rho']:.3f})")
        print(f"最大偏差: {results_old['max_delta']:.2f} → {results_new['max_delta']:.2f} (改善 {results_old['max_delta']-results_new['max_delta']:.2f}分)")
        
        if results_new['mean_delta'] < 10:
            print("\n✅ 修复成功！平均偏差<10分，L4可以使用L2输出")
        elif results_new['spearman_rho'] > 0.8:
            print("\n⚠️  部分成功：排名一致性高(ρ>0.8)，但需要线性校准")
        else:
            print("\n❌ 修复不足，需要进一步调查")
            
except FileNotFoundError:
    print("文件不存在，等待生成完成\n")

print("="*70)
