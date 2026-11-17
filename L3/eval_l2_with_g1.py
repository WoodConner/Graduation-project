#!/usr/bin/env python3
"""用G1 checkpoint直接评估L2测试集（不重训，5分钟拿结果）"""
import argparse, sys, logging
import numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from L3.dataset import H36M_ACTIONS, create_test_windows, center_root
from L3.stgcn import STGCN

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_l2_with_g1(g1_ckpt_path, l2_data_path, device='cuda'):
    """用G1的checkpoint直接评估L2测试集"""
    # 加载L2测试数据
    l2 = np.load(l2_data_path, allow_pickle=True)
    l2_seqs = list(l2['val_seqs'])
    l2_lbls = list(l2['val_labels'])
    logger.info(f'加载L2测试数据: {len(l2_seqs)} 个序列')
    
    # 创建中心窗口（与G1评估一致）
    windows, window_lbls = create_test_windows(l2_seqs, l2_lbls, window_len=120)
    logger.info(f'生成测试窗口: {len(windows)} 个（每序列1个中心窗口）')
    
    # 计算统计量（从窗口数据）
    all_centered = np.concatenate([center_root(w) for w in windows], axis=0)
    mean = all_centered.mean(axis=0, keepdims=True)
    std = all_centered.std(axis=0, keepdims=True) + 1e-8
    
    # 加载G1模型
    model = STGCN(num_classes=15, dropout=0.5).to(device)
    state = torch.load(g1_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f'加载G1 checkpoint: {g1_ckpt_path}')
    
    # 评估
    correct = 0
    conf = np.zeros((15, 15), dtype=int)
    for w, lbl in zip(windows, window_lbls):
        w = center_root(w.copy().astype(np.float32))
        w = (w - mean) / std
        t = torch.from_numpy(w).unsqueeze(0).to(device)
        pred = model(t).argmax(1).item()
        conf[lbl, pred] += 1
        if pred == lbl:
            correct += 1
    
    acc = correct / len(windows)
    logger.info(f'\n=== L2测试集评估结果（用G1权重） ===')
    logger.info(f'准确率: {acc:.4f} ({correct}/{len(windows)})')
    
    # 逐类精度
    logger.info('\n各动作精度:')
    for i, action in enumerate(H36M_ACTIONS):
        total = conf[i].sum()
        if total > 0:
            class_acc = conf[i, i] / total
            logger.info(f'  {action:15s}: {class_acc:.4f} ({conf[i,i]}/{total})')
    
    return acc, conf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--g1_ckpt', default='L3/results/G1_s0_best.pth')
    p.add_argument('--l2_data', default='L3/l2_poses_test_fixed_v5.npz')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    acc, conf = evaluate_l2_with_g1(args.g1_ckpt, args.l2_data, args.device)
    
    # 判据
    logger.info('\n=== 判据 ===')
    if acc >= 0.35:
        logger.info(f'✅ {acc:.2%} ≥ 35%: L2输出可用，Δ ≤ 7%')
        logger.info('   结论: L3判决完成，L2定稿进L4')
    elif acc >= 0.25:
        logger.info(f'⚠️  {acc:.2%} 在25-35%: L2需优化，但接口已通')
        logger.info('   结论: L3框架完整可写入报告，L2作为已知限制')
    else:
        logger.info(f'❌ {acc:.2%} < 25%: L2输出存在结构性问题')
        logger.info('   建议: 运行Procrustes诊断检查朝向/尺度')

if __name__ == '__main__':
    main()
