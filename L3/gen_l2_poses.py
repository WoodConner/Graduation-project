"""
Generate L2 output 3D poses using batched inference + FP16.
Phase 1 (fast): inference on test subjects only (S9, S11) for quick G2-test-only.
Phase 2 (full): inference on all subjects for full G2.
"""
import argparse, sys, logging
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from L3.dataset import H36M_ACTIONS, H36M_TRAIN_SUBJECTS, H36M_TEST_SUBJECTS

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model(ckpt_path, device='cuda'):
    from fmpose3d.models.base_model import get_model
    class A:
        channel=512; d_hid=1024; token_dim=256; layers=5; n_joints=17
        use_kan_gcn=0; kan_blocks=1; kan_grid_size=5; kan_spline_order=3
    model = get_model('fmpose3d_humans')(A()).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ('model_pos', 'state_dict', None):
        try:
            model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
            logger.info(f'Loaded checkpoint (key={key})')
            break
        except (KeyError, TypeError):
            continue
    return model


@torch.no_grad()
def batched_euler_sample_with_tta(model, frames_2d, steps=5, num_hyp=4, chunk=128, device='cuda'):
    """
    Batched inference with TTA (Test-Time Augmentation): (T, 17, 2) -> (T, 17, 3)
    Replicates demo/vis_in_the_wild.py's flip augmentation strategy.
    num_hyp=4 gives speedup over 8 with minimal P1 degradation (~+0.5mm).
    """
    from fmpose3d.common.camera import camera_to_world
    
    # H36M joint indices for left/right symmetry
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    
    T = frames_2d.shape[0]
    dt = 1.0 / steps
    
    # === Part 1: Non-flipped inference ===
    all_preds_non_flip = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        batch_2d = torch.from_numpy(frames_2d[start:end]).to(device)  # (B, 17, 2)
        B = batch_2d.shape[0]
        x2d = batch_2d.unsqueeze(1).expand(B, num_hyp, 17, 2).reshape(B * num_hyp, 1, 17, 2)
        
        x3d = torch.randn(B * num_hyp, 1, 17, 3, device=device)
        with torch.cuda.amp.autocast():
            for s in range(steps):
                t_s = torch.full((B * num_hyp, 1, 1, 1), s * dt, device=device)
                v = model(x2d, x3d, t_s)
                x3d = x3d + dt * v
        
        x3d = x3d.float().squeeze(1).reshape(B, num_hyp, 17, 3).mean(dim=1)
        all_preds_non_flip.append(x3d.cpu().numpy())
    
    output_non_flip = np.concatenate(all_preds_non_flip, axis=0)  # (T, 17, 3)
    
    # === Part 2: Flipped inference ===
    frames_2d_flip = frames_2d.copy()
    frames_2d_flip[:, :, 0] *= -1  # Flip x-coordinate
    frames_2d_flip[:, joints_left + joints_right] = frames_2d_flip[:, joints_right + joints_left]
    
    all_preds_flip = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        batch_2d = torch.from_numpy(frames_2d_flip[start:end]).to(device)
        B = batch_2d.shape[0]
        x2d = batch_2d.unsqueeze(1).expand(B, num_hyp, 17, 2).reshape(B * num_hyp, 1, 17, 2)
        
        x3d = torch.randn(B * num_hyp, 1, 17, 3, device=device)
        with torch.cuda.amp.autocast():
            for s in range(steps):
                t_s = torch.full((B * num_hyp, 1, 1, 1), s * dt, device=device)
                v = model(x2d, x3d, t_s)
                x3d = x3d + dt * v
        
        x3d = x3d.float().squeeze(1).reshape(B, num_hyp, 17, 3).mean(dim=1)
        all_preds_flip.append(x3d.cpu().numpy())
    
    output_flip = np.concatenate(all_preds_flip, axis=0)  # (T, 17, 3)
    
    # Unflip the 3D output
    output_flip[:, :, 0] *= -1  # Flip x back
    output_flip[:, joints_left + joints_right] = output_flip[:, joints_right + joints_left]
    
    # === Part 3: Average two predictions (TTA fusion) ===
    output_3d = (output_non_flip + output_flip) / 2
    
    # === Part 4: Post-processing (matching demo) ===
    # Root joint to origin
    output_3d[:, 0, :] = 0
    
    # Camera to world transformation (demo line 241)
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype=np.float32)
    output_3d = camera_to_world(output_3d, R=rot, t=0)
    
    # Z-axis alignment
    output_3d[:, 2] -= np.min(output_3d[:, 2])
    
    return output_3d


def gen_sequences(model, det_positions, subjects, gt_positions, steps, num_hyp, chunk, device):
    seqs, labels = [], []
    total_actions = sum(len(gt_positions[f'S{s}']) for s in subjects if f'S{s}' in gt_positions)
    done = 0
    for subj in subjects:
        key = f'S{subj}'
        if key not in gt_positions:
            continue
        for action_name in gt_positions[key]:
            base = action_name.split(' ')[0]
            if base not in H36M_ACTIONS:
                continue
            label = H36M_ACTIONS.index(base)
            det_subj = det_positions.get(key, {})
            if action_name not in det_subj:
                continue
            seq_2d = det_subj[action_name]
            # Data format: list of 4 camera views, each (T, 17, 2)
            # Pick camera view 0 (consistent with baseline default)
            if isinstance(seq_2d, (list, tuple)):
                seq_2d = seq_2d[0]  # take camera view 0
            if isinstance(seq_2d, np.ndarray) and seq_2d.dtype != np.float32:
                seq_2d = seq_2d.astype(np.float32)
            elif not isinstance(seq_2d, np.ndarray):
                seq_2d = np.array(seq_2d, dtype=np.float32)
            # Batched inference with TTA
            pred_3d = batched_euler_sample_with_tta(model, seq_2d, steps=steps,
                                                     num_hyp=num_hyp, chunk=chunk, device=device)
            
            # **FIXED**: Keep full sequence length for sliding window in training
            # Do NOT truncate to 120 frames - let train_pairwise_complete.py handle windowing
            
            seqs.append(pred_3d)
            labels.append(label)
            done += 1
            if done % 10 == 0:
                logger.info(f'  {done}/{total_actions} actions done')
    logger.info(f'  Generated {len(seqs)} seqs from subjects {subjects}')
    return seqs, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt_path',    default='dataset/data_3d_h36m.npz')
    p.add_argument('--det2d_path', default='dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz')
    p.add_argument('--ckpt_path',  default='checkpoint/261108_2203_17/CFM_10_4811_best.pth')
    p.add_argument('--out_path',   default='L3/l2_poses.npz')
    p.add_argument('--steps',      type=int, default=5)
    p.add_argument('--num_hyp',    type=int, default=4,  help='Hypotheses (4 for speed, 8 for quality)')
    p.add_argument('--chunk',      type=int, default=128, help='Batch size for inference')
    p.add_argument('--test_only',  action='store_true', help='Only infer test subjects (S9/S11), use GT for train')
    p.add_argument('--device',     default='cuda')
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device={device} steps={args.steps} num_hyp={args.num_hyp} chunk={args.chunk}')

    gt_data  = np.load(args.gt_path, allow_pickle=True)
    gt_pos   = gt_data['positions_3d'].item()
    det_data = np.load(args.det2d_path, allow_pickle=True)
    det_pos  = det_data['positions_2d'].item()

    model = load_model(args.ckpt_path, device)

    if args.test_only:
        # Phase 1: test-only fast version (G2-test-only)
        logger.info('Phase 1 (test-only): inferring S9/S11 ...')
        va_seqs, va_lbls = gen_sequences(model, det_pos, H36M_TEST_SUBJECTS,
                                         gt_pos, args.steps, args.num_hyp, args.chunk, device)
        # Training still uses GT 3D (passed as None to signal G2-test-only)
        logger.info('Loading GT 3D for train sequences (G2-test-only mode)...')
        from L3.train import load_h36m_sequences
        tr_seqs, tr_lbls = load_h36m_sequences(args.gt_path, H36M_TRAIN_SUBJECTS)
        logger.info(f'G2-test-only: train=GT({len(tr_seqs)}), val=L2({len(va_seqs)})')
    else:
        # Phase 2: full G2 (both train and test use L2 output)
        logger.info('Phase 2 (full): inferring all subjects ...')
        logger.info('Generating training L2 poses (S1,S5,S6,S7,S8)...')
        tr_seqs, tr_lbls = gen_sequences(model, det_pos, H36M_TRAIN_SUBJECTS,
                                         gt_pos, args.steps, args.num_hyp, args.chunk, device)
        logger.info('Generating validation L2 poses (S9,S11)...')
        va_seqs, va_lbls = gen_sequences(model, det_pos, H36M_TEST_SUBJECTS,
                                         gt_pos, args.steps, args.num_hyp, args.chunk, device)

    np.savez(args.out_path,
             train_seqs=np.array(tr_seqs, dtype=object),
             train_labels=np.array(tr_lbls),
             val_seqs=np.array(va_seqs, dtype=object),
             val_labels=np.array(va_lbls))
    logger.info(f'Saved to {args.out_path} | train={len(tr_seqs)} val={len(va_seqs)}')


if __name__ == '__main__':
    main()
