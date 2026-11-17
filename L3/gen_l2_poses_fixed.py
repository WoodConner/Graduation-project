#!/usr/bin/env python3
"""Generate L2 output with FIXED camera-to-world transformation"""
import argparse, sys, logging
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from L3.dataset import H36M_ACTIONS, H36M_TRAIN_SUBJECTS, H36M_TEST_SUBJECTS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

def load_model(ckpt_path, device="cuda"):
    from fmpose3d.models.base_model import get_model
    class A:
        channel=512; d_hid=1024; token_dim=256; layers=5; n_joints=17
        use_kan_gcn=0; kan_blocks=1; kan_grid_size=5; kan_spline_order=3
    model = get_model("fmpose3d_humans")(A()).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("model_pos", "state_dict", None):
        try:
            model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
            logger.info(f"Loaded checkpoint (key={key})")
            break
        except (KeyError, TypeError):
            continue
    return model

H36M_CAM_INTRINSIC = [
    {"res_w": 1000, "res_h": 1002},  # cam0
    {"res_w": 1000, "res_h": 1000},  # cam1
    {"res_w": 1000, "res_h": 1000},  # cam2
    {"res_w": 1000, "res_h": 1002},  # cam3
]

def load_cameras():
    """Load H36M camera parameters from standard definitions"""
    # Standard H36M extrinsic parameters (from h36m_dataset.py)
    h36m_cameras = {
        "S9": [
            {
                "orientation": [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
                "translation": [2044.45849609375, 4935.1171875, 1481.2275390625],
            },
            {
                "orientation": [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
                "translation": [1990.959716796875, -5123.810546875, 1568.8048095703125],
            },
            {
                "orientation": [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
                "translation": [-1670.9921875, 5211.98583984375, 1528.387939453125],
            },
            {
                "orientation": [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
                "translation": [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
            },
        ],
        "S11": [
            {
                "orientation": [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
                "translation": [2098.440185546875, 4926.5546875, 1500.278564453125],
            },
            {
                "orientation": [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
                "translation": [2083.182373046875, -4912.1728515625, 1561.07861328125],
            },
            {
                "orientation": [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
                "translation": [-1609.8153076171875, 5177.3359375, 1537.896728515625],
            },
            {
                "orientation": [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
                "translation": [-1590.738037109375, -3854.1689453125, 1578.017578125],
            },
        ],
    }
    
    # Convert to numpy arrays and organize by subject then camera index
    cameras = {}
    for subj, cams in h36m_cameras.items():
        cameras[subj] = {}
        for cam_idx, cam in enumerate(cams):
            cameras[subj][cam_idx] = {
                "res_w": H36M_CAM_INTRINSIC[cam_idx]["res_w"],
                "res_h": H36M_CAM_INTRINSIC[cam_idx]["res_h"],
                'orientation': np.array(cam['orientation'], dtype=np.float32),
                'translation': np.array(cam['translation'], dtype=np.float32),
            }
    
    logger.info(f"Loaded cameras for {len(cameras)} subjects (S9, S11)")
    return cameras

@torch.no_grad()
def batched_euler_sample_with_tta(model, frames_2d, camera_params, steps=5, num_hyp=4, chunk=128, device="cuda"):
    from fmpose3d.common.camera import camera_to_world, normalize_screen_coordinates
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    
    # CRITICAL FIX: Normalize 2D coordinates (models are trained on normalized coords)
    frames_2d = normalize_screen_coordinates(frames_2d, w=camera_params["res_w"], h=camera_params["res_h"])
    # Fix dtype: normalize_screen_coordinates returns float64, but model expects float32
    if not isinstance(frames_2d, np.ndarray):
        frames_2d = np.array(frames_2d, dtype=np.float32)
    elif frames_2d.dtype != np.float32:
        frames_2d = frames_2d.astype(np.float32)
    
    T = frames_2d.shape[0]
    dt = 1.0 / steps
    
    # Non-flipped
    all_preds_non_flip = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        batch_2d = torch.from_numpy(frames_2d[start:end]).to(device)
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
    output_non_flip = np.concatenate(all_preds_non_flip, axis=0)
    
    # Flipped
    frames_2d_flip = frames_2d.copy()
    frames_2d_flip[:, :, 0] *= -1
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
    output_flip = np.concatenate(all_preds_flip, axis=0)
    output_flip[:, :, 0] *= -1
    output_flip[:, joints_left + joints_right] = output_flip[:, joints_right + joints_left]
    
    output_cam = (output_non_flip + output_flip) / 2.0
    
    # FIXED v8: 模型输出不是root-relative的，需要先强制根关节为零（参考demo第236行）
    output_cam[:, 0, :] = 0  # 强制根关节为零
    
    # orientation是四元数(w,x,y,z)，转换为旋转矩阵R_cw (camera-to-world)
    q = camera_params["orientation"]
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], dtype=np.float32)
    
    # Root-relative变换: p_world_rel = R @ p_cam_rel (t=0，参考demo第241行)
    # output_cam: (T,17,3), R: (3,3) -> einsum('ij,tkj->tki')实现批量R@p
    output_world_rel = np.einsum('ij,tkj->tki', R, output_cam)
    
    # 确保root-relative (防止数值误差)
    output_world_rel = output_world_rel - output_world_rel[:, 0:1, :]
    return output_world_rel

def gen_sequences(model, det_positions, subjects, gt_positions, cameras, steps, num_hyp, chunk, device):
    seqs, labels = [], []
    total = sum(len(gt_positions[f"S{s}"]) for s in subjects if f"S{s}" in gt_positions)
    done = 0
    for subj in subjects:
        subj_key = f"S{subj}"
        if subj_key not in gt_positions or subj_key not in det_positions:
            continue
        for action_key in gt_positions[subj_key].keys():
            # Extract action name (e.g., "Discussion 2" -> "Discussion")
            action_name = action_key.split()[0]
            label = H36M_ACTIONS.index(action_name) if action_name in H36M_ACTIONS else -1
            if label == -1:
                continue
            
            # CRITICAL FIX: seq_2d is a list of 4 camera views, use camera 0 (baseline default)
            if action_key not in det_positions[subj_key]:
                logger.warning(f"Skip {subj_key} {action_key}: not in 2D detections")
                continue
            seq_2d_list = det_positions[subj_key][action_key]
            if not isinstance(seq_2d_list, (list, tuple)) or len(seq_2d_list) == 0:
                logger.warning(f"Skip {subj_key} {action_key}: invalid 2D format")
                continue
            
            # Use camera 0 (consistent with baseline and most H36M protocols)
            cam_idx = 0
            seq_2d = seq_2d_list[cam_idx]  # (T, 17, 2) ndarray
            if not isinstance(seq_2d, np.ndarray):
                seq_2d = np.array(seq_2d, dtype=np.float32)
            
            if subj_key not in cameras or cam_idx not in cameras[subj_key]:
                logger.warning(f"Skip {subj_key} {action_key}: camera {cam_idx} missing")
                continue
            camera_params = cameras[subj_key][cam_idx]
            
            pred_3d = batched_euler_sample_with_tta(model, seq_2d, camera_params, steps, num_hyp, chunk, device)
            seqs.append(pred_3d)
            labels.append(label)
            done += 1
            if done % 10 == 0:
                logger.info(f"  {done}/{total} done")
    logger.info(f"Generated {len(seqs)} seqs from subjects {subjects}")
    return seqs, labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path", default="dataset/data_3d_h36m.npz")
    p.add_argument("--det2d_path", default="dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz")
    p.add_argument("--ckpt_path", default="checkpoint/261108_2203_17/CFM_10_4811_best.pth")
    p.add_argument("--out_path", default="L3/l2_poses_fixed.npz")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num_hyp", type=int, default=4)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Device={device} steps={args.steps} num_hyp={args.num_hyp}")
    logger.info("FIXED: Using per-camera transformation parameters")
    
    gt_data = np.load(args.gt_path, allow_pickle=True)
    gt_pos = gt_data["positions_3d"].item()
    det_data = np.load(args.det2d_path, allow_pickle=True)
    det_pos = det_data["positions_2d"].item()
    cameras = load_cameras()  # FIXED: No longer needs dataset_path
    model = load_model(args.ckpt_path, device)
    
    if args.test_only:
        logger.info("Test-only mode: S9/S11")
        va_seqs, va_lbls = gen_sequences(model, det_pos, H36M_TEST_SUBJECTS, gt_pos, cameras, args.steps, args.num_hyp, args.chunk, device)
        from L3.train import load_h36m_sequences
        tr_seqs, tr_lbls = load_h36m_sequences(args.gt_path, H36M_TRAIN_SUBJECTS)
        logger.info(f"Test-only: train=GT({len(tr_seqs)}), val=L2({len(va_seqs)})")
    else:
        logger.info("Full mode: all subjects")
        tr_seqs, tr_lbls = gen_sequences(model, det_pos, H36M_TRAIN_SUBJECTS, gt_pos, cameras, args.steps, args.num_hyp, args.chunk, device)
        va_seqs, va_lbls = gen_sequences(model, det_pos, H36M_TEST_SUBJECTS, gt_pos, cameras, args.steps, args.num_hyp, args.chunk, device)
    
    np.savez(args.out_path, train_seqs=np.array(tr_seqs, dtype=object), train_labels=np.array(tr_lbls), val_seqs=np.array(va_seqs, dtype=object), val_labels=np.array(va_lbls))
    logger.info(f"Saved to {args.out_path} | train={len(tr_seqs)} val={len(va_seqs)}")

if __name__ == "__main__":
    main()
