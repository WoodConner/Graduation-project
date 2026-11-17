# 文件内容较长，使用heredoc创建
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from metrics.joint_angles import (
    compute_all_angles, compute_symmetry_score,
    compute_body_tilt, compute_motion_range,
    compute_velocity_smoothness, compute_support_stability
)
from rules.action_standards import get_standard, JOINT_NAMES
from rules.action_configs import get_action_config


class QualityScorer:
    def __init__(self, action_name: str):
        self.action_name = action_name
        self.standard = get_standard(action_name)
        self.config = get_action_config(action_name)
        self.weights = self.config['dimension_weights']
    
    def score_angles(self, sequence):
        if len(self.standard.angle_ranges) == 0:
            return 1.0
        scores = []
        for t in range(len(sequence)):
            angles = compute_all_angles(sequence[t])
            frame_scores = []
            for joint, (min_angle, max_angle) in self.standard.angle_ranges.items():
                if joint not in angles:
                    continue
                actual_angle = angles[joint]
                if min_angle <= actual_angle <= max_angle:
                    frame_scores.append(1.0)
                else:
                    deviation = min(abs(actual_angle - min_angle), abs(actual_angle - max_angle))
                    score = max(0.0, 1.0 - deviation / 30.0)
                    frame_scores.append(score)
            if frame_scores:
                scores.append(np.mean(frame_scores))
        return np.mean(scores) if scores else 1.0
    
    def score_amplitude(self, sequence):
        if len(self.standard.motion_ranges) == 0:
            return 1.0
        scores = []
        for joint_name, min_range in self.standard.motion_ranges.items():
            if joint_name not in JOINT_NAMES:
                continue
            joint_idx = JOINT_NAMES.index(joint_name)
            actual_range = compute_motion_range(sequence, joint_idx)
            scores.append(1.0 if actual_range >= min_range else actual_range / (min_range + 1e-8))
        return np.mean(scores) if scores else 1.0
    
    def score_symmetry(self, sequence):
        symmetry_scores = []
        for t in range(len(sequence)):
            angles = compute_all_angles(sequence[t])
            sym = compute_symmetry_score(angles)
            symmetry_scores.append(np.mean(list(sym.values())))
        avg_symmetry = np.mean(symmetry_scores)
        weight = self.config['symmetry_weight']
        return avg_symmetry * weight + (1.0 - weight) * 1.0
    
    def score_rhythm(self, sequence):
        key_joints = [2, 5, 12, 15]
        smoothness_scores = []
        for joint_idx in key_joints:
            smoothness = compute_velocity_smoothness(sequence, joint_idx)
            smoothness_scores.append(max(0.0, 1.0 - smoothness / 0.01))
        avg_smoothness = np.mean(smoothness_scores)
        weight = self.config['rhythm_weight']
        return avg_smoothness * weight + (1.0 - weight) * 1.0
    
    def score_stability(self, sequence):
        com_range = compute_support_stability(sequence)
        tilts = [compute_body_tilt(sequence[t]) for t in range(len(sequence))]
        avg_shoulder_tilt = np.mean([t[0] for t in tilts])
        avg_hip_tilt = np.mean([t[1] for t in tilts])
        threshold = self.config['stability_threshold']
        com_score = max(0.0, 1.0 - com_range / threshold)
        shoulder_score = max(0.0, 1.0 - avg_shoulder_tilt / 0.2)
        hip_score = max(0.0, 1.0 - avg_hip_tilt / 0.15)
        return com_score * 0.5 + shoulder_score * 0.3 + hip_score * 0.2
    
    def score(self, sequence, verbose=False):
        angle_raw = self.score_angles(sequence)
        amplitude_raw = self.score_amplitude(sequence)
        symmetry_raw = self.score_symmetry(sequence)
        rhythm_raw = self.score_rhythm(sequence)
        stability_raw = self.score_stability(sequence)
        
        angle_score = angle_raw * 100 * self.weights['angle']
        amplitude_score = amplitude_raw * 100 * self.weights['amplitude']
        symmetry_score = symmetry_raw * 100 * self.weights['symmetry']
        rhythm_score = rhythm_raw * 100 * self.weights['rhythm']
        stability_score = stability_raw * 100 * self.weights['stability']
        
        total = angle_score + amplitude_score + symmetry_score + rhythm_score + stability_score
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"动作: {self.action_name}")
            print(f"关节角度: {angle_score:.1f}/{self.weights['angle']*100:.0f}")
            print(f"幅度: {amplitude_score:.1f}/{self.weights['amplitude']*100:.0f}")
            print(f"对称性: {symmetry_score:.1f}/{self.weights['symmetry']*100:.0f} (要求度{self.config['symmetry_weight']:.0%})")
            print(f"节奏: {rhythm_score:.1f}/{self.weights['rhythm']*100:.0f}")
            print(f"稳定性: {stability_score:.1f}/{self.weights['stability']*100:.0f}")
            print(f"总分: {total:.1f}/100")
            print(f"{'='*60}\n")
        
        return {
            'total': total, 'angle': angle_score, 'amplitude': amplitude_score,
            'symmetry': symmetry_score, 'rhythm': rhythm_score, 'stability': stability_score,
            'action': self.action_name, 'weights': self.weights
        }


def batch_score(sequences, labels, action_names, verbose=False):
    results = []
    for seq, label in zip(sequences, labels):
        scorer = QualityScorer(action_names[label])
        results.append(scorer.score(seq, verbose=verbose))
    return results
