import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from metrics.joint_angles import compute_all_angles, compute_symmetry_score, compute_body_tilt, compute_motion_range, compute_velocity_smoothness, compute_support_stability
from rules.action_standards import get_standard, JOINT_NAMES
from rules.action_configs import get_action_config


class QualityScorer:
    def __init__(self, action_name):
        self.action_name = action_name
        self.standard = get_standard(action_name)
        self.config = get_action_config(action_name)
        self.weights = self.config["dimension_weights"]
    
    def score_angles(self, seq):
        if not self.standard.angle_ranges: return 1.0
        scores = []
        for t in range(len(seq)):
            angles = compute_all_angles(seq[t])
            fs = []
            for j, (mi, ma) in self.standard.angle_ranges.items():
                if j not in angles: continue
                a = angles[j]
                if mi <= a <= ma: fs.append(1.0)
                else: fs.append(max(0.0, 1.0 - min(abs(a-mi), abs(a-ma)) / 30.0))
            if fs: scores.append(np.mean(fs))
        return np.mean(scores) if scores else 1.0
    
    def score_amplitude(self, seq):
        if not self.standard.motion_ranges: return 1.0
        scores = []
        for jn, mr in self.standard.motion_ranges.items():
            if jn not in JOINT_NAMES: continue
            ji = JOINT_NAMES.index(jn)
            ar = compute_motion_range(seq, ji)
            scores.append(1.0 if ar >= mr else ar / (mr + 1e-8))
        return np.mean(scores) if scores else 1.0
    
    def score_symmetry(self, seq):
        ss = []
        for t in range(len(seq)):
            angles = compute_all_angles(seq[t])
            sym = compute_symmetry_score(angles)
            ss.append(np.mean(list(sym.values())))
        avg = np.mean(ss)
        w = self.config["symmetry_weight"]
        return avg * w + (1.0 - w)
    
    def score_rhythm(self, seq):
        kj = [2, 5, 12, 15]
        ss = []
        for ji in kj:
            sm = compute_velocity_smoothness(seq, ji)
            ss.append(max(0.0, 1.0 - sm / 0.01))
        avg = np.mean(ss)
        w = self.config["rhythm_weight"]
        return avg * w + (1.0 - w)
    
    def score_stability(self, seq):
        com = compute_support_stability(seq)
        tilts = [compute_body_tilt(seq[t]) for t in range(len(seq))]
        sh = np.mean([t[0] for t in tilts])
        hi = np.mean([t[1] for t in tilts])
        th = self.config["stability_threshold"]
        cs = max(0.0, 1.0 - com / th)
        ss = max(0.0, 1.0 - sh / 0.2)
        hs = max(0.0, 1.0 - hi / 0.15)
        return cs * 0.5 + ss * 0.3 + hs * 0.2
    
    def score(self, seq, verbose=False):
        ar = self.score_angles(seq)
        ampr = self.score_amplitude(seq)
        sr = self.score_symmetry(seq)
        rr = self.score_rhythm(seq)
        str_val = self.score_stability(seq)
        a = ar * 100 * self.weights["angle"]
        amp = ampr * 100 * self.weights["amplitude"]
        s = sr * 100 * self.weights["symmetry"]
        r = rr * 100 * self.weights["rhythm"]
        st = str_val * 100 * self.weights["stability"]
        tot = a + amp + s + r + st
        if verbose:
            print(f"动作: {self.action_name}")
            print(f"角度{a:.1f} 幅度{amp:.1f} 对称{s:.1f}(要求{self.config['symmetry_weight']:.0%}) 节奏{r:.1f} 稳定{st:.1f} = {tot:.1f}/100")
        return {"total": tot, "angle": a, "amplitude": amp, "symmetry": s, "rhythm": r, "stability": st, "action": self.action_name, "weights": self.weights}


def batch_score(seqs, labels, names, verbose=False):
    return [QualityScorer(names[l]).score(s, verbose) for s, l in zip(seqs, labels)]

