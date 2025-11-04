import numpy as np
import torch
from PIL import Image
import pyiqa

# (opsiyonel) TV için piq; yoksa manuel TV kullanacağız
try:
    import piq
    HAS_PIQ = True
except Exception:
    HAS_PIQ = False

# --- helpers ---
def load_tensor01(path):
    img = Image.open(path).convert("RGB")
    ten = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float() / 255.0
    return ten

def total_variation_manual(x):
    # x: (N,C,H,W), [0,1]
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return (dx + dy).item()

def normalize(v, lo, hi, invert=False):
    v = float(v)
    s = (v - lo) / (hi - lo + 1e-9)
    s = max(0.0, min(1.0, s))
    return 1.0 - s if invert else s

# --- load ---
img_path = "test/images/IMG_7012.jpg"
x = load_tensor01(img_path)

print("\nAdvanced Image Quality Metrics (PyIQA + PIQ)")
print("="*60)

# 1) BRISQUE (pyiqa) - lower better
brisque = pyiqa.create_metric('brisque', device='cpu')(x).item()
print(f"BRISQUE (lower is better): {brisque:.4f}")

# 2) NIQE (pyiqa) - lower better
niqe = pyiqa.create_metric('niqe', device='cpu')(x).item()
print(f"NIQE (lower is better):    {niqe:.4f}")

# 3) Total Variation (smoothness proxy) - lower is smoother
if HAS_PIQ:
    tv = piq.total_variation(x, reduction='mean').item()
else:
    tv = total_variation_manual(x)
print(f"Total Variation (lower=smoother): {tv:.6f}")

print("-"*60)

# ---- Combine to a 0..1 TrueQualityScore ----
# Normalize ranges (pratik, saha için mantıklı bandlar):
# BRISQUE:   10..60  (düşük iyi)   -> invert normalize
# NIQE:       2..7   (düşük iyi)   -> invert normalize
# TV:      0.005..0.05 (düşük iyi) -> invert normalize
q_brisque = normalize(brisque, 10, 60, invert=True)
q_niqe    = normalize(niqe,    2.0, 7.0, invert=True)
q_tv      = normalize(tv,   0.005, 0.05, invert=True)

# Ağırlıklar: algısal metriklere daha fazla önem
true_quality = 0.5*q_brisque + 0.35*q_niqe + 0.15*q_tv

print(f"Norm BRISQUE quality: {q_brisque:.3f}")
print(f"Norm NIQE quality   : {q_niqe:.3f}")
print(f"Norm TV quality     : {q_tv:.3f}")
print(f"=> TrueQualityScore (0..1, higher=better): {true_quality:.3f}")
print("="*60)
