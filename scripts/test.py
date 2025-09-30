# ============================
# Testing Script
# ============================

import os, glob, torch, numpy as np, matplotlib.pyplot as plt
from src.networks import AccMagNet, AccOnlyNet
from src.losses import geodesic_quat_loss_sign_invariant

# ---------------- Config ----------------
HID = 32
BASE_SAVE_DIR = "models"
SPLIT_DIR = "data/splits"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ---------------- Quaternion utilities ----------------
def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

def geodesic_quat_loss_sign_invariant(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    qp = quat_normalize(q_pred)
    qg = quat_normalize(q_gt)
    dot = torch.sum(qp * qg, dim=-1)
    sign = torch.sign(dot).unsqueeze(-1)
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
    qp_aligned = qp * sign
    dot2 = (qp_aligned * qg).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    ang = 2.0 * torch.acos(dot2)
    return torch.where(torch.isfinite(ang), ang, torch.zeros_like(ang)).mean()

# ---------------- Models ----------------
class AccMagNet(nn.Module):
    def __init__(self, hid=HID):
        super().__init__()
        self.gru = nn.GRU(6, hid, batch_first=True)
        self.fc_q = nn.Linear(hid, 4)
        self.double()
    def forward(self, acc, mag):
        x = torch.cat([acc, mag], dim=-1)
        out, _ = self.gru(x)
        return quat_normalize(self.fc_q(out))

class AccOnlyNet(nn.Module):
    def __init__(self, hid=HID):
        super().__init__()
        self.gru = nn.GRU(3, hid, batch_first=True)
        self.fc_q = nn.Linear(hid, 4)
        self.double()
    def forward(self, acc):
        out, _ = self.gru(acc)
        return quat_normalize(self.fc_q(out))

# ---------------- Evaluation ----------------
def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Testing with float64")

    # Load trained networks
    net_acc_mag = AccMagNet().to(device).eval()
    net_acc_mag.load_state_dict(torch.load(
        os.path.join(BASE_SAVE_DIR, f"accmag_h{HID}_fp64.pth"), map_location=device))

    net_acc_only = AccOnlyNet().to(device).eval()
    net_acc_only.load_state_dict(torch.load(
        os.path.join(BASE_SAVE_DIR, f"acconly_h{HID}_fp64.pth"), map_location=device))

    # Find test datasets
    test_files = sorted(glob.glob(os.path.join(SPLITS_DIR, "*_test.pt")))
    if not test_files:
        print("❌ No test datasets found in:", SPLITS_DIR)
        return

    dataset_names = []
    losses_accmag_deg, losses_acconly_deg = [], []

    for path in test_files:
        test_seq = torch.load(path, weights_only=False) # Add weights_only=False

        acc  = torch.from_numpy(test_seq['acc']).unsqueeze(0).double().to(device)
        mag  = torch.from_numpy(test_seq['mag']).unsqueeze(0).double().to(device)
        quat = torch.from_numpy(test_seq['quat']).unsqueeze(0).double().to(device)

        # Acc+Mag
        q_pred_accmag = net_acc_mag(acc, mag)
        loss_accmag = geodesic_quat_loss_sign_invariant(
            q_pred_accmag.reshape(-1, 4), quat.reshape(-1, 4))

        # Acc-Only
        q_pred_acconly = net_acc_only(acc)
        loss_acconly = geodesic_quat_loss_sign_invariant(
            q_pred_acconly.reshape(-1, 4), quat.reshape(-1, 4))

        dataset_name = os.path.basename(path).replace("_test.pt", "")
        dataset_names.append(dataset_name)
        losses_accmag_deg.append(np.degrees(loss_accmag.item()))
        losses_acconly_deg.append(np.degrees(loss_acconly.item()))

        print(f"{dataset_name} -> "
              f"Acc+Mag: {losses_accmag_deg[-1]:.2f}° | "
              f"Acc-Only: {losses_acconly_deg[-1]:.2f}°")

    # --- Plot results ---
    x = np.arange(len(dataset_names))
    width = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, losses_accmag_deg, width, label="Acc+Mag (°)")
    plt.bar(x + width/2, losses_acconly_deg, width, label="Acc-Only (°)")
    plt.xticks(x, dataset_names, rotation=45, ha="right")
    plt.ylabel("Mean Angular Error (°)")
    plt.title("Model Performance Across Test Datasets")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- Run ----------------
evaluate_models()