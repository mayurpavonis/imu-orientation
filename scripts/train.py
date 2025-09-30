import os, glob, torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.networks import AccMagNet, AccOnlyNet
from src.losses import geodesic_quat_loss_sign_invariant
from src.datasets import WindowDataset

HID = 32
BATCH_SIZE = 16
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

# ---------------- Dataset ----------------
class WindowDataset(Dataset):
    def __init__(self, pt_files):
        self.windows = []
        for f in pt_files:
            # Explicitly set weights_only=False to load the full pickled object
            self.windows.extend(torch.load(f, weights_only=False))  # each is a list of dicts
        print(f"Loaded {len(self.windows)} windows from {len(pt_files)} files")

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        w = self.windows[idx]
        return {k: torch.tensor(v, dtype=torch.float64) for k, v in w.items()}

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

# ---------------- Training ----------------
def train_network(net, dataloader, device, epochs=10, mode="acc_mag", label="AccMag", lr=5e-4):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    all_losses_rad, all_losses_deg = [], []

    for ep in range(epochs):
        losses = []
        for batch in dataloader:
            acc = batch['acc'].to(device)
            mag = batch['mag'].to(device)
            quat = batch['quat'].to(device)
            q_pred = net(acc, mag) if mode=="acc_mag" else net(acc)

            loss = geodesic_quat_loss_sign_invariant(q_pred.reshape(-1, 4), quat.reshape(-1, 4))
            if torch.isnan(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            losses.append(loss.item())

        mean_loss_rad = np.mean(losses)
        mean_loss_deg = np.degrees(mean_loss_rad)
        all_losses_rad.append(mean_loss_rad)
        all_losses_deg.append(mean_loss_deg)
        print(f"Epoch {ep:02d}: {mean_loss_rad:.6f} rad ({mean_loss_deg:.2f}°)")

    # Plot training curve
    plt.figure(figsize=(8,5))
    plt.plot(all_losses_rad, label="Loss (rad)")
    plt.plot(all_losses_deg, label="Loss (deg)")
    plt.title(f"Training Loss Curve - {label}")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.show()

    return net

# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Using float64 (double precision)")

    # Load pre-split train windows
    train_files = sorted(glob.glob(os.path.join(SPLIT_DIR, "*_train.pt")))
    if not train_files:
        print("❌ No train split files found")
        return

    train_dataset = WindowDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # epochs = max(5, len(train_dataset)//64)
    epochs = max(5, 200) # should be enough for Acc-Only convergence

    # Train Acc+Mag network --- temporarily commented out
    """ net_acc_mag = AccMagNet()
    print("\n=== Training Acc+Mag network ===")
    net_acc_mag = train_network(net_acc_mag, train_loader, device, epochs, mode="acc_mag", label="Acc+Mag")
    torch.save(net_acc_mag.state_dict(), os.path.join(BASE_SAVE_DIR, f"accmag_h{HID}_fp64.pth"))
    print("✅ Saved Acc+Mag network") """

    # Train Acc-Only network
    net_acc_only = AccOnlyNet()
    print("\n=== Training Acc-Only network ===")
    net_acc_only = train_network(net_acc_only, train_loader, device, epochs, mode="acc_only", label="Acc-Only")
    torch.save(net_acc_only.state_dict(), os.path.join(BASE_SAVE_DIR, f"acconly_h{HID}_fp64.pth"))
    print("✅ Saved Acc-Only network")

if __name__=="__main__":
    main()