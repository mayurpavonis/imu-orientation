import torch
import torch.nn as nn
from .losses import quat_normalize

HID = 32
BASE_SAVE_DIR = "models"

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

# Load trained fp64 models
device = torch.device("cpu")
net_acc_mag = AccMagNet(HID)
net_acc_mag.load_state_dict(torch.load(f"{BASE_SAVE_DIR}/accmag_h{HID}_fp64.pth", map_location="cpu"))
net_acc_mag.eval()

net_acc_only = AccOnlyNet(HID)
net_acc_only.load_state_dict(torch.load(f"{BASE_SAVE_DIR}/acconly_h{HID}_fp64.pth", map_location="cpu"))
net_acc_only.eval()

# Convert from fp64 → fp32 (quantization expects float32 first)
net_acc_mag = net_acc_mag.float()
net_acc_only = net_acc_only.float()

# Apply dynamic quantization
q_net_acc_mag = torch.quantization.quantize_dynamic(
    net_acc_mag, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
)
q_net_acc_only = torch.quantization.quantize_dynamic(
    net_acc_only, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized models
torch.save(q_net_acc_mag.state_dict(), f"{BASE_SAVE_DIR}/accmag_h{HID}_int8.pth")
torch.save(q_net_acc_only.state_dict(), f"{BASE_SAVE_DIR}/acconly_h{HID}_int8.pth")

print("Models quantized and saved (fp64 → int8)")