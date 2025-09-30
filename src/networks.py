import torch
import torch.nn as nn
from .losses import quat_normalize

HID = 32  # default hidden size

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