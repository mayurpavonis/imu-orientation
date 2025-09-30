import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, pt_files):
        self.windows = []
        for f in pt_files:
            self.windows.extend(torch.load(f, weights_only=False))
        print(f"Loaded {len(self.windows)} windows from {len(pt_files)} files")

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        w = self.windows[idx]
        return {k: torch.tensor(v, dtype=torch.float64) for k, v in w.items()}