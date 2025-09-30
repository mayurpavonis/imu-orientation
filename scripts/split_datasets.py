# ============================
# Split and Save Datasets
# ============================

import os, glob, h5py, torch
import numpy as np

# ---------------- Config ----------------
WIN_LEN     = 200    # no. of samples per window (train only)
STRIDE      = 100    # overlap for train
TRAIN_RATIO = 0.8    # 80/20 split
BASE_SAVE_DIR = "models"
SPLIT_DIR = "data/splits"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ---------------- Split Function ----------------
def split_and_save_dataset(h5_path, win_len=WIN_LEN, stride=STRIDE, train_ratio=TRAIN_RATIO):
    with h5py.File(h5_path, 'r') as f:
        acc  = np.nan_to_num(f['imu_acc'][:])
        mag  = np.nan_to_num(f['imu_mag'][:])
        quat = np.nan_to_num(f['opt_quat'][:])
    N = len(acc)
    split_idx = int(train_ratio * N)

    train_windows = []

    # -------- Train windows (with overlap) --------
    for i in range(0, split_idx - win_len, stride):
        if i + win_len <= split_idx:
            train_windows.append({
                "acc": acc[i:i+win_len].astype(np.float64),
                "mag": mag[i:i+win_len].astype(np.float64),
                "quat": quat[i:i+win_len].astype(np.float64),
            })

    # -------- Test sequence (no windows) --------
    test_seq = {
        "acc": acc[split_idx:].astype(np.float64),
        "mag": mag[split_idx:].astype(np.float64),
        "quat": quat[split_idx:].astype(np.float64),
    }

    # Save as .pt files
    base = os.path.splitext(os.path.basename(h5_path))[0]
    train_file = os.path.join(SAVE_DIR, f"{base}_train.pt")
    test_file  = os.path.join(SAVE_DIR, f"{base}_test.pt")

    torch.save(train_windows, train_file)
    torch.save(test_seq, test_file)

    print(f"{base}: saved {len(train_windows)} train windows, 1 test sequence ({len(test_seq['acc'])} samples)")
    return len(train_windows), len(test_seq['acc'])

# ---------------- Run Splitting ----------------
h5_paths = sorted(glob.glob(f"{DATA_DIR}/*.hdf5"))
if not h5_paths:
    print("No .hdf5 datasets found")
else:
    total_train, total_test_samples = 0, 0
    for path in h5_paths:
        n_train, n_test_samples = split_and_save_dataset(path)
        total_train += n_train
        total_test_samples += n_test_samples
    print(f"\nDone. Total -> Train windows: {total_train}, Test samples: {total_test_samples}")