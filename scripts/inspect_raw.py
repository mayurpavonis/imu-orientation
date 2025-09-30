import h5py
import os

import glob

# Point to your dataset directory
data_dir = "data/raw"

# Grab all .hdf5 files
h5_paths = sorted(glob.glob(f"{data_dir}/*.hdf5"))

print("Found datasets:")
for p in h5_paths:
    print(p)

def print_h5_structure(filename):
    print(f"\n=== File: {os.path.basename(filename)} ===")
    with h5py.File(filename, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[DATASET] {name} -> shape {obj.shape}, dtype {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"[GROUP]   {name}")
        f.visititems(visitor)

for path in h5_paths:
    print_h5_structure(path)