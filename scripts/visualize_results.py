import numpy as np
import matplotlib.pyplot as plt

# ---------------- Raw data ----------------
accmag_deg = [4.16, 13.22, 19.57, 6.20, 21.77, 4.40, 17.53, 14.64, 18.64,
              5.19, 24.54, 20.62, 6.24, 15.76, 4.87, 29.60, 6.87, 28.13,
              16.68, 45.05, 16.49, 18.28, 43.02, 24.61, 24.28, 12.37,
              17.74, 21.61, 17.68, 20.16, 35.26, 25.56, 26.48, 29.76,
              25.60, 20.90, 11.62]

acconly_deg = [1.41, 8.27, 16.31, 3.64, 38.80, 2.28, 13.96, 19.36, 44.92,
               3.20, 20.96, 17.56, 3.32, 12.72, 3.44, 27.14, 7.01, 24.52,
               25.97, 60.91, 11.59, 29.34, 56.32, 20.57, 20.95, 9.60,
               21.25, 17.72, 13.58, 18.19, 33.81, 25.27, 24.74, 29.32,
               26.09, 22.51, 13.42]

datasets = [f"{i:02d}" for i in range(1, len(accmag_deg)+1)]

# ---------------- Split ----------------
# Corrected indices: undisturbed are the first 23 datasets (0-22), disturbed are the rest
undisturbed_idx = list(range(0, 23))
disturbed_idx   = list(range(23, len(accmag_deg)))

# ---------------- Aggregate stats ----------------
def mean_std(arr, idxs):
    vals = [arr[i] for i in idxs]
    return np.mean(vals), np.std(vals)

u_accmag_mean, u_accmag_std = mean_std(accmag_deg, undisturbed_idx)
u_acconly_mean, u_acconly_std = mean_std(acconly_deg, undisturbed_idx)
d_accmag_mean, d_accmag_std = mean_std(accmag_deg, disturbed_idx)
d_acconly_mean, d_acconly_std = mean_std(acconly_deg, disturbed_idx)

print("Undisturbed → Acc+Mag:", u_accmag_mean, "±", u_accmag_std,
      "| Acc-Only:", u_acconly_mean, "±", u_acconly_std)
print("Disturbed   → Acc+Mag:", d_accmag_mean, "±", d_accmag_std,
      "| Acc-Only:", d_acconly_mean, "±", d_acconly_std)

# ---------------- Plots ----------------
x = np.arange(len(datasets))
width = 0.35

# (1) Bar chart per dataset
plt.figure(figsize=(14,6))
plt.bar(x - width/2, accmag_deg, width, label="Acc+Mag")
plt.bar(x + width/2, acconly_deg, width, label="Acc-Only")
plt.axhline(15, color="green", linestyle="--", label="Acc+Mag Threshold (15°)")
plt.axhline(37, color="red", linestyle="--", label="Acc-Only Threshold (37°)")
plt.xticks(x, datasets, rotation=45)
plt.ylabel("Mean Angular Error (°)")
plt.title("Performance Across Test Datasets")
plt.legend()
plt.tight_layout()
plt.show()

# (2) Boxplots for subsets
plt.figure(figsize=(10,6))
plt.boxplot([
    [accmag_deg[i] for i in undisturbed_idx],
    [acconly_deg[i] for i in undisturbed_idx],
    [accmag_deg[i] for i in disturbed_idx],
    [acconly_deg[i] for i in disturbed_idx],
], labels=["Acc+Mag (U)", "Acc-Only (U)", "Acc+Mag (D)", "Acc-Only (D)"])
plt.ylabel("Error (°)")
plt.title("Error Distributions: Undisturbed (U) vs Disturbed (D)")
plt.grid(True)
plt.show()

# (3) Aggregate bar (mean ± std)
means = [u_accmag_mean, u_acconly_mean, d_accmag_mean, d_acconly_mean]
stds  = [u_accmag_std, u_acconly_std, d_accmag_std, d_acconly_std]
labels = ["Acc+Mag (U)", "Acc-Only (U)", "Acc+Mag (D)", "Acc-Only (D)"]

plt.figure(figsize=(8,6))
plt.bar(labels, means, yerr=stds, capsize=5)
plt.ylabel("Error (°)")
plt.title("Aggregate Performance (mean ± std)")
plt.grid(axis="y")
plt.show()