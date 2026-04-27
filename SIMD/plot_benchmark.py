import matplotlib.pyplot as plt

# Measured data from the latest run
N = [256, 512, 1024]

serial_time = [3.5813, 41.399, 212.884]
avx_unaligned_time = [0.7771, 6.6227, 69.2815]
avx_aligned_time = [0.9293, 6.2389, 73.9819]
autovec_time = [0.9441, 6.5690, 84.0547]

speedup_unaligned = [4.60854, 6.25108, 3.07274]
speedup_aligned = [3.85376, 6.63562, 2.87751]
speedup_autovec = [3.79335, 6.30218, 2.53268]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# Left: runtime curve
ax1 = axes[0]
ax1.plot(N, serial_time, marker="o", linewidth=2.0, label="Serial")
ax1.plot(N, avx_unaligned_time, marker="s", linewidth=2.0, label="AVX Unaligned")
ax1.plot(N, avx_aligned_time, marker="^", linewidth=2.0, label="AVX Aligned")
ax1.plot(N, autovec_time, marker="d", linewidth=2.0, label="Auto-Vectorization")
ax1.set_title("Runtime vs Matrix Size")
ax1.set_xlabel("Matrix Size N")
ax1.set_ylabel("Time (ms)")
ax1.set_xticks(N)
ax1.legend()

# Right: speedup curve (relative to serial)
ax2 = axes[1]
ax2.plot(N, speedup_unaligned, marker="s", linewidth=2.0, label="AVX Unaligned")
ax2.plot(N, speedup_aligned, marker="^", linewidth=2.0, label="AVX Aligned")
ax2.plot(N, speedup_autovec, marker="d", linewidth=2.0, label="Auto-Vectorization")
ax2.set_title("Speedup vs Matrix Size")
ax2.set_xlabel("Matrix Size N")
ax2.set_ylabel("Speedup (x)")
ax2.set_xticks(N)
ax2.legend()

fig.suptitle("Gaussian Elimination Performance Comparison", fontsize=13)
fig.tight_layout()
fig.savefig("benchmark_runtime_speedup.png", bbox_inches="tight")
print("Saved: benchmark_runtime_speedup.png")
