from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "crossover_results.csv"
OUT = ROOT / "results" / "figures" / "crossover.png"

d = pd.read_csv(CSV)
d = d[d["correct"].astype(str).str.lower() == "true"].copy()
cpu_names = ["openmp_static_16", "openmp_avx2_12"]
cpu = d[d.implementation.isin(cpu_names)].groupby("N", as_index=False).time_ms.min()
cpu["implementation"] = "CPU best"
curves = pd.concat([cpu, d[d.implementation.isin(["cuda_2d_element", "cuda_row_kernel"])][["N", "time_ms", "implementation"]]])
for name, g in curves.groupby("implementation"):
    g = g.sort_values("N")
    plt.plot(g.N, g.time_ms, marker="o", linewidth=2, label=name)
plt.xlabel("N"); plt.ylabel("time_ms (median)"); plt.grid(True, alpha=.3); plt.legend()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(OUT, dpi=180); plt.close()

wide = d.pivot(index="N", columns="implementation", values="time_ms")
wide["cpu_best"] = wide[cpu_names].min(axis=1)
wide["cuda_minus_cpu"] = wide["cuda_2d_element"] - wide["cpu_best"]
cross = None
ns = list(wide.index.sort_values())
for lo, hi in zip(ns, ns[1:]):
    if wide.loc[lo, "cuda_minus_cpu"] >= 0 and wide.loc[hi, "cuda_minus_cpu"] < 0:
        cross = (lo, hi)
        break
print(f"crossover_interval={cross[0]}-{cross[1]}" if cross else "crossover_interval=not_observed")
print(wide[["cpu_best", "cuda_2d_element", "cuda_row_kernel", "cuda_minus_cpu"]].to_string())
