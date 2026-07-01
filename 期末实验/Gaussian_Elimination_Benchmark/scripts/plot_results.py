from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def save(name):
    plt.tight_layout()
    plt.savefig(ROOT / "results" / name, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "cpu_gpu_results.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["correct"].astype(str).str.lower() == "true"].copy()
    if df.empty:
        raise SystemExit("No correct benchmark rows found")

    best = df.groupby(["implementation", "N"], as_index=False)["time_ms"].min()
    for impl, g in best.groupby("implementation"):
        plt.plot(g.N, g.time_ms, marker="o", label=impl)
    plt.xlabel("N"); plt.ylabel("Median time (ms)"); plt.yscale("log"); plt.grid(True, alpha=.3); plt.legend(fontsize=8)
    save("time_vs_N.png")

    serial = best[best.implementation == "serial"].set_index("N").time_ms
    best["speedup"] = best.apply(lambda r: serial.get(r.N, float("nan")) / r.time_ms, axis=1)
    for impl, g in best.groupby("implementation"):
        plt.plot(g.N, g.speedup, marker="o", label=impl)
    plt.xlabel("N"); plt.ylabel("Speedup vs serial"); plt.grid(True, alpha=.3); plt.legend(fontsize=8)
    save("speedup_vs_N.png")

    scaling = df[df.implementation.isin(["openmp_static", "openmp_dynamic", "openmp_guided", "openmp_avx2_dynamic"])]
    for (impl, n), g in scaling.groupby(["implementation", "N"]):
        g = g.sort_values("threads")
        plt.plot(g.threads, g.time_ms, marker="o", label=f"{impl}, N={n}")
    plt.xlabel("Threads"); plt.ylabel("Median time (ms)"); plt.grid(True, alpha=.3); plt.legend(fontsize=7, ncol=2)
    save("thread_scaling.png")

    cuda = df[df.implementation.str.startswith("cuda")].sort_values(["implementation", "N"])
    if not cuda.empty:
        x = range(len(cuda)); width = .38
        plt.bar([v-width/2 for v in x], cuda.gpu_elim_ms, width, label="GPU elimination")
        plt.bar([v+width/2 for v in x], cuda.gpu_total_ms, width, label="GPU total")
        plt.xticks(list(x), [f"{r.implementation}\nN={r.N}" for r in cuda.itertuples()], rotation=20, ha="right")
        plt.ylabel("Median time (ms)"); plt.legend(); plt.grid(True, axis="y", alpha=.3)
        save("cuda_breakdown.png")

    families = {
        "Serial": ["serial"], "AVX2": ["avx2_unaligned"],
        "OpenMP": ["openmp_static", "openmp_dynamic", "openmp_guided"],
        "OpenMP+AVX2": ["openmp_avx2_dynamic"],
        "CUDA": ["cuda_2d_element", "cuda_row_kernel"],
    }
    for label, names in families.items():
        g = df[df.implementation.isin(names)].groupby("N", as_index=False).time_ms.min()
        if not g.empty: plt.plot(g.N, g.time_ms, marker="o", linewidth=2, label=label)
    plt.xlabel("N"); plt.ylabel("Best median time (ms)"); plt.yscale("log"); plt.grid(True, alpha=.3); plt.legend()
    save("architecture_comparison.png")

if __name__ == "__main__":
    main()

