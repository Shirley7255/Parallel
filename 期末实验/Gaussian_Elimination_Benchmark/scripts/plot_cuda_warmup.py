from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "cuda_warmup_results.csv"
FIG_DIR = ROOT / "results" / "figures"
OUT = FIG_DIR / "cuda_warmup_compare.png"


def main() -> None:
    if not CSV.exists():
        raise FileNotFoundError(f"missing input CSV: {CSV}")

    df = pd.read_csv(CSV)
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    bad = df[~df["correct"]]
    if not bad.empty:
        raise RuntimeError(f"incorrect CUDA warm-up rows found:\n{bad}")

    med = (
        df.groupby(["N", "mode"], as_index=False)[["gpu_elim_ms", "gpu_total_ms"]]
        .median()
        .sort_values(["N", "mode"])
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    colors = {"cold": "#d55e00", "warm": "#0072b2"}
    for mode in ["cold", "warm"]:
        raw = df[df["mode"] == mode]
        m = med[med["mode"] == mode]

        axes[0].scatter(raw["N"], raw["gpu_total_ms"], alpha=0.22, s=28, color=colors[mode])
        axes[0].plot(m["N"], m["gpu_total_ms"], marker="o", linewidth=2.0, label=f"{mode} median", color=colors[mode])

        axes[1].scatter(raw["N"], raw["gpu_elim_ms"], alpha=0.22, s=28, color=colors[mode])
        axes[1].plot(m["N"], m["gpu_elim_ms"], marker="o", linewidth=2.0, label=f"{mode} median", color=colors[mode])

    axes[0].set_title("GPU total time: cold vs warm")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("gpu_total_ms")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("GPU elimination time: cold vs warm")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("gpu_elim_ms")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("CUDA warm-up comparison (cuda_2d_element, repeat=5)")
    fig.savefig(OUT, dpi=180)

    pivot = med.pivot(index="N", columns="mode", values=["gpu_elim_ms", "gpu_total_ms"])
    rows = []
    for n in pivot.index:
        cold_total = pivot.loc[n, ("gpu_total_ms", "cold")]
        warm_total = pivot.loc[n, ("gpu_total_ms", "warm")]
        cold_elim = pivot.loc[n, ("gpu_elim_ms", "cold")]
        warm_elim = pivot.loc[n, ("gpu_elim_ms", "warm")]
        rows.append(
            {
                "N": int(n),
                "cold_total_ms": cold_total,
                "warm_total_ms": warm_total,
                "total_delta_ms": cold_total - warm_total,
                "cold_elim_ms": cold_elim,
                "warm_elim_ms": warm_elim,
                "elim_delta_ms": cold_elim - warm_elim,
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"figure={OUT}")


if __name__ == "__main__":
    main()
