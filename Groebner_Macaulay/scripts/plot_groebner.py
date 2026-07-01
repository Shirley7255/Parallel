from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "groebner_results.csv"
FIG_DIR = ROOT / "results" / "figures"


def load_data() -> pd.DataFrame:
    if not CSV.exists():
        raise FileNotFoundError(f"missing CSV: {CSV}")
    df = pd.read_csv(CSV)
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    bad = df[~df["correct"]]
    if not bad.empty:
        raise RuntimeError(f"incorrect rows found:\n{bad}")
    df["case"] = df.apply(lambda r: f"n={int(r.n_vars)},D={int(r.D)}\n{int(r.rows)}x{int(r.cols)}", axis=1)
    return df


def best_openmp(df: pd.DataFrame) -> pd.DataFrame:
    omp = df[df["impl"] == "openmp"].copy()
    idx = omp.groupby(["n_vars", "D"])["time_ms"].idxmin()
    return omp.loc[idx].sort_values(["n_vars", "D"])


def serial_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["impl"] == "serial"].sort_values(["n_vars", "D"])


def plot_time(df: pd.DataFrame) -> None:
    ser = serial_rows(df)
    omp = best_openmp(df)
    labels = ser["case"].tolist()
    x = range(len(labels))

    plt.figure(figsize=(10, 5))
    plt.plot(x, ser["time_ms"], marker="o", linewidth=2, label="serial")
    plt.plot(x, omp["time_ms"], marker="s", linewidth=2, label="OpenMP best")
    for i, row in enumerate(omp.itertuples()):
        plt.annotate(f"{int(row.threads)}T", (i, row.time_ms), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xticks(list(x), labels)
    plt.ylabel("time_ms")
    plt.title("Macaulay finite-field Gaussian elimination time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "groebner_time.png", dpi=180)
    plt.close()


def plot_fillin(df: pd.DataFrame) -> None:
    ser = serial_rows(df)
    labels = ser["case"].tolist()
    x = range(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], ser["density_before"], width=width, label="density_before")
    plt.bar([i + width / 2 for i in x], ser["density_after"], width=width, label="density_after")
    plt.xticks(list(x), labels)
    plt.ylabel("density")
    plt.title("Macaulay matrix fill-in during modular elimination")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "groebner_fillin.png", dpi=180)
    plt.close()


def plot_speedup(df: pd.DataFrame) -> None:
    ser = serial_rows(df)[["n_vars", "D", "time_ms", "case"]].rename(columns={"time_ms": "serial_ms"})
    omp = df[df["impl"] == "openmp"].copy()
    merged = omp.merge(ser, on=["n_vars", "D"], how="left")
    merged["speedup"] = merged["serial_ms"] / merged["time_ms"]

    plt.figure(figsize=(10, 5))
    for t, g in merged.groupby("threads"):
        g = g.sort_values(["n_vars", "D"])
        plt.plot(range(len(g)), g["speedup"], marker="o", linewidth=2, label=f"{int(t)} threads")
    labels = ser.sort_values(["n_vars", "D"])["case"].tolist()
    plt.xticks(list(range(len(labels))), labels)
    plt.ylabel("speedup vs serial")
    plt.title("OpenMP speedup for Macaulay modular elimination")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "groebner_speedup.png", dpi=180)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    plot_time(df)
    plot_fillin(df)
    plot_speedup(df)
    print(f"wrote {FIG_DIR / 'groebner_time.png'}")
    print(f"wrote {FIG_DIR / 'groebner_fillin.png'}")
    print(f"wrote {FIG_DIR / 'groebner_speedup.png'}")


if __name__ == "__main__":
    main()
