import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(csv_path: Path):
    n = []
    naive_ms = []
    optimized_ms = []
    speedup = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n.append(int(row["n"]))
            naive_ms.append(float(row["naive_ms"]))
            optimized_ms.append(float(row["optimized_ms"]))
            speedup.append(float(row["speedup"]))

    return n, naive_ms, optimized_ms, speedup


def save_single_curve(x, y, title, y_label, output_path, color, marker, legend_label):
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(x, y, color=color, marker=marker, linewidth=2.0, label=legend_label)
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "benchmark_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    n, naive_ms, optimized_ms, speedup = load_data(csv_path)

    save_single_curve(
        n,
        naive_ms,
        "Naive Algorithm Time vs n",
        "Time (ms)",
        base_dir / "curve_naive_ms.png",
        color="#d1495b",
        marker="o",
        legend_label="naive(ms)",
    )

    save_single_curve(
        n,
        optimized_ms,
        "Optimized Algorithm Time vs n",
        "Time (ms)",
        base_dir / "curve_optimized_ms.png",
        color="#00798c",
        marker="s",
        legend_label="optimized(ms)",
    )

    save_single_curve(
        n,
        speedup,
        "Speedup vs n",
        "Speedup (naive/optimized)",
        base_dir / "curve_speedup.png",
        color="#edae49",
        marker="^",
        legend_label="speedup",
    )

    print("Generated:")
    print(base_dir / "curve_naive_ms.png")
    print(base_dir / "curve_optimized_ms.png")
    print(base_dir / "curve_speedup.png")


if __name__ == "__main__":
    main()
