import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_ilp_data(csv_path: Path):
    n = []
    naive_ms = []
    unrolled_ms = []
    speedup = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n.append(int(row["n"]))
            naive_ms.append(float(row["naive_ms"]))
            unrolled_ms.append(float(row["unrolled_ms"]))
            speedup.append(float(row["speedup"]))

    return n, naive_ms, unrolled_ms, speedup


def plot_curve(x, y, title, y_label, output_path, color, marker, legend_label):
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
    csv_path = base_dir / "ilp_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    n, naive_ms, unrolled_ms, speedup = load_ilp_data(csv_path)

    plot_curve(
        n,
        naive_ms,
        "Naive Sum Time vs n",
        "Time (ms)",
        base_dir / "ilp_curve_naive_ms.png",
        color="#c3423f",
        marker="o",
        legend_label="naive(ms)",
    )

    plot_curve(
        n,
        unrolled_ms,
        "Unrolled Sum Time vs n",
        "Time (ms)",
        base_dir / "ilp_curve_unrolled_ms.png",
        color="#2a6f97",
        marker="s",
        legend_label="unrolled(ms)",
    )

    plot_curve(
        n,
        speedup,
        "ILP Speedup vs n",
        "Speedup (naive/unrolled)",
        base_dir / "ilp_curve_speedup.png",
        color="#f4a259",
        marker="^",
        legend_label="speedup",
    )

    print("Generated:")
    print(base_dir / "ilp_curve_naive_ms.png")
    print(base_dir / "ilp_curve_unrolled_ms.png")
    print(base_dir / "ilp_curve_speedup.png")


if __name__ == "__main__":
    main()
