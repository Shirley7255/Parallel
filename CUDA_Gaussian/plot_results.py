import csv
import os

import matplotlib.pyplot as plt


# Second run results are used as the main report data.
DEFAULT_DATA = {
    "N": [256, 512, 768, 1024],
    "CPU_Time_ms": [2.407, 18.913, 64.425, 155.962],
    "GPU_Elim_ms": [4.319, 11.724, 27.042, 55.651],
    "GPU_Total_ms": [5.321, 13.845, 30.677, 60.792],
    "Speedup": [0.452, 1.366, 2.100, 2.566],
    "Max_Error": [2.776e-17, 1.388e-17, 6.939e-18, 1.041e-17],
    "Residual": [2.487e-14, 3.908e-14, 6.040e-14, 9.237e-14],
}

FIRST_GPU_TOTAL_MS = [120.329, 13.701, 31.601, 61.922]


def load_results_if_requested():
    """Use results.csv only when explicitly requested by USE_RESULTS_CSV=1."""
    if os.environ.get("USE_RESULTS_CSV") != "1":
        return DEFAULT_DATA

    csv_path = "results.csv"
    if not os.path.exists(csv_path):
        return DEFAULT_DATA

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return DEFAULT_DATA

        data = {
            "N": [],
            "CPU_Time_ms": [],
            "GPU_Elim_ms": [],
            "GPU_Total_ms": [],
            "Speedup": [],
            "Max_Error": [],
            "Residual": [],
        }

        for row in rows:
            data["N"].append(int(row["N"]))
            data["CPU_Time_ms"].append(float(row["CPU Time(ms)"]))
            data["GPU_Elim_ms"].append(float(row["GPU Elim(ms)"]))
            data["GPU_Total_ms"].append(float(row["GPU Total(ms)"]))
            data["Speedup"].append(float(row["Speedup"]))
            data["Max_Error"].append(float(row["Max Solution Error"]))
            data["Residual"].append(float(row["Residual Max Error"]))

        return data
    except (KeyError, ValueError, OSError):
        return DEFAULT_DATA


def annotate_points(x_values, y_values, fmt="{:.3f}", y_offset=6):
    for x, y in zip(x_values, y_values):
        plt.annotate(
            fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            fontsize=8,
        )


def save_current_figure(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_time_comparison(data):
    # Figure 1 compares serial CPU time with full GPU time.
    plt.figure(figsize=(7, 4.5))
    plt.plot(data["N"], data["CPU_Time_ms"], marker="o", linewidth=2, label="CPU Time")
    plt.plot(data["N"], data["GPU_Total_ms"], marker="s", linewidth=2, label="GPU Total Time")
    annotate_points(data["N"], data["CPU_Time_ms"])
    annotate_points(data["N"], data["GPU_Total_ms"], y_offset=-14)
    plt.title("CPU vs GPU Total Time")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Time (ms)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    save_current_figure("figure1_time_comparison.png")


def plot_speedup(data):
    # Figure 2 shows when GPU total time becomes faster than CPU time.
    plt.figure(figsize=(7, 4.5))
    plt.plot(data["N"], data["Speedup"], marker="o", linewidth=2, label="Speedup")
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.2, label="Break-even")
    annotate_points(data["N"], data["Speedup"])
    plt.title("GPU Speedup over CPU")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    save_current_figure("figure2_speedup.png")


def plot_accuracy(data):
    # Figure 3 verifies numerical correctness with solution error and residual.
    plt.figure(figsize=(7, 4.5))
    plt.plot(data["N"], data["Max_Error"], marker="o", linewidth=2, label="Max Error")
    plt.plot(data["N"], data["Residual"], marker="s", linewidth=2, label="Residual")
    plt.yscale("log")
    plt.title("Numerical Accuracy")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Error")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    save_current_figure("figure3_accuracy.png")


def plot_first_vs_second(data):
    # Figure 4 compares first and second GPU total time to show initialization effects.
    plt.figure(figsize=(7, 4.5))
    x_positions = list(range(len(data["N"])))
    width = 0.36

    first_positions = [x - width / 2 for x in x_positions]
    second_positions = [x + width / 2 for x in x_positions]

    first_bars = plt.bar(first_positions, FIRST_GPU_TOTAL_MS, width, label="First Run")
    second_bars = plt.bar(second_positions, data["GPU_Total_ms"], width, label="Second Run")

    for bars in (first_bars, second_bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.title("First Run vs Second Run GPU Total Time")
    plt.xlabel("Matrix Size N")
    plt.ylabel("GPU Total Time (ms)")
    plt.xticks(x_positions, data["N"])
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    save_current_figure("figure4_first_vs_second_run.png")


def main():
    data = load_results_if_requested()
    generated_files = [
        "figure1_time_comparison.png",
        "figure2_speedup.png",
        "figure3_accuracy.png",
        "figure4_first_vs_second_run.png",
    ]

    plot_time_comparison(data)
    plot_speedup(data)
    plot_accuracy(data)
    plot_first_vs_second(data)

    print("Generated figures:")
    for filename in generated_files:
        print(f"- {filename}")
    print(f"Current working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
