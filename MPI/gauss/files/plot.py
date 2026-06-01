#!/usr/bin/env python3
"""
plot.py — Extract [RESULT] lines from *.o files and generate 4 HPC charts.
Output: fig1_speedup.png, fig2_load_balance.png, fig3_topology.png, fig4_hybrid.png
DPI: 300
"""

import re
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# Font & global style (English-only for reliability)
# ============================================================
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

# Global palette
COLORS = {
    "N=1000": "#2196F3",
    "N=2048": "#FF9800",
    "N=4096": "#E91E63",
    "Ideal":  "#333333",
    "Block":  "#90CAF9",
    "Cyclic": "#FFCC80",
    "Hybrid": "#A5D6A7",
    "Serial": "#BDBDBD",
    "Node1":  "#4CAF50",
    "Node2":  "#FF5722",
}
MARKERS    = {"N=1000": "o", "N=2048": "s", "N=4096": "D"}
LINESTYLES  = {"N=1000": "-", "N=2048": "--", "N=4096": "-."}

# ============================================================
# Data parsing
# ============================================================
PATTERN = re.compile(
    r"\[RESULT\]\s+ALG:\s+(\S+),\s+N:\s+(\d+),\s+P:\s+(\d+),\s+EXEC_TIME:\s+([\d.e+\-]+)"
)

def load_data():
    """Parse all .o files -> data[alg][N][P] = [(time, is_2node), ...]"""
    data = {}
    for fname in sorted(glob.glob("*.o")):
        is_2node = "_n2" in os.path.basename(fname)
        with open(fname) as f:
            for line in f:
                m = PATTERN.match(line.strip())
                if m:
                    alg = m.group(1)
                    N   = int(m.group(2))
                    P   = int(m.group(3))
                    t   = float(m.group(4))
                    data.setdefault(alg, {}).setdefault(N, {}).setdefault(P, []).append((t, is_2node))
    return data

def get_time(data, alg, N, P, prefer_2node=False):
    """Get exec time for given alg/N/P. For P=8, prefer_2node picks 2-node entry."""
    entries = data.get(alg, {}).get(N, {}).get(P, [])
    if not entries:
        return None
    node1 = [t for t, n2 in entries if not n2]
    node2 = [t for t, n2 in entries if n2]
    if prefer_2node and node2:
        return node2[0]
    if node1:
        return node1[0]
    return entries[0][0]

def get_both_p8(data, alg, N):
    """Return (1node_time, 2node_time) for P=8."""
    entries = data.get(alg, {}).get(N, {}).get(8, [])
    t1 = t2 = None
    for t, is_2node in entries:
        if is_2node:
            t2 = t
        else:
            t1 = t
    return t1, t2

def calc_speedup(data, alg, N, P_list):
    """Compute speedup S = T(1) / T(P)."""
    t1 = get_time(data, alg, N, 1)
    if t1 is None or t1 == 0:
        return {p: None for p in P_list}
    return {p: (t1 / tp if tp and tp > 0 else None)
            for p, tp in ((p, get_time(data, alg, N, p)) for p in P_list)}

# ============================================================
# Load data
# ============================================================
os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
data = load_data()
if not data:
    print("ERROR: No .o files or [RESULT] lines found.")
    exit(1)

P_LIST     = [1, 2, 4, 8, 16]
N_SPEEDUP  = [1000, 2048, 4096]
ALG_CYCLIC = "MPI_Cyclic"
DPI        = 300

# ============================================================
# Fig 1 — Speedup curves (MPI_Cyclic)
# ============================================================
def plot_fig1_speedup():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for N in N_SPEEDUP:
        sp = calc_speedup(data, ALG_CYCLIC, N, P_LIST)
        xy = [(p, sp[p]) for p in P_LIST if sp.get(p) is not None]
        if xy:
            xs, ys = zip(*xy)
            label = f"N={N}"
            ax.plot(xs, ys, color=COLORS[label], marker=MARKERS[label],
                    linestyle=LINESTYLES[label], linewidth=2, markersize=8, label=label)

    ax.plot(P_LIST, P_LIST, color=COLORS["Ideal"], linestyle=":", linewidth=2,
            marker="x", markersize=7, label="Ideal Speedup (y=P)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(P_LIST)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of Processes P", fontsize=13)
    ax.set_ylabel("Speedup S = T1 / Tp", fontsize=13)
    ax.set_title("MPI_Cyclic Speedup Trends at Different Matrix Sizes", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0.8, 22)
    fig.tight_layout()
    fig.savefig("fig1_speedup.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig1_speedup.png")

# ============================================================
# Fig 2 — Load balance: Block vs Cyclic (N=4096)
# ============================================================
def plot_fig2_load_balance():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    N = 4096
    blk = [get_time(data, "MPI_Block", N, p) for p in P_LIST]
    cyc = [get_time(data, ALG_CYCLIC, N, p) for p in P_LIST]

    x = np.arange(len(P_LIST))
    w = 0.35
    b1 = ax.bar(x - w/2, blk, w, color=COLORS["Block"], edgecolor="#1565C0", linewidth=0.8,
                label="MPI_Block (contiguous)")
    b2 = ax.bar(x + w/2, cyc, w, color=COLORS["Cyclic"], edgecolor="#E65100", linewidth=0.8,
                label="MPI_Cyclic (round-robin)")

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.15, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.15, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in P_LIST])
    ax.set_xlabel("Number of Processes P", fontsize=13)
    ax.set_ylabel("Execution Time (s)", fontsize=13)
    ax.set_title("Impact of Row Distribution Strategy on Execution Time (N=4096)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_load_balance.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig2_load_balance.png")

# ============================================================
# Fig 3 — Topology: 1-Node vs 2-Node at P=8
# ============================================================
def plot_fig3_topology():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))

    for ax, N, subtitle in [(ax1, 1000, "N=1000"), (ax2, 4096, "N=4096")]:
        t1_blk, t2_blk = get_both_p8(data, "MPI_Block", N)
        t1_cyc, t2_cyc = get_both_p8(data, ALG_CYCLIC, N)
        t_n1 = t1_cyc if t1_cyc is not None else t1_blk
        t_n2 = t2_cyc if t2_cyc is not None else t2_blk

        cats = ["1 Node x8 Cores", "2 Nodes x4 Cores"]
        times = [t_n1, t_n2]
        bars = ax.bar(cats, times, color=[COLORS["Node1"], COLORS["Node2"]],
                      edgecolor="#333", linewidth=0.8, width=0.5)

        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{t:.4f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(subtitle, fontsize=14, fontweight="bold")
        ax.set_ylabel("Execution Time (s)", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        if t1_cyc is not None and t1_blk is not None:
            faster = "Cyclic" if t1_cyc < t1_blk else "Block"
            ax.text(0.5, 0.92, f"1-node best: {faster}",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    fig.suptitle("Time Comparison: Same Process Count (P=8) Under Different Topologies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("fig3_topology.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig3_topology.png")

# ============================================================
# Fig 4 — Hybrid oversubscription (N=4096)
# ============================================================
def plot_fig4_hybrid():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    N = 4096
    hyb = [get_time(data, "MPI_Hybrid", N, p) for p in P_LIST]
    cyc = [get_time(data, ALG_CYCLIC, N, p) for p in P_LIST]
    ser = get_time(data, "Serial", N, 1)

    ax.plot(P_LIST, hyb, color="#2E7D32", marker="D", linestyle="-", linewidth=2.5,
            markersize=9, label="MPI_Hybrid (MPI+OpenMP+NEON)")
    ax.plot(P_LIST, cyc, color="#E65100", marker="s", linestyle="--", linewidth=2,
            markersize=8, label="MPI_Cyclic (MPI only)")

    if ser is not None:
        ax.axhline(y=ser, color=COLORS["Serial"], linestyle=":", linewidth=2,
                   label=f"Serial baseline ({ser:.1f}s)")

    if hyb[0] is not None:
        ax.annotate(f"P=1 best\n{hyb[0]:.2f}s",
                    xy=(1, hyb[0]),
                    xytext=(2.5, hyb[0] + 2),
                    arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.8),
                    fontsize=10, color="#2E7D32", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                              edgecolor="#2E7D32", alpha=0.8))

    ax.set_xscale("log", base=2)
    ax.set_xticks(P_LIST)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of Processes P", fontsize=13)
    ax.set_ylabel("Execution Time (s)", fontsize=13)
    ax.set_title("Performance of MPI+OpenMP+SIMD Hybrid Programming (N=4096)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0.8, 22)
    fig.tight_layout()
    fig.savefig("fig4_hybrid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig4_hybrid.png")

# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("Generating charts...")
    plot_fig1_speedup()
    plot_fig2_load_balance()
    plot_fig3_topology()
    plot_fig4_hybrid()
    print("Done — 4 figures saved.")
