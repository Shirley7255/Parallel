import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import os

# Try to find the exact file name
csv_file = 'data-arm.csv' if os.path.exists('data-arm.csv') else 'data_arm.csv'

# Read the data robustly
with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

csv_lines = []
for line in lines:
    line = line.strip()
    if line.startswith('Algorithm') or re.match(r'^[A-Za-z0-9\+\-\s]+,\s*\d+,\s*\d+,\s*[\d\.]+', line):
        csv_lines.append(line)

csv_data = '\n'.join(csv_lines)
df = pd.read_csv(io.StringIO(csv_data))
df.columns = [c.strip() for c in df.columns]

# Helper to get specific value
def get_time(algo, n, t):
    res = df[(df['Algorithm'].str.strip() == algo) & (df['N'] == n) & (df['Threads'] == t)]['Time(ms)']
    return res.values[0] if not res.empty else None

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# =========================================================================
# Fig 1: ARM_Fig1_NEON.png
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
ns = sorted(df['N'].unique())

for algo, fmt, label in [('Serial', 'o-', 'Serial (T=1)'), ('NEON', 's--', 'NEON (T=1)')]:
    times = [get_time(algo, n, 1) for n in ns]
    valid_n = [n for n, t in zip(ns, times) if t is not None]
    valid_t = [t for t in times if t is not None]
    ax1.plot(valid_n, valid_t, fmt, label=label, markersize=8, linewidth=2)

ax1.set_xlabel('Matrix Dimension (N)')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('ARM Platform: Algorithm Complexity and NEON Vectorization')
ax1.set_xticks(ns)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
fig1.savefig('ARM_Fig1_NEON.png')
plt.close(fig1)

# =========================================================================
# Fig 2: ARM_Fig2_Scalability.png
# =========================================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
threads = [1, 2, 4, 8]

for ax, n_val in zip(axes, [1024, 2048]):
    # Base T=1 points
    t_serial = get_time('Serial', n_val, 1)
    t_neon = get_time('NEON', n_val, 1)
    
    # OpenMP
    omp_times = [t_serial] + [get_time('OpenMP', n_val, t) for t in [2,4,8]]
    # Pthread
    pth_times = [t_serial] + [get_time('Pthread', n_val, t) for t in [2,4,8]]
    # Pthread+NEON
    pth_neon_times = [t_neon] + [get_time('Pthread+NEON', n_val, t) for t in [2,4,8]]
    
    ax.plot(threads, omp_times, 'o-', label='OpenMP', markersize=8, linewidth=2)
    ax.plot(threads, pth_times, 's-', label='Pthread', markersize=8, linewidth=2)
    ax.plot(threads, pth_neon_times, '^--', label='Pthread+NEON', markersize=8, linewidth=2)
    
    if t_serial:
        ax.axhline(t_serial, color='gray', linestyle=':', label='Serial Baseline')
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title(f'Thread Scalability at N={n_val}')
    ax.set_xticks(threads)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_ylim(bottom=0)

plt.tight_layout()
fig2.savefig('ARM_Fig2_Scalability.png')
plt.close(fig2)

# =========================================================================
# Fig 3: ARM_Fig3_Ultimate.png
# =========================================================================
fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=300)

labels = ['Serial (T=1)', 'Best Pthread (T=8)', 'Best OpenMP (T=8)', 'NEON (T=1)', 'Pthread+NEON (T=8)']
vals = [
    get_time('Serial', 2048, 1),
    get_time('Pthread', 2048, 8),
    get_time('OpenMP', 2048, 8),
    get_time('NEON', 2048, 1),
    get_time('Pthread+NEON', 2048, 8)
]

# Only plot if we have data
valid_indices = [i for i, v in enumerate(vals) if v is not None]
labels = [labels[i] for i in valid_indices]
vals = [vals[i] for i in valid_indices]

colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
colors = [colors[i] for i in valid_indices]

bars = ax3.bar(labels, vals, color=colors, alpha=0.8, edgecolor='black')

# Add values above bars
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + (max(vals)*0.01), f'{yval:.1f}ms', ha='center', va='bottom', fontweight='bold')

ax3.set_ylabel('Execution Time (ms)')
ax3.set_title('ARM Platform: Ultimate Performance Breakdown at N=2048')
ax3.set_xticklabels(labels, rotation=15, ha='right')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
fig3.savefig('ARM_Fig3_Ultimate.png')
plt.close(fig3)
