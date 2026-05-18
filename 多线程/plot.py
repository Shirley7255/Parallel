import pandas as pd
import matplotlib.pyplot as plt
import io
import re

# 1. 读取并清洗数据
with open('data.csv', 'r', encoding='utf-16', errors='ignore') as f:
    lines = f.readlines()

csv_lines = []
for line in lines:
    if line.startswith('Algorithm') or re.match(r'^[A-Za-z0-9_+]+,\d+,\d+,[\d\.]+', line):
        csv_lines.append(line)

csv_data = ''.join(csv_lines)
df = pd.read_csv(io.StringIO(csv_data))

# 2. 全局样式设置 (学术风)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# =========================================================================
# Fig 1: 不同优化算法在不同规模 (N) 下的时间对比折线图
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

ns = sorted(df['N'].unique())
algos_to_plot = [
    ('Serial', 1, 'o-', 'Serial'),
    ('AVX2', 1, 's--', 'AVX2'),
    ('OpenMP', 4, '^-.', 'OpenMP (T=4)'),
    ('Pthread', 4, 'd:', 'Pthread (T=4)'),
    ('Pthread+AVX2', 4, '*--', 'Pthread+AVX2 (T=4)')
]

for algo, threads, fmt, label in algos_to_plot:
    subset = df[(df['Algorithm'] == algo) & (df['Threads'] == threads)]
    if not subset.empty:
        subset = subset.sort_values(by='N')
        ax1.plot(subset['N'], subset['Time(ms)'], fmt, label=label, markersize=8, linewidth=2)

ax1.set_xlabel('Matrix Dimension (N)')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Performance Scaling with Matrix Size (N)')
ax1.set_xticks(ns)
ax1.set_xticklabels(ns)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fancybox=True, shadow=True)

# 留足够顶部空间
ylim_max = ax1.get_ylim()[1]
ax1.set_ylim(0, ylim_max * 1.15)

plt.tight_layout()
fig1.savefig('Fig1_Algorithm_Scaling.png')
plt.close(fig1)

# =========================================================================
# Fig 2: 不同多线程策略在不同线程数下的扩展性对比图 (1x2)
# =========================================================================
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
threads_list = [1, 2, 4, 8]

# --- Left Plot: N=1024 ---
df_1024 = df[df['N'] == 1024]
serial_1024_val = df_1024[(df_1024['Algorithm'] == 'Serial') & (df_1024['Threads'] == 1)]['Time(ms)']
if not serial_1024_val.empty:
    s_val = serial_1024_val.values[0]
    ax2.axhline(s_val, color='gray', linestyle='--', linewidth=1.5, label='Serial Baseline')

for algo, fmt in [('OpenMP', 'o-'), ('Pthread', 's-')]:
    subset = df_1024[df_1024['Algorithm'] == algo].sort_values(by='Threads')
    # Prepend thread=1 as serial_val if algo is OpenMP/Pthread (usually OpenMP T=1 isn't purely serial but close)
    # Actually, the user asked to let the lines come out from the serial point, or draw a horiz line. We drew horiz line.
    if not subset.empty:
        ax2.plot(subset['Threads'], subset['Time(ms)'], fmt, label=algo, markersize=8, linewidth=2)

ax2.set_xlabel('Number of Threads')
ax2.set_ylabel('Execution Time (ms)')
ax2.set_title('Thread Scalability at N=1024')
ax2.set_xticks([1, 2, 4, 8])
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
ax2.set_ylim(bottom=0)

# --- Right Plot: N=2048 ---
df_2048 = df[df['N'] == 2048]
serial_2048_val = df_2048[(df_2048['Algorithm'] == 'Serial')]['Time(ms)']
avx2_2048_val = df_2048[(df_2048['Algorithm'] == 'AVX2')]['Time(ms)']

if not serial_2048_val.empty:
    s_val_2048 = serial_2048_val.values[0]
    ax3.axhline(s_val_2048, color='gray', linestyle='--', linewidth=1.5, label='Serial Baseline')

if not avx2_2048_val.empty:
    a_val_2048 = avx2_2048_val.values[0]
    ax3.axhline(a_val_2048, color='purple', linestyle=':', linewidth=1.5, label='AVX2 Baseline (T=1)')

for algo, fmt in [('OpenMP', 'o-'), ('Pthread', 's-'), ('Pthread+AVX2', '^--'), ('Pthread+AVX2_Unroll', 'd:')]:
    subset = df_2048[df_2048['Algorithm'] == algo].sort_values(by='Threads')
    if not subset.empty:
        ax3.plot(subset['Threads'], subset['Time(ms)'], fmt, label=algo, markersize=8, linewidth=2)

ax3.set_xlabel('Number of Threads')
ax3.set_ylabel('Execution Time (ms)')
ax3.set_title('Thread Scalability at N=2048')
ax3.set_xticks([1, 2, 4, 8])
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
ax3.set_ylim(bottom=0)

plt.tight_layout()
fig2.savefig('Fig2_Thread_Scalability.png')
plt.close(fig2)
