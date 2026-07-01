# Groebner / Macaulay 有限域高斯消去实验

本目录实现“随机二次多项式系统 -> Macaulay 矩阵 -> GF(65521) 有限域高斯消去”的核心线性代数实验，用于观察 Groebner 基计算中模运算、稀疏矩阵 fill-in 和 OpenMP 行并行的性能特点。

本实验不是完整的 Buchberger、F4 或 F5 求解器；不包含 S-polynomial 选择、符号预处理或多项式约简队列。

## 环境与运行

- Windows 11 x64
- Visual Studio 2022 C++ Desktop Build Tools（MSVC）
- PowerShell 5.1 或 7
- Python 3.9+，以及 `pandas`、`matplotlib`

在本目录中运行：

    Set-ExecutionPolicy -Scope Process Bypass
    .\scripts\run_groebner.ps1
    python -m pip install pandas matplotlib
    python .\scripts\plot_groebner.py

运行脚本使用 `/O2 /openmp /std:c++17` 编译，测试 `(n_vars,D)=(4,4),(4,5),(5,5),(5,6),(6,6),(6,7)`，线程数为 `1,4,8,16`，每组重复 3 次。

结果保存在 `results/groebner_results.csv`，绘图输出位于 `results/figures/`。OpenMP 结果会与串行参考结果比较 rank 与 checksum，并在 CSV 中给出 `correct` 标记。
