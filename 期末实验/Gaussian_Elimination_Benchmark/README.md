# 普通高斯消去统一并行化实验

本目录比较普通高斯消去前向消去阶段在 serial、AVX2、OpenMP、CUDA 和 MPI 架构上的性能。输入为固定随机种子生成的严格对角占优 `float` 矩阵，采用一维行优先连续存储，不进行主元选取。

## 实现

- `serial`：串行基线。
- `avx2_unaligned`：使用 AVX2 intrinsic，每次处理 8 个 `float`。
- `openmp_static`、`openmp_dynamic`、`openmp_guided`：固定 pivot 后并行消去各行。
- `openmp_avx2_dynamic`：OpenMP 行级并行与 AVX2 列级向量化组合。
- `cuda_2d_element`：二维线程网格按元素更新。
- `cuda_row_kernel`：每个 CUDA 线程负责一整行。
- MPI block/cyclic：连续块与循环行划分，每轮广播 pivot 行。

所有并行版本均与 serial baseline 比较，并输出 `max_abs_error`、`relative_residual` 和 `correct`。

## Windows 环境与正式实验

需要 Windows 11 x64、Visual Studio 2022 C++ Build Tools、CUDA Toolkit、PowerShell，以及安装了 `pandas`、`matplotlib` 的 Python 3。

    Set-ExecutionPolicy -Scope Process Bypass
    .\scripts\run_cpu_gpu.ps1
    python .\scripts\plot_results.py

正式实验测试 N=`512,1024,1536,2048`、线程数=`1,2,4,8,12,16`、repeat=5，结果写入 `results/cpu_gpu_results.csv`。

## 扩展测量

CPU 与 CUDA 临界规模实验：

    .\scripts\run_crossover.ps1
    python .\scripts\plot_crossover.py

CUDA cold/warm 启动对比：

    .\scripts\run_cuda_warmup_compare.ps1
    python .\scripts\plot_cuda_warmup.py

## MPI（Linux/WSL）

    mkdir -p build
    mpic++ -O3 -std=c++17 src/gauss_mpi.cpp -o build/gauss_mpi
    mpirun -np 4 ./build/gauss_mpi --n 1024 --repeat 5

## 结果文件

- `results/cpu_gpu_results.csv`：正式实验原始结果。
- `results/crossover_results.csv`：CPU/CUDA 临界规模结果。
- `results/cuda_warmup_results.csv`：CUDA 冷启动与预热结果。
- `results/` 与 `results/figures/`：由绘图脚本生成的性能图表。

CUDA 首次运行可能受到上下文初始化影响；线程数过多会增加同步与调度成本；GPU 在小规模矩阵上也可能因 kernel 启动开销而慢于 CPU。
