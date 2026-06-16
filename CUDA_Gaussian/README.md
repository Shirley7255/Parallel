# CUDA 高斯消去实验

## 实验目的

本项目实现“高斯消去选题的 GPU 编程”实验，包含 CPU 串行 baseline 和 CUDA GPU 并行前向消去版本，并对不同矩阵规模下的运行时间和正确性进行对比。

## 文件说明

- `gaussian_elimination_cuda.cu`：CUDA C++ 主程序，包含数据生成、CPU 求解、GPU 前向消去、CPU 回代、误差验证和 CSV 输出。
- `run.ps1`：PowerShell 自动编译运行脚本。
- `results.csv`：程序运行后生成的实验结果表格。
- `gaussian_elimination_cuda.exe`：编译后生成的可执行文件。

## 算法原理

线性方程组为 `Ax=b`。高斯消去分为两个阶段：

1. 前向消去：第 `k` 轮以 `A[k][k]` 为主元，将第 `k` 行以下的 `A[i][k]` 消为 0，并同步更新右端向量 `b[i]`。
2. 回代求解：在得到上三角矩阵后，从最后一行向上依次求出未知量。

程序随机生成严格对角占优矩阵：

`abs(A[i][i]) > sum(abs(A[i][j])), j != i`

这样可以避免主元为 0，并提高数值稳定性。程序会保留原始 `A` 和 `b`，最后用 GPU 解计算 `Ax-b` 的最大残差。

## CUDA 并行化设计

GPU 版本将前向消去并行化。每一轮 pivot `k` 后启动一次 CUDA kernel：

- 二维线程划分待更新区域。
- `y` 方向对应第 `k` 行以下的待消去行。
- `x` 方向对应第 `k+1` 列到最后一列的待更新元素。
- 每行由 `colOffset == 0` 的线程同步更新 `b[i]` 并将 `A[i][k]` 置 0。
- 每轮 kernel 结束后使用 `cudaDeviceSynchronize()` 保证当前 pivot 消去完成，再进入下一轮。

回代部分放在 CPU 上完成，这样实现更清晰，便于和 CPU baseline 对照。

## 编译和运行

在 PowerShell 中进入本目录后运行：

```powershell
.\run.ps1
```

如果 PowerShell 执行策略限制脚本运行，可以在当前终端临时放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

`run.ps1` 会自动处理环境：

1. 优先尝试直接调用 `nvcc.exe`。
2. 如果 `nvcc.exe` 不在 PATH 中，则使用 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe`。
3. 如果当前环境找不到 `cl.exe`，则通过 Visual Studio 2022 的 `VsDevCmd.bat -arch=x64` 初始化 C++ 编译环境后再调用 nvcc。

## 输出字段含义

- `N`：矩阵规模。
- `CPU Time(ms)`：CPU 串行完整高斯消去和回代时间。
- `GPU Elim(ms)`：CUDA kernel 前向消去时间，使用 `cudaEvent_t` 计时。
- `GPU Total(ms)`：GPU 版本总时间，包括显存分配、数据传输、GPU 前向消去、传回结果和 CPU 回代。
- `Speedup`：`CPU Time / GPU Total Time`。
- `Max Error`：CPU 解和 GPU 解之间的最大绝对误差。
- `Residual`：用原始 `A` 和原始 `b` 计算 GPU 解的 `Ax-b` 最大残差。

## 实验结果

本机环境运行结果如下：

```text
Device: NVIDIA RTX 2000 Ada Generation Laptop GPU
Precision: double

N           CPU Time(ms)    GPU Elim(ms)   GPU Total(ms)     Speedup         Max Error          Residual
--------------------------------------------------------------------------------------------------------
256                2.700         118.561         120.329       0.022         2.776e-17         2.487e-14
512               20.659          11.212          13.701       1.508         1.388e-17         3.908e-14
768               67.425          27.567          31.601       2.134         6.939e-18         6.040e-14
1024             175.336          55.880          61.922       2.832         1.041e-17         9.237e-14

Correctness check: PASS
```

程序同时已生成 `results.csv`：

```csv
N,CPU Time(ms),GPU Elim(ms),GPU Total(ms),Speedup,Max Solution Error,Residual Max Error
256,2.7001,118.5608292,120.3288,0.02243934952,2.7755575616e-17,2.4868995752e-14
512,20.6593000000,11.2117757797,13.7010000000,1.5078680388,1.3877787808e-17,3.9079850467e-14
768,67.4251000000,27.5672321320,31.6008000000,2.1336516797,6.9388939039e-18,6.0396132540e-14
1024,175.3360000000,55.8796806335,61.9222000000,2.8315531425,1.0408340856e-17,9.2370555649e-14
```

本次结果通过正确性验证。CPU 解和 GPU 解的最大绝对误差约为 `1e-17`，GPU 解代回原始方程组后的最大残差约为 `1e-14`，说明结果稳定可靠。

## 结果分析说明

GPU 版本不一定在所有规模上明显加速，原因包括：

- 高斯消去每一轮 pivot 之间存在数据依赖。
- 每轮都要启动 kernel，kernel launch 开销较高。
- 小规模矩阵 GPU 利用率不足。
- 回代部分仍在 CPU 上执行。
- 数据传输和同步会影响 GPU 总时间。

本实验使用 double 精度。由于浮点运算顺序不同，CPU 和 GPU 结果可能存在微小差异；若误差在 `1e-5` 到 `1e-3` 范围内，通常属于并行浮点计算的正常舍入误差。本项目使用严格对角占优矩阵，实际误差通常应更小。
