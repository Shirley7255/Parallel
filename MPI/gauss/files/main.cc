#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <arm_neon.h>

using namespace std;


//算法 1: gauss_serial — 串行基线高斯消去
 
void gauss_serial(float* A, int N) {
    for (int k = 0; k < N - 1; ++k) {
        // 主元 A[k][k] 在对角占优假设下非零
        float pivot_val = A[k * N + k];

        // 遍历主元下方的所有行
        for (int i = k + 1; i < N; ++i) {
            // 计算消去因子
            float factor = A[i * N + k] / pivot_val;

            // 更新行 i 的第 k+1 到 N-1 列
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] -= factor * A[k * N + j];
            }
        }
    }
}


//算法 2: gauss_mpi_block — MPI 块划分高斯消去
 
void gauss_mpi_block(float* A, int N, int rank, int size) {
    // 计算本进程负责的行范围 [start_row, end_row)
    int rows_per_proc = N / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? N : (rank + 1) * rows_per_proc;

    // 主元行接收缓冲区（所有进程统一分配）
    float* pivot_row = new float[N];

    for (int k = 0; k < N - 1; ++k) {
        // --- 确定主元行 k 的拥有者 ---
        int owner = k / rows_per_proc;
        if (owner >= size) owner = size - 1;  // 钳位：最后一块收纳所有余数行

        // --- 拥有者提取主元行 ---
        if (rank == owner) {
            // 从本地全矩阵 A 中复制第 k 行
            for (int j = 0; j < N; ++j) {
                pivot_row[j] = A[k * N + j];
            }
        }

        // --- 广播主元行：使所有进程获得一致的 pivot_row ---
        MPI_Bcast(pivot_row, N, MPI_FLOAT, owner, MPI_COMM_WORLD);

        float pivot_val = pivot_row[k];

        // --- 各进程消去自己负责的、位于主元行之下的行 ---
        // 计算本进程实际需要消去的起始行：max(k+1, start_row)
        int my_begin = (k + 1 > start_row) ? (k + 1) : start_row;
        for (int i = my_begin; i < end_row; ++i) {
            float factor = A[i * N + k] / pivot_val;
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] -= factor * pivot_row[j];
            }
        }
    }

    delete[] pivot_row;
}


//算法 3: gauss_mpi_cyclic — MPI 循环划分高斯消去
 
void gauss_mpi_cyclic(float* A, int N, int rank, int size) {
    float* pivot_row = new float[N];

    for (int k = 0; k < N - 1; ++k) {
        // --- 循环划分下，行 k 的拥有者 ---
        int owner = k % size;

        // --- 拥有者提取主元行 ---
        if (rank == owner) {
            for (int j = 0; j < N; ++j) {
                pivot_row[j] = A[k * N + j];
            }
        }

        // --- 广播主元行 ---
        MPI_Bcast(pivot_row, N, MPI_FLOAT, owner, MPI_COMM_WORLD);

        float pivot_val = pivot_row[k];

        // --- 各进程仅消去属于自己的行（row % size == rank 且 row > k）---
        // 使用步长 size 跳转，避免遍历所有行后判断归属
        // 起始行：大于 k 的第一个满足 i % size == rank 的行
        int start_i = k + 1;
        while (start_i < N && start_i % size != rank) {
            start_i++;
        }

        for (int i = start_i; i < N; i += size) {
            float factor = A[i * N + k] / pivot_val;
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] -= factor * pivot_row[j];
            }
        }
    }

    delete[] pivot_row;
}


//算法 4: gauss_mpi_hybrid — MPI + OpenMP + ARM NEON 混合编程

void gauss_mpi_hybrid(float* A, int N, int rank, int size) {
    float* pivot_row = new float[N];

    for (int k = 0; k < N - 1; ++k) {
        // --- 主元行拥有者（循环划分）---
        int owner = k % size;

        // --- 拥有者提取主元行 ---
        if (rank == owner) {
            for (int j = 0; j < N; ++j) {
                pivot_row[j] = A[k * N + j];
            }
        }

        // --- 广播主元行 ---
        MPI_Bcast(pivot_row, N, MPI_FLOAT, owner, MPI_COMM_WORLD);

        float pivot_val = pivot_row[k];

        // --- OpenMP 并行：多线程同时处理本进程拥有的行 ---
        // schedule(static) 将行范围均匀分块分配给各线程
        // 由于各行工作量相同（列数相同），static 调度最优
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < N; ++i) {
            // 跳过不属于本进程的行
            if (i % size != rank) continue;

            // 计算消去因子
            float factor = A[i * N + k] / pivot_val;

            // --- ARM NEON 向量化内层循环 ---
            // 将 -factor 广播到 NEON 寄存器，使用 vfmaq_f32 实现减法
            float32x4_t neg_factor = vdupq_n_f32(-factor);

            int j = k + 1;

            // 每次迭代处理 4 个 float（128-bit / 32-bit = 4）
            for (; j <= N - 4; j += 4) {
                // 加载 A[i][j..j+3] 的 4 个元素
                float32x4_t vec_A = vld1q_f32(&A[i * N + j]);
                // 加载 pivot_row[j..j+3] 的 4 个元素
                float32x4_t vec_pivot = vld1q_f32(&pivot_row[j]);
                // 融合乘加：vec_A + neg_factor * vec_pivot
                // 等价于 vec_A - factor * vec_pivot（即消去操作）
                float32x4_t vec_res = vfmaq_f32(vec_A, neg_factor, vec_pivot);
                // 将结果写回 A[i][j..j+3]
                vst1q_f32(&A[i * N + j], vec_res);
            }

            // --- 标量尾部：处理剩余不足 4 个的列 ---
            // （当前 N 值均为 4 的倍数，此循环作为安全兜底）
            for (; j < N; ++j) {
                A[i * N + j] -= factor * pivot_row[j];
            }
        }
    }

    delete[] pivot_row;
}


int main(int argc, char* argv[]) {
    // --- MPI 环境初始化 ---
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    vector<int> N_list = {500, 1000, 2048, 4096};

    
    // 主循环：遍历每个问题规模 N
    for (int N : N_list) {
        // 矩阵 A 的总元素数
        long long total_elements = (long long)N * N;

        // --- 矩阵指针定义 ---
        // Rank 0：拥有完整矩阵 + 备份
        // 其他进程：在 MPI 算法前通过 Bcast 接收完整矩阵
        float* A = new float[total_elements];
        float* A_backup = nullptr;

        if (rank == 0) {
            A_backup = new float[total_elements];

            // --- 随机初始化矩阵 A ---
            // 使用固定种子 2023 保证每次运行结果可复现
            mt19937 gen(2023);
            uniform_real_distribution<float> dis(1.0f, 10.0f);

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    A[i * N + j] = dis(gen);
                }
                // 对角占优处理：A[i][i] 增大 N*10.0
                // 确保主元非零，避免消去过程中出现除零错误
                // 同时保证数值稳定性，减少舍入误差的放大
                A[i * N + i] += N * 10.0f;
            }

            // --- 备份原始矩阵（用于每次算法测试前恢复）---
            memcpy(A_backup, A, total_elements * sizeof(float));
        }

        
        // 算法 1: gauss_serial — 串行基线
        {
            MPI_Barrier(MPI_COMM_WORLD);

            // 恢复原始矩阵
            if (rank == 0) {
                memcpy(A, A_backup, total_elements * sizeof(float));
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                auto t_start = chrono::high_resolution_clock::now();
                gauss_serial(A, N);
                auto t_end = chrono::high_resolution_clock::now();

                double exec_time = chrono::duration<double>(t_end - t_start).count();
                cout << "[RESULT] ALG: Serial, N: " << N
                     << ", P: " << size
                     << ", EXEC_TIME: " << exec_time << endl;
            }
        }

       
        // 算法 2: gauss_mpi_block — MPI 块划分
        {
            MPI_Barrier(MPI_COMM_WORLD);

            // Rank 0 恢复原始矩阵
            if (rank == 0) {
                memcpy(A, A_backup, total_elements * sizeof(float));
            }

            // 将完整矩阵 A 广播给所有进程
            MPI_Bcast(A, total_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // 同步：确保所有进程都已收到完整矩阵
            MPI_Barrier(MPI_COMM_WORLD);

            // --- 计时开始（仅包含消去过程）---
            double t_start = MPI_Wtime();
            gauss_mpi_block(A, N, rank, size);
            MPI_Barrier(MPI_COMM_WORLD);  // 等待所有进程完成消去
            double t_end = MPI_Wtime();

            if (rank == 0) {
                double exec_time = t_end - t_start;
                cout << "[RESULT] ALG: MPI_Block, N: " << N
                     << ", P: " << size
                     << ", EXEC_TIME: " << exec_time << endl;
            }
        }

       
        // 算法 3: gauss_mpi_cyclic — MPI 循环划分
        {
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                memcpy(A, A_backup, total_elements * sizeof(float));
            }

            MPI_Bcast(A, total_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            double t_start = MPI_Wtime();
            gauss_mpi_cyclic(A, N, rank, size);
            MPI_Barrier(MPI_COMM_WORLD);
            double t_end = MPI_Wtime();

            if (rank == 0) {
                double exec_time = t_end - t_start;
                cout << "[RESULT] ALG: MPI_Cyclic, N: " << N
                     << ", P: " << size
                     << ", EXEC_TIME: " << exec_time << endl;
            }
        }

 
        // 算法 4: gauss_mpi_hybrid — MPI + OpenMP + ARM NEON 混合编程

        {
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                memcpy(A, A_backup, total_elements * sizeof(float));
            }

            MPI_Bcast(A, total_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            double t_start = MPI_Wtime();
            gauss_mpi_hybrid(A, N, rank, size);
            MPI_Barrier(MPI_COMM_WORLD);
            double t_end = MPI_Wtime();

            if (rank == 0) {
                double exec_time = t_end - t_start;
                cout << "[RESULT] ALG: MPI_Hybrid, N: " << N
                     << ", P: " << size
                     << ", EXEC_TIME: " << exec_time << endl;
            }
        }

        // 释放内存 
        if (rank == 0) {
            delete[] A_backup;
        }
        delete[] A;

       
        if (rank == 0) {
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}