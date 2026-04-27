#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <malloc.h>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <cmath>

inline int idx(int row, int col, int n) {
    return row * n + col;
}

float* alloc_aligned_matrix(int n) {
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    void* ptr = _aligned_malloc(bytes, 32);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(ptr);
}

void free_aligned_matrix(float* mat) {
    _aligned_free(mat);
}

// 初始化为上三角矩阵：
// 1) 对角线 = 1.0f
// 2) 下三角 = 0.0f
// 3) 上三角 = 随机值
// 该构造确保主元初始不为 0，避免除零风险。
void init_matrix(float* a, int n, uint32_t seed = 42u) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);

    std::memset(a, 0, static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float));

    for (int i = 0; i < n; ++i) {
        a[idx(i, i, n)] = 1.0f;
        for (int j = i + 1; j < n; ++j) {
            a[idx(i, j, n)] = dist(rng);
        }
    }
}

// 朴素串行高斯消去（仅前向消去，不回代）
void gaussian_elimination_serial(float* a, int n) {
    for (int k = 0; k < n; ++k) {
        const float pivot = a[idx(k, k, n)];

        // 对本构造数据，pivot 不会为 0；保留检查以增强健壮性。
        if (pivot == 0.0f) {
            continue;
        }

        for (int i = k + 1; i < n; ++i) {
            const float factor = a[idx(i, k, n)] / pivot;
            a[idx(i, k, n)] = 0.0f;

            for (int j = k + 1; j < n; ++j) {
                a[idx(i, j, n)] -= factor * a[idx(k, j, n)];
            }
        }
    }
}

// 进阶测试：编译器自动向量化版本
// 通过 GCC 的 pragma 强制对该函数开启 O3 极速优化和树状向量化
#pragma GCC push_options
#pragma GCC optimize("O3", "tree-vectorize")
void gaussian_elimination_autovec(float* a, int n) {
    for (int k = 0; k < n; ++k) {
        const float pivot = a[idx(k, k, n)];
        if (pivot == 0.0f) continue;

        for (int i = k + 1; i < n; ++i) {
            const float factor = a[idx(i, k, n)] / pivot;
            a[idx(i, k, n)] = 0.0f;

            // 编译器会自动识别这个内层循环，并将其转换为 AVX 指令
            for (int j = k + 1; j < n; ++j) {
                a[idx(i, j, n)] -= factor * a[idx(k, j, n)];
            }
        }
    }
}
#pragma GCC pop_options

// AVX2 非对齐版本高斯消去（仅前向消去，不回代）
// 使用 _mm256_loadu_ps / _mm256_storeu_ps，并在每段向量循环后处理尾部标量。
void GaussianElimination_AVX_Unaligned(float* a, int n) {
    for (int k = 0; k < n; ++k) {
        const float pivot = a[idx(k, k, n)];
        if (pivot == 0.0f) {
            continue;
        }

        // 1) 除法循环: A[k][j] = A[k][j] / A[k][k], j = k + 1 .. n - 1
        const __m256 pivot_vec = _mm256_set1_ps(pivot);
        int j = k + 1;
        for (; j + 8 <= n; j += 8) {
            __m256 row_k = _mm256_loadu_ps(&a[idx(k, j, n)]);
            row_k = _mm256_div_ps(row_k, pivot_vec);
            _mm256_storeu_ps(&a[idx(k, j, n)], row_k);
        }
        for (; j < n; ++j) {
            a[idx(k, j, n)] = a[idx(k, j, n)] / pivot;
        }
        a[idx(k, k, n)] = 1.0f;

        // 2) 消去循环: A[i][j] = A[i][j] - A[i][k] * A[k][j], i = k + 1 .. n - 1
        for (int i = k + 1; i < n; ++i) {
            const float aik = a[idx(i, k, n)];
            const __m256 aik_vec = _mm256_set1_ps(aik);

            int jj = k + 1;
            for (; jj + 8 <= n; jj += 8) {
                __m256 row_i = _mm256_loadu_ps(&a[idx(i, jj, n)]);
                const __m256 row_k = _mm256_loadu_ps(&a[idx(k, jj, n)]);
                const __m256 prod = _mm256_mul_ps(aik_vec, row_k);
                row_i = _mm256_sub_ps(row_i, prod);
                _mm256_storeu_ps(&a[idx(i, jj, n)], row_i);
            }
            for (; jj < n; ++jj) {
                a[idx(i, jj, n)] -= aik * a[idx(k, jj, n)];
            }

            a[idx(i, k, n)] = 0.0f;
        }
    }
}

// AVX2 对齐版本高斯消去（仅前向消去，不回代）
// 关键点：
// 1) 不直接使用 loadu/storeu，而是先用标量代码把 j 推进到 32 字节对齐地址。
// 2) 一旦对齐后，使用 _mm256_load_ps / _mm256_store_ps 批量处理。
// 3) 末尾不足 8 个 float 的部分，用标量收尾，避免越界。
void GaussianElimination_AVX_Aligned(float* a, int n) {
    for (int k = 0; k < n; ++k) {
        const float pivot = a[idx(k, k, n)];
        if (pivot == 0.0f) {
            continue;
        }

        const __m256 pivot_vec = _mm256_set1_ps(pivot);

        // ---------------------------
        // 1) 除法循环（处理第 k 行）
        // A[k][j] = A[k][j] / A[k][k], j = k + 1 .. n - 1
        // ---------------------------
        int j = k + 1;

        // 前缀标量剥离：
        // 每次检查当前元素地址 addr = &A[k][j]。
        // 若 addr % 32 != 0，则还未到 AVX 对齐边界，先做一个标量元素并 j++。
        // 当 addr % 32 == 0 时，说明可安全使用 _mm256_load_ps/_store_ps。
        for (; j < n; ++j) {
            const uintptr_t addr = reinterpret_cast<uintptr_t>(&a[idx(k, j, n)]);
            if ((addr % 32u) == 0u) {
                break;
            }
            a[idx(k, j, n)] = a[idx(k, j, n)] / pivot;
        }

        // 对齐后的主体向量循环（每次处理 8 个 float = 32 字节）
        for (; j + 8 <= n; j += 8) {
            __m256 row_k = _mm256_load_ps(&a[idx(k, j, n)]);
            row_k = _mm256_div_ps(row_k, pivot_vec);
            _mm256_store_ps(&a[idx(k, j, n)], row_k);
        }

        // 尾部标量收尾：处理不足 8 个元素，确保不越界
        for (; j < n; ++j) {
            a[idx(k, j, n)] = a[idx(k, j, n)] / pivot;
        }
        a[idx(k, k, n)] = 1.0f;

        // ---------------------------
        // 2) 消去循环（处理第 k 列以下各行）
        // A[i][j] = A[i][j] - A[i][k] * A[k][j]
        // i = k + 1 .. n - 1, j = k + 1 .. n - 1
        // ---------------------------
        for (int i = k + 1; i < n; ++i) {
            const float aik = a[idx(i, k, n)];
            const __m256 aik_vec = _mm256_set1_ps(aik);

            int jj = k + 1;

            // 前缀标量剥离：这里需要同时从 A[i][jj] 读写和 A[k][jj] 读取。
            // 为了使用对齐指令，要求这两个地址都在 32 字节边界。
            // 若任一地址不对齐，则先做标量并 jj++，直到都对齐或到达行尾。
            for (; jj < n; ++jj) {
                const uintptr_t addr_i = reinterpret_cast<uintptr_t>(&a[idx(i, jj, n)]);
                const uintptr_t addr_k = reinterpret_cast<uintptr_t>(&a[idx(k, jj, n)]);
                if ((addr_i % 32u) == 0u && (addr_k % 32u) == 0u) {
                    break;
                }
                a[idx(i, jj, n)] -= aik * a[idx(k, jj, n)];
            }

            // 对齐后的主体向量循环
            for (; jj + 8 <= n; jj += 8) {
                __m256 row_i = _mm256_load_ps(&a[idx(i, jj, n)]);
                const __m256 row_k = _mm256_load_ps(&a[idx(k, jj, n)]);
                const __m256 prod = _mm256_mul_ps(aik_vec, row_k);
                row_i = _mm256_sub_ps(row_i, prod);
                _mm256_store_ps(&a[idx(i, jj, n)], row_i);
            }

            // 尾部标量收尾：处理不足 8 个元素，确保不越界
            for (; jj < n; ++jj) {
                a[idx(i, jj, n)] -= aik * a[idx(k, jj, n)];
            }

            a[idx(i, k, n)] = 0.0f;
        }
    }
}

bool checkResult(float* serial_mat, float* simd_mat, int N) {
    const float eps = 1e-4f;
    const int total = N * N;
    for (int i = 0; i < total; ++i) {
        if (std::fabs(serial_mat[i] - simd_mat[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    try {
        std::vector<int> test_sizes = {256, 512, 1024};

        for (int n : test_sizes) {
            const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);

            float* base = alloc_aligned_matrix(n);
            float* serial_mat = alloc_aligned_matrix(n);
            float* avx_unaligned_mat = alloc_aligned_matrix(n);
            float* avx_aligned_mat = alloc_aligned_matrix(n);
            float* autovec_mat = alloc_aligned_matrix(n);

            init_matrix(base, n, 42u + static_cast<uint32_t>(n));
            std::memcpy(serial_mat, base, bytes);
            std::memcpy(avx_unaligned_mat, base, bytes);
            std::memcpy(avx_aligned_mat, base, bytes);
            std::memcpy(autovec_mat, base, bytes);

            const auto s1 = std::chrono::high_resolution_clock::now();
            gaussian_elimination_serial(serial_mat, n);
            const auto e1 = std::chrono::high_resolution_clock::now();

            const auto s2 = std::chrono::high_resolution_clock::now();
            GaussianElimination_AVX_Unaligned(avx_unaligned_mat, n);
            const auto e2 = std::chrono::high_resolution_clock::now();

            const auto s3 = std::chrono::high_resolution_clock::now();
            GaussianElimination_AVX_Aligned(avx_aligned_mat, n);
            const auto e3 = std::chrono::high_resolution_clock::now();

            const auto s4 = std::chrono::high_resolution_clock::now();
            gaussian_elimination_autovec(autovec_mat, n);
            const auto e4 = std::chrono::high_resolution_clock::now();

            const std::chrono::duration<double, std::milli> serial_ms = e1 - s1;
            const std::chrono::duration<double, std::milli> avx_unaligned_ms = e2 - s2;
            const std::chrono::duration<double, std::milli> avx_aligned_ms = e3 - s3;
            const std::chrono::duration<double, std::milli> autovec_ms = e4 - s4;

            const double speedup_unaligned = serial_ms.count() / avx_unaligned_ms.count();
            const double speedup_aligned = serial_ms.count() / avx_aligned_ms.count();
            const double speedup_autovec = serial_ms.count() / autovec_ms.count();

            const bool ok_unaligned = checkResult(serial_mat, avx_unaligned_mat, n);
            const bool ok_aligned = checkResult(serial_mat, avx_aligned_mat, n);
            const bool ok_autovec = checkResult(serial_mat, autovec_mat, n);

            std::cout << "N = " << n << std::endl;
            std::cout << "  Serial Time           : " << serial_ms.count() << " ms" << std::endl;
            std::cout << "  AVX Unaligned Time    : " << avx_unaligned_ms.count() << " ms"
                      << "  (Speedup: " << speedup_unaligned << "x)" << std::endl;
            std::cout << "  AVX Aligned Time      : " << avx_aligned_ms.count() << " ms"
                      << "  (Speedup: " << speedup_aligned << "x)" << std::endl;
            std::cout << "  Auto-Vectorization Time : " << autovec_ms.count() << " ms"
                      << "  (Speedup: " << speedup_autovec << "x)" << std::endl;
            std::cout << "  Check AVX Unaligned   : " << (ok_unaligned ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Check AVX Aligned     : " << (ok_aligned ? "PASS" : "FAIL") << std::endl;
            std::cout << "  Check Auto-Vectorization: " << (ok_autovec ? "PASS" : "FAIL") << std::endl;
            std::cout << std::endl;

            free_aligned_matrix(avx_aligned_mat);
            free_aligned_matrix(avx_unaligned_mat);
            free_aligned_matrix(serial_mat);
            free_aligned_matrix(base);
            free_aligned_matrix(autovec_mat);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
