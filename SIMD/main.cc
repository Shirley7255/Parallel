#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>
#include <cstdlib>
#include <cmath>
#include <random>
// 自行添加需要的头文件

static inline std::size_t aligned_size(std::size_t bytes, std::size_t alignment)
{
    return ((bytes + alignment - 1) / alignment) * alignment;
}

void init_matrix(float *a, int n, unsigned seed = 2026)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            a[i * n + j] = dist(rng);
        }
        // 对角增强，降低无主元高斯消去出现零主元的概率
        a[i * n + i] += static_cast<float>(n);
    }
}

bool checkResult(const float *a, const float *b, int n, float eps = 1e-3f)
{
    const int total = n * n;
    for (int idx = 0; idx < total; ++idx)
    {
        if (std::fabs(a[idx] - b[idx]) > eps)
        {
            return false;
        }
    }
    return true;
}

void gaussian_elimination_serial(float *a, int n)
{
    for (int k = 0; k < n; ++k)
    {
        float pivot = a[k * n + k];
        for (int j = k + 1; j < n; ++j)
        {
            a[k * n + j] /= pivot;
        }
        a[k * n + k] = 1.0f;

        for (int i = k + 1; i < n; ++i)
        {
            float factor = a[i * n + k];
            for (int j = k + 1; j < n; ++j)
            {
                a[i * n + j] -= factor * a[k * n + j];
            }
            a[i * n + k] = 0.0f;
        }
    }
}

void GaussianElimination_NEON(float *a, int n)
{
    for (int k = 0; k < n; ++k)
    {
        float pivot = a[k * n + k];
        float32x4_t pivot_vec = vdupq_n_f32(pivot);

        int j = k + 1;
        for (; j + 4 <= n; j += 4)
        {
            float32x4_t row = vld1q_f32(&a[k * n + j]);
            row = vdivq_f32(row, pivot_vec);
            vst1q_f32(&a[k * n + j], row);
        }
        // 处理不能被4整除的尾部元素
        for (; j < n; ++j)
        {
            a[k * n + j] /= pivot;
        }
        a[k * n + k] = 1.0f;

        for (int i = k + 1; i < n; ++i)
        {
            float factor = a[i * n + k];
            float32x4_t factor_vec = vdupq_n_f32(factor);

            int col = k + 1;
            for (; col + 4 <= n; col += 4)
            {
                float32x4_t row_i = vld1q_f32(&a[i * n + col]);
                float32x4_t row_k = vld1q_f32(&a[k * n + col]);
                float32x4_t prod = vmulq_f32(factor_vec, row_k);
                row_i = vsubq_f32(row_i, prod);
                vst1q_f32(&a[i * n + col], row_i);
            }
            // 处理不能被4整除的尾部元素
            for (; col < n; ++col)
            {
                a[i * n + col] -= factor * a[k * n + col];
            }
            a[i * n + k] = 0.0f;
        }
    }
}

int main(int argc, char *argv[])
{
    auto Start = std::chrono::high_resolution_clock::now();
    // TODO : 完成你的高斯消元
    std::vector<int> sizes = {256, 512, 1024};
    std::cout << std::fixed << std::setprecision(3);

    for (int n : sizes)
    {
        std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);
        std::size_t alloc_bytes = aligned_size(bytes, 32);

        float *a_base = static_cast<float *>(aligned_alloc(32, alloc_bytes));
        float *a_serial = static_cast<float *>(aligned_alloc(32, alloc_bytes));
        float *a_neon = static_cast<float *>(aligned_alloc(32, alloc_bytes));

        if (a_base == nullptr || a_serial == nullptr || a_neon == nullptr)
        {
            std::cerr << "aligned_alloc failed for N=" << n << std::endl;
            std::free(a_base);
            std::free(a_serial);
            std::free(a_neon);
            return 1;
        }

        init_matrix(a_base, n, 2026 + static_cast<unsigned>(n));
        std::memcpy(a_serial, a_base, bytes);
        std::memcpy(a_neon, a_base, bytes);

        auto t1 = std::chrono::high_resolution_clock::now();
        gaussian_elimination_serial(a_serial, n);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> serial_ms = t2 - t1;

        auto t3 = std::chrono::high_resolution_clock::now();
        GaussianElimination_NEON(a_neon, n);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> neon_ms = t4 - t3;

        bool ok = checkResult(a_serial, a_neon, n);
        double speedup = serial_ms.count() / neon_ms.count();

        std::cout << "N=" << n << std::endl;
        std::cout << "  Serial Time : " << serial_ms.count() << " ms" << std::endl;
        std::cout << "  NEON Time   : " << neon_ms.count() << " ms" << std::endl;
        std::cout << "  Speedup     : " << speedup << "x" << std::endl;
        std::cout << "  Check       : " << (ok ? "PASS" : "FAIL") << std::endl;

        std::free(a_base);
        std::free(a_serial);
        std::free(a_neon);
    }

    // 如果需要文件输入输出, 请使用相对路径并放在 files 文件夹下
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli>elapsed = End - Start; // 注意这里我改成了milli以便输出毫秒
    std::cout<<"average latency  : "<<elapsed.count()<<" (ms) "<<std::endl;
    return 0;
}
