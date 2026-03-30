#include <iostream>
#include <windows.h>
#include <vector>

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

NOINLINE double sum_naive(const std::vector<double>& data) {
    double sum = 0.0;
    const size_t n = data.size();
    // 单路链式累加：每次加法都依赖上一次结果，指令并行度较低。
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

NOINLINE double sum_unrolled(const std::vector<double>& data) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;

    const size_t n = data.size();
    size_t i = 0;
    // 4 路独立累加链，减少相邻加法的写后读依赖，提升 ILP。
    for (; i + 3 < n; i += 4) {
        sum1 += data[i];
        sum2 += data[i + 1];
        sum3 += data[i + 2];
        sum4 += data[i + 3];
    }

    // 合并 4 路部分和，再处理尾部不足 4 个的元素。
    double sum = sum1 + sum2 + sum3 + sum4;
    for (; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

double benchmark_ms_naive(const std::vector<double>& data, int runs, volatile double& total_result) {
    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    // total_result 使用 volatile，避免编译器将求和循环优化掉。
    for (int r = 0; r < runs; ++r) {
        total_result += sum_naive(data);
    }
    QueryPerformanceCounter(&end);

    return (end.QuadPart - start.QuadPart) * 1000.0 / static_cast<double>(freq.QuadPart);
}

double benchmark_ms_unrolled(const std::vector<double>& data, int runs, volatile double& total_result) {
    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    // 与 naive 使用同样防优化策略，保证对比公平。
    for (int r = 0; r < runs; ++r) {
        total_result += sum_unrolled(data);
    }
    QueryPerformanceCounter(&end);

    return (end.QuadPart - start.QuadPart) * 1000.0 / static_cast<double>(freq.QuadPart);
}

int main() {
    const int n = 4096;
    const int runs = 5000000;

    std::cout << "n\truns\tnaive(ms)\tunrolled(ms)\tspeedup\tnaive_total\tunrolled_total\n";

    std::vector<double> data(static_cast<size_t>(n), 1.0);

    volatile double naive_total = 0.0;
    volatile double unrolled_total = 0.0;

    const double naive_ms = benchmark_ms_naive(data, runs, naive_total);
    const double unrolled_ms = benchmark_ms_unrolled(data, runs, unrolled_total);
    const double speedup = naive_ms / unrolled_ms;

    std::cout
        << n << '\t'
        << runs << '\t'
        << naive_ms << '\t'
        << unrolled_ms << '\t'
        << speedup << '\t'
        << static_cast<double>(naive_total) << '\t'
        << static_cast<double>(unrolled_total) << '\n';

    return 0;
}
