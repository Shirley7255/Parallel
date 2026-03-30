#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

// 数据初始化：b[i][j] = i + j，a[i] = 1。
void init_data(int n, std::vector<double>& a, std::vector<double>& b) {
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0;
        for (int j = 0; j < n; ++j) {
            b[i * n + j] = static_cast<double>(i + j);
        }
    }
}

// 平凡算法（Cache 不友好）：外层 i（列），内层 j（行）。
void dot_columns_naive(int n, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& sum) {
    for (int i = 0; i < n; ++i) {
        sum[i] = 0.0;
        // 固定列 i，按 b[j][i] 访问会跨行跳跃，局部性较差。
        for (int j = 0; j < n; ++j) {
            sum[i] += b[j * n + i] * a[j];
        }
    }
}

// Cache 优化算法（行主序访问）：外层 j（行），内层 i（列）。
void dot_columns_cache_friendly(int n, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& sum) {
    std::fill(sum.begin(), sum.end(), 0.0);
    for (int j = 0; j < n; ++j) {
        const double aj = a[j];
        const int row_base = j * n;
        // 固定行 j，连续访问 b[j][i]，符合行主序布局。
        for (int i = 0; i < n; ++i) {
            sum[i] += b[row_base + i] * aj;
        }
    }
}

template <typename Func>
double benchmark_ms(Func func, int runs) {
#ifdef _WIN32
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    for (int r = 0; r < runs; ++r) {
        func();
    }
    QueryPerformanceCounter(&end);

    return (end.QuadPart - start.QuadPart) * 1000.0 / static_cast<double>(freq.QuadPart);
#else
    const auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        func();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
#endif
}

int main() {
    const int n = 3000;
    const int runs = 50;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "n\truns\tnaive(ms)\toptimized(ms)\tspeedup\tchecksum\n";

    volatile double guard = 0.0;

    std::vector<double> a(n);
    std::vector<double> b(n * n);
    std::vector<double> sum(n);

    init_data(n, a, b);

    // 预热，减少首次执行带来的波动。
    dot_columns_naive(n, a, b, sum);
    guard += sum[0];
    dot_columns_cache_friendly(n, a, b, sum);
    guard += sum[0];

    const double naive_ms = benchmark_ms([&]() {
        dot_columns_naive(n, a, b, sum);
        guard += sum[n - 1];
    }, runs);

    const double optimized_ms = benchmark_ms([&]() {
        dot_columns_cache_friendly(n, a, b, sum);
        guard += sum[n - 1];
    }, runs);

    const double checksum = std::accumulate(sum.begin(), sum.end(), 0.0);
    const double speedup = naive_ms / optimized_ms;

    std::cout
        << n << '\t'
        << runs << '\t'
        << naive_ms << '\t'
        << optimized_ms << '\t'
        << speedup << '\t'
        << checksum << '\n';

    std::cout << "guard = " << guard << '\n';
    return 0;
}
