#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define IDX(i, j, N) ((std::size_t)(i) * (std::size_t)(N) + (std::size_t)(j))

struct Problem {
    int n{};
    std::vector<float> a, b;
};

struct Options {
    int n = 512;
    int threads = 1;
    int repeat = 5;
    std::string impl = "all";
    std::string warmup_mode = "";
    int run_index = 1;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto take = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::stoi(argv[++i]);
        };
        const std::string s = argv[i];
        if (s == "--n") o.n = take("--n");
        else if (s == "--threads") o.threads = take("--threads");
        else if (s == "--repeat") o.repeat = take("--repeat");
        else if (s == "--run-index") o.run_index = take("--run-index");
        else if (s == "--impl") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --impl");
            o.impl = argv[++i];
        } else if (s == "--warmup-mode") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --warmup-mode");
            o.warmup_mode = argv[++i];
        } else throw std::runtime_error("unknown argument: " + s);
    }
    if (o.n < 2 || o.threads < 1 || o.repeat < 1 || o.run_index < 1) throw std::runtime_error("arguments must be positive");
    return o;
}

inline Problem make_problem(int n, std::uint32_t seed = 20260629u) {
    Problem p{n, std::vector<float>((std::size_t)n * n), std::vector<float>(n)};
    std::mt19937 rng(seed + (std::uint32_t)n);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const float v = d(rng);
            p.a[IDX(i, j, n)] = v;
            sum += std::abs((double)v);
        }
        p.a[IDX(i, i, n)] = (float)(sum + 2.0);
        p.b[i] = d(rng);
    }
    return p;
}

inline void gauss_serial(std::vector<float>& a, std::vector<float>& b, int n) {
    for (int k = 0; k < n - 1; ++k) {
        const float pivot = a[IDX(k, k, n)];
        for (int i = k + 1; i < n; ++i) {
            const float factor = a[IDX(i, k, n)] / pivot;
            a[IDX(i, k, n)] = 0.0f;
            for (int j = k + 1; j < n; ++j) a[IDX(i, j, n)] -= factor * a[IDX(k, j, n)];
            b[i] -= factor * b[k];
        }
    }
}

inline std::vector<float> back_substitute(const std::vector<float>& u, const std::vector<float>& y, int n) {
    std::vector<float> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= (double)u[IDX(i, j, n)] * x[j];
        x[i] = (float)(s / u[IDX(i, i, n)]);
    }
    return x;
}

inline double relative_residual(const Problem& original, const std::vector<float>& u,
                                const std::vector<float>& y) {
    const auto x = back_substitute(u, y, original.n);
    long double nr = 0.0, nb = 0.0;
    for (int i = 0; i < original.n; ++i) {
        long double ax = 0.0;
        for (int j = 0; j < original.n; ++j) ax += (long double)original.a[IDX(i, j, original.n)] * x[j];
        const long double r = ax - original.b[i];
        nr += r * r;
        nb += (long double)original.b[i] * original.b[i];
    }
    return (double)(std::sqrt(nr) / std::max(std::sqrt(nb), (long double)1e-30));
}

inline double max_abs_error(const std::vector<float>& a, const std::vector<float>& b,
                            const std::vector<float>& ref_a, const std::vector<float>& ref_b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs((double)a[i] - ref_a[i]));
    for (std::size_t i = 0; i < b.size(); ++i) e = std::max(e, std::abs((double)b[i] - ref_b[i]));
    return e;
}

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

inline bool correct_result(double error, double residual) {
    return std::isfinite(error) && std::isfinite(residual) && error <= 1e-3 && residual <= 1e-3;
}

inline void print_csv(const std::string& impl, int n, int threads, int repeat, double time_ms,
                      double elim_ms, double total_ms, double error, double residual, bool correct) {
    std::cout << impl << ',' << n << ',' << threads << ',' << repeat << ',' << std::fixed
              << std::setprecision(6) << time_ms << ',' << elim_ms << ',' << total_ms << ','
              << std::scientific << error << ',' << residual << ',' << (correct ? "true" : "false") << '\n';
}
