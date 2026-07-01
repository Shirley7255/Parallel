#include "common.hpp"
#include <immintrin.h>
#include <omp.h>

enum class Schedule { Static, Dynamic, Guided };

static void gauss_avx2(std::vector<float>& a, std::vector<float>& b, int n) {
    for (int k = 0; k < n - 1; ++k) {
        for (int i = k + 1; i < n; ++i) {
            const float f = a[IDX(i, k, n)] / a[IDX(k, k, n)];
            a[IDX(i, k, n)] = 0.0f;
            const __m256 vf = _mm256_set1_ps(f);
            int j = k + 1;
            for (; j + 7 < n; j += 8) {
                __m256 vi = _mm256_loadu_ps(&a[IDX(i, j, n)]);
                const __m256 vk = _mm256_loadu_ps(&a[IDX(k, j, n)]);
                vi = _mm256_sub_ps(vi, _mm256_mul_ps(vf, vk));
                _mm256_storeu_ps(&a[IDX(i, j, n)], vi);
            }
            for (; j < n; ++j) a[IDX(i, j, n)] -= f * a[IDX(k, j, n)];
            b[i] -= f * b[k];
        }
    }
}

static inline void update_row(std::vector<float>& a, std::vector<float>& b, int n, int k, int i, bool avx) {
            const float f = a[IDX(i, k, n)] / a[IDX(k, k, n)];
            a[IDX(i, k, n)] = 0.0f;
            int j = k + 1;
            if (avx) {
                const __m256 vf = _mm256_set1_ps(f);
                for (; j + 7 < n; j += 8) {
                    __m256 vi = _mm256_loadu_ps(&a[IDX(i, j, n)]);
                    const __m256 vk = _mm256_loadu_ps(&a[IDX(k, j, n)]);
                    _mm256_storeu_ps(&a[IDX(i, j, n)], _mm256_sub_ps(vi, _mm256_mul_ps(vf, vk)));
                }
            }
            for (; j < n; ++j) a[IDX(i, j, n)] -= f * a[IDX(k, j, n)];
            b[i] -= f * b[k];
}

static void gauss_omp(std::vector<float>& a, std::vector<float>& b, int n, Schedule schedule, bool avx) {
    for (int k = 0; k < n - 1; ++k) {
        if (schedule == Schedule::Static) {
#pragma omp parallel for schedule(static)
            for (int i = k + 1; i < n; ++i) update_row(a,b,n,k,i,avx);
        } else if (schedule == Schedule::Dynamic) {
#pragma omp parallel for schedule(dynamic,8)
            for (int i = k + 1; i < n; ++i) update_row(a,b,n,k,i,avx);
        } else {
#pragma omp parallel for schedule(guided,8)
            for (int i = k + 1; i < n; ++i) update_row(a,b,n,k,i,avx);
        }
    }
}

template<class F>
static void benchmark(const char* name, const Problem& p, const std::vector<float>& ref_a,
                      const std::vector<float>& ref_b, const Options& o, F fn) {
    std::vector<double> times;
    std::vector<float> a, b;
    for (int r = 0; r < o.repeat; ++r) {
        a = p.a; b = p.b;
        const double t0 = omp_get_wtime();
        fn(a, b, p.n);
        times.push_back((omp_get_wtime() - t0) * 1000.0);
    }
    const double e = max_abs_error(a, b, ref_a, ref_b);
    const double rr = relative_residual(p, a, b);
    print_csv(name, p.n, o.threads, o.repeat, median(times), 0, 0, e, rr, correct_result(e, rr));
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        omp_set_dynamic(0); omp_set_num_threads(o.threads);
        const Problem p = make_problem(o.n);
        auto ref_a = p.a; auto ref_b = p.b;
        gauss_serial(ref_a, ref_b, o.n);
        auto selected = [&](const char* s) { return o.impl == "all" || o.impl == s; };
        if (selected("serial")) benchmark("serial", p, ref_a, ref_b, o, gauss_serial);
        if (selected("avx2_unaligned")) benchmark("avx2_unaligned", p, ref_a, ref_b, o, gauss_avx2);
        if (selected("openmp_static")) benchmark(o.impl == "all" ? "openmp_static" : "openmp_static_16", p, ref_a, ref_b, o, [](auto& a, auto& b, int n){ gauss_omp(a,b,n,Schedule::Static,false); });
        if (selected("openmp_dynamic")) benchmark("openmp_dynamic", p, ref_a, ref_b, o, [](auto& a, auto& b, int n){ gauss_omp(a,b,n,Schedule::Dynamic,false); });
        if (selected("openmp_guided")) benchmark("openmp_guided", p, ref_a, ref_b, o, [](auto& a, auto& b, int n){ gauss_omp(a,b,n,Schedule::Guided,false); });
        if (selected("openmp_avx2_dynamic")) benchmark(o.impl == "all" ? "openmp_avx2_dynamic" : "openmp_avx2_12", p, ref_a, ref_b, o, [](auto& a, auto& b, int n){ gauss_omp(a,b,n,Schedule::Dynamic,true); });
        if (o.impl != "all" && o.impl != "serial" && o.impl != "avx2_unaligned" && o.impl != "openmp_static" && o.impl != "openmp_dynamic" && o.impl != "openmp_guided" && o.impl != "openmp_avx2_dynamic") throw std::runtime_error("unknown CPU implementation: " + o.impl);
        return 0;
    } catch (const std::exception& e) { std::cerr << "gauss_cpu: " << e.what() << '\n'; return 1; }
}
