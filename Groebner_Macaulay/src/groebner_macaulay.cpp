#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int MOD = 65521;

struct Options {
    int n_vars = 4;
    int degree = 4;
    int threads = 1;
    int repeat = 3;
    std::string impl = "serial";
};

struct Monomial {
    std::vector<int> exp;
    int degree = 0;
};

struct Term {
    std::vector<int> exp;
    int coeff = 0;
};

struct Polynomial {
    std::vector<Term> terms;
    int degree = 2;
};

struct Macaulay {
    int rows = 0;
    int cols = 0;
    std::vector<int> a;
    std::uint64_t nnz_before = 0;
};

struct ElimResult {
    int rank = 0;
    std::uint64_t nnz_after = 0;
    std::uint64_t checksum = 0;
};

int mod_norm(long long x) {
    x %= MOD;
    if (x < 0) x += MOD;
    return static_cast<int>(x);
}

int mod_pow(int a, int e) {
    long long base = mod_norm(a);
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = (r * base) % MOD;
        base = (base * base) % MOD;
        e >>= 1;
    }
    return static_cast<int>(r);
}

int mod_inv(int a) {
    if (a == 0) throw std::runtime_error("zero has no modular inverse");
    return mod_pow(a, MOD - 2);
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need_int = [&](const std::string& name) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return std::stoi(argv[++i]);
        };
        if (s == "--n-vars") o.n_vars = need_int(s);
        else if (s == "--D") o.degree = need_int(s);
        else if (s == "--threads") o.threads = need_int(s);
        else if (s == "--repeat") o.repeat = need_int(s);
        else if (s == "--impl") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --impl");
            o.impl = argv[++i];
        } else {
            throw std::runtime_error("unknown argument: " + s);
        }
    }
    if (o.n_vars < 1 || o.degree < 2 || o.threads < 1 || o.repeat < 1) {
        throw std::runtime_error("invalid positive argument");
    }
    if (o.impl != "serial" && o.impl != "openmp") {
        throw std::runtime_error("--impl must be serial or openmp");
    }
    return o;
}

void gen_degree_rec(int n, int pos, int remaining, std::vector<int>& cur,
                    std::vector<Monomial>& out) {
    if (pos == n - 1) {
        cur[pos] = remaining;
        out.push_back({cur, std::accumulate(cur.begin(), cur.end(), 0)});
        return;
    }
    for (int v = remaining; v >= 0; --v) {
        cur[pos] = v;
        gen_degree_rec(n, pos + 1, remaining - v, cur, out);
    }
}

std::vector<Monomial> generate_monomials(int n_vars, int max_degree) {
    std::vector<Monomial> out;
    std::vector<int> cur(n_vars, 0);
    for (int d = max_degree; d >= 0; --d) gen_degree_rec(n_vars, 0, d, cur, out);
    return out;
}

std::vector<Polynomial> generate_polynomials(int n_vars) {
    std::mt19937 rng(20260630u + static_cast<unsigned>(n_vars) * 131u);
    std::uniform_int_distribution<int> coeff_dist(1, MOD - 1);

    std::vector<Monomial> deg2 = generate_monomials(n_vars, 2);
    deg2.erase(std::remove_if(deg2.begin(), deg2.end(),
                              [](const Monomial& m) { return m.degree != 2; }),
               deg2.end());
    std::shuffle(deg2.begin(), deg2.end(), rng);

    std::vector<Polynomial> polys;
    polys.reserve(n_vars);
    for (int p = 0; p < n_vars; ++p) {
        Polynomial poly;
        std::map<std::vector<int>, int> acc;

        const int q_terms = std::min<int>(static_cast<int>(deg2.size()), std::max(4, n_vars + 1));
        for (int t = 0; t < q_terms; ++t) {
            const auto& e = deg2[(p * 3 + t) % deg2.size()].exp;
            acc[e] = mod_norm(acc[e] + coeff_dist(rng));
        }

        for (int v = 0; v < n_vars; ++v) {
            std::vector<int> e(n_vars, 0);
            e[v] = 1;
            acc[e] = mod_norm(acc[e] + coeff_dist(rng));
        }

        std::vector<int> c(n_vars, 0);
        acc[c] = mod_norm(acc[c] + coeff_dist(rng));

        for (const auto& kv : acc) {
            if (kv.second != 0) poly.terms.push_back({kv.first, kv.second});
        }
        polys.push_back(std::move(poly));
    }
    return polys;
}

Macaulay build_macaulay(int n_vars, int D) {
    const auto columns = generate_monomials(n_vars, D);
    const auto multipliers = generate_monomials(n_vars, D - 2);
    const auto polys = generate_polynomials(n_vars);

    std::map<std::vector<int>, int> col_index;
    for (int i = 0; i < static_cast<int>(columns.size()); ++i) col_index[columns[i].exp] = i;

    Macaulay m;
    m.cols = static_cast<int>(columns.size());
    m.rows = static_cast<int>(multipliers.size() * polys.size());
    m.a.assign(static_cast<std::size_t>(m.rows) * m.cols, 0);

    int row = 0;
    for (const auto& poly : polys) {
        for (const auto& mult : multipliers) {
            for (const auto& term : poly.terms) {
                std::vector<int> e(n_vars);
                int deg = 0;
                for (int v = 0; v < n_vars; ++v) {
                    e[v] = mult.exp[v] + term.exp[v];
                    deg += e[v];
                }
                if (deg > D) continue;
                auto it = col_index.find(e);
                if (it == col_index.end()) continue;
                int& cell = m.a[static_cast<std::size_t>(row) * m.cols + it->second];
                const int old = cell;
                cell = mod_norm(cell + term.coeff);
                if (old == 0 && cell != 0) ++m.nnz_before;
                else if (old != 0 && cell == 0) --m.nnz_before;
            }
            ++row;
        }
    }
    return m;
}

std::uint64_t count_nnz(const std::vector<int>& a) {
    std::uint64_t n = 0;
    for (int v : a) n += (v != 0);
    return n;
}

std::uint64_t checksum_matrix(const std::vector<int>& a) {
    std::uint64_t h = 1469598103934665603ull;
    for (int v : a) {
        h ^= static_cast<std::uint64_t>(v + 1);
        h *= 1099511628211ull;
    }
    return h;
}

ElimResult mod_gauss(std::vector<int>& a, int rows, int cols, int threads, bool parallel) {
#ifdef _OPENMP
    if (parallel) omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    int rank = 0;
    for (int col = 0; col < cols && rank < rows; ++col) {
        int pivot = -1;
        for (int r = rank; r < rows; ++r) {
            if (a[static_cast<std::size_t>(r) * cols + col] != 0) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) continue;

        if (pivot != rank) {
            for (int j = col; j < cols; ++j) {
                std::swap(a[static_cast<std::size_t>(pivot) * cols + j],
                          a[static_cast<std::size_t>(rank) * cols + j]);
            }
        }

        int* prow = a.data() + static_cast<std::size_t>(rank) * cols;
        const int inv = mod_inv(prow[col]);
        std::vector<int> pivot_nz;
        pivot_nz.reserve(cols - col);
        for (int j = col; j < cols; ++j) {
            if (prow[j] != 0) {
                prow[j] = static_cast<int>((static_cast<long long>(prow[j]) * inv) % MOD);
                if (prow[j] != 0) pivot_nz.push_back(j);
            }
        }

        if (parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = rank + 1; i < rows; ++i) {
                int* row = a.data() + static_cast<std::size_t>(i) * cols;
                const int factor = row[col];
                if (factor == 0) continue;
                for (int j : pivot_nz) {
                    row[j] = mod_norm(static_cast<long long>(row[j]) -
                                      static_cast<long long>(factor) * prow[j]);
                }
            }
        } else {
            for (int i = rank + 1; i < rows; ++i) {
                int* row = a.data() + static_cast<std::size_t>(i) * cols;
                const int factor = row[col];
                if (factor == 0) continue;
                for (int j : pivot_nz) {
                    row[j] = mod_norm(static_cast<long long>(row[j]) -
                                      static_cast<long long>(factor) * prow[j]);
                }
            }
        }
        ++rank;
    }
    return {rank, count_nnz(a), checksum_matrix(a)};
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);
        const Macaulay base = build_macaulay(opt.n_vars, opt.degree);
        const long double total = static_cast<long double>(base.rows) * base.cols;
        const double density_before = total > 0 ? static_cast<double>(base.nnz_before / total) : 0.0;

        std::vector<int> ref = base.a;
        const ElimResult ref_result = mod_gauss(ref, base.rows, base.cols, 1, false);

        std::vector<double> times;
        ElimResult last{};
        for (int r = 0; r < opt.repeat; ++r) {
            std::vector<int> work = base.a;
            const auto t0 = std::chrono::steady_clock::now();
            last = mod_gauss(work, base.rows, base.cols, opt.threads, opt.impl == "openmp");
            const auto t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        const double time_ms = median(times);
        const double density_after = total > 0 ? static_cast<double>(last.nnz_after / total) : 0.0;
        const double fillin_ratio = base.nnz_before ? static_cast<double>(last.nnz_after) / base.nnz_before : 0.0;
        const bool correct = (last.rank == ref_result.rank && last.checksum == ref_result.checksum);

        std::cout << opt.n_vars << ',' << opt.degree << ',' << base.rows << ',' << base.cols << ','
                  << opt.threads << ',' << opt.repeat << ',' << opt.impl << ','
                  << std::fixed << std::setprecision(6) << time_ms << ','
                  << last.rank << ',' << base.nnz_before << ','
                  << std::setprecision(9) << density_before << ','
                  << last.nnz_after << ',' << density_after << ','
                  << fillin_ratio << ',' << last.checksum << ','
                  << (correct ? "true" : "false") << '\n';
        return correct ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "groebner_macaulay: " << e.what() << '\n';
        return 1;
    }
}
