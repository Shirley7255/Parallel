#include "common.hpp"
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)

__global__ void element_kernel(float* a, float* b, int n, int k) {
    const int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const float f = a[IDX(row,k,n)] / a[IDX(k,k,n)];
        a[IDX(row,col,n)] -= f * a[IDX(k,col,n)];
        if (col == k + 1) b[row] -= f * b[k];
    }
}

__global__ void row_kernel(float* a, float* b, int n, int k) {
    const int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const float f = a[IDX(row,k,n)] / a[IDX(k,k,n)];
        a[IDX(row,k,n)] = 0.0f;
        for (int j = k + 1; j < n; ++j) a[IDX(row,j,n)] -= f * a[IDX(k,j,n)];
        b[row] -= f * b[k];
    }
}

__global__ void zero_lower_kernel(float* a, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < row) a[IDX(row,col,n)] = 0.0f;
}

__global__ void warmup_kernel(float* x) {
    if (blockIdx.x == 0 && threadIdx.x == 0) x[0] = 1.0f;
}

enum class Kind { Element, Row };

static void launch(float* da, float* db, int n, Kind kind) {
    for (int k = 0; k < n - 1; ++k) {
        if (kind == Kind::Element) {
            const dim3 block(32, 8);
            const dim3 grid((n-k-1+block.x-1)/block.x, (n-k-1+block.y-1)/block.y);
            element_kernel<<<grid,block>>>(da,db,n,k);
        } else {
            const int block = 256;
            row_kernel<<<(n-k-1+block-1)/block,block>>>(da,db,n,k);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (kind == Kind::Element) {
        const dim3 block(32,8), grid((n+31)/32,(n+7)/8);
        zero_lower_kernel<<<grid,block>>>(da,n);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    }
}

static void benchmark(const char* name, Kind kind, const Problem& p, const Options& o,
                      const std::vector<float>& ref_a, const std::vector<float>& ref_b,
                      float* da, float* db) {
    const std::size_t ab = p.a.size()*sizeof(float), bb = p.b.size()*sizeof(float);
    std::vector<double> elim, total;
    std::vector<float> a(p.a.size()), b(p.b.size());
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    for (int r=0; r<o.repeat; ++r) {
        const auto wall0=std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(da,p.a.data(),ab,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db,p.b.data(),bb,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start)); launch(da,db,p.n,kind); CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop)); float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        CUDA_CHECK(cudaMemcpy(a.data(),da,ab,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(),db,bb,cudaMemcpyDeviceToHost));
        elim.push_back(ms);
        total.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall0).count());
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    const double e=max_abs_error(a,b,ref_a,ref_b), rr=relative_residual(p,a,b);
    const double em=median(elim), tm=median(total);
    print_csv(name,p.n,0,o.repeat,tm,em,tm,e,rr,correct_result(e,rr));
}

static void cuda_warmup_once() {
    CUDA_CHECK(cudaFree(0));
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
    warmup_kernel<<<1, 32>>>(d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d));
}

static void benchmark_warmup_compare(const Problem& p, const Options& o,
                                     const std::vector<float>& ref_a,
                                     const std::vector<float>& ref_b) {
    if (o.warmup_mode != "cold" && o.warmup_mode != "warm") {
        throw std::runtime_error("--warmup-mode must be cold or warm");
    }

    if (o.warmup_mode == "warm") cuda_warmup_once();

    float *da = nullptr, *db = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::vector<float> a(p.a.size()), b(p.b.size());
    const std::size_t ab = p.a.size() * sizeof(float), bb = p.b.size() * sizeof(float);

    const auto wall0 = std::chrono::steady_clock::now();
    try {
        CUDA_CHECK(cudaMalloc(&da, ab));
        CUDA_CHECK(cudaMalloc(&db, bb));
        CUDA_CHECK(cudaMemcpy(da, p.a.data(), ab, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, p.b.data(), bb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        launch(da, db, p.n, Kind::Element);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elim_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elim_ms, start, stop));
        CUDA_CHECK(cudaMemcpy(a.data(), da, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), db, bb, cudaMemcpyDeviceToHost));
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall0).count();
        const double e = max_abs_error(a, b, ref_a, ref_b);
        const double rr = relative_residual(p, a, b);
        std::cout << p.n << ',' << o.warmup_mode << ',' << o.run_index << ','
                  << std::fixed << std::setprecision(6) << elim_ms << ',' << total_ms << ','
                  << (correct_result(e, rr) ? "true" : "false") << '\n';
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (da) cudaFree(da);
        if (db) cudaFree(db);
        throw;
    }
    if (start) CUDA_CHECK(cudaEventDestroy(start));
    if (stop) CUDA_CHECK(cudaEventDestroy(stop));
    if (da) CUDA_CHECK(cudaFree(da));
    if (db) CUDA_CHECK(cudaFree(db));
}

int main(int argc,char** argv) {
    float *da=nullptr,*db=nullptr;
    try {
        const Options o=parse_options(argc,argv);
        if (!o.warmup_mode.empty() && o.impl != "all" && o.impl != "cuda_2d_element") {
            throw std::runtime_error("warm-up comparison uses cuda_2d_element; omit --impl or use --impl cuda_2d_element");
        }
        const Problem p=make_problem(o.n); auto ra=p.a; auto rb=p.b; gauss_serial(ra,rb,o.n);
        if (!o.warmup_mode.empty()) {
            benchmark_warmup_compare(p, o, ra, rb);
            return 0;
        }
        CUDA_CHECK(cudaFree(0));
        CUDA_CHECK(cudaMalloc(&da,p.a.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&db,p.b.size()*sizeof(float)));
        if(o.impl=="all" || o.impl=="cuda_2d_element") benchmark("cuda_2d_element",Kind::Element,p,o,ra,rb,da,db);
        if(o.impl=="all" || o.impl=="cuda_row_kernel") benchmark("cuda_row_kernel",Kind::Row,p,o,ra,rb,da,db);
        if(o.impl!="all" && o.impl!="cuda_2d_element" && o.impl!="cuda_row_kernel") throw std::runtime_error("unknown CUDA implementation: "+o.impl);
        CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); return 0;
    } catch(const std::exception& e) { if(da) cudaFree(da); if(db) cudaFree(db); std::cerr<<"gauss_cuda: "<<e.what()<<'\n'; return 1; }
}
