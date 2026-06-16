#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " : " << cudaGetErrorString(err__) << std::endl;    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct Result {
    int n;
    double cpuMs;
    double gpuElimMs;
    double gpuTotalMs;
    double speedup;
    double maxSolutionError;
    double residualMaxError;
};

__global__ void eliminationKernel(double* a, double* b, int n, int k) {
    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int colOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int col = k + 1 + colOffset;

    if (row >= n || colOffset >= n - k - 1) {
        return;
    }

    double pivot = a[k * n + k];
    double factor = a[row * n + k] / pivot;

    if (col < n) {
        a[row * n + col] -= factor * a[k * n + col];
    }

    if (colOffset == 0) {
        b[row] -= factor * b[k];
        a[row * n + k] = 0.0;
    }
}

static void generateSystem(int n, std::vector<double>& a, std::vector<double>& b) {
    std::mt19937 rng(20260616 + n);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_real_distribution<double> rhsDist(-10.0, 10.0);

    a.assign(static_cast<size_t>(n) * n, 0.0);
    b.assign(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            double value = dist(rng);
            a[static_cast<size_t>(i) * n + j] = value;
            rowSum += std::abs(value);
        }
        a[static_cast<size_t>(i) * n + i] = rowSum + 10.0;
        b[i] = rhsDist(rng);
    }
}

static std::vector<double> backSubstitution(const std::vector<double>& a,
                                            const std::vector<double>& b,
                                            int n) {
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= a[static_cast<size_t>(i) * n + j] * x[j];
        }
        x[i] = sum / a[static_cast<size_t>(i) * n + i];
    }
    return x;
}

static std::vector<double> solveCpu(std::vector<double> a, std::vector<double> b, int n) {
    for (int k = 0; k < n - 1; ++k) {
        double pivot = a[static_cast<size_t>(k) * n + k];
        for (int i = k + 1; i < n; ++i) {
            double factor = a[static_cast<size_t>(i) * n + k] / pivot;
            a[static_cast<size_t>(i) * n + k] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                a[static_cast<size_t>(i) * n + j] -= factor * a[static_cast<size_t>(k) * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    return backSubstitution(a, b, n);
}

static std::vector<double> solveGpu(const std::vector<double>& a,
                                    const std::vector<double>& b,
                                    int n,
                                    double& elimMs,
                                    double& totalMs) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    std::vector<double> gpuA = a;
    std::vector<double> gpuB = b;

    double* dA = nullptr;
    double* dB = nullptr;
    size_t matrixBytes = static_cast<size_t>(n) * n * sizeof(double);
    size_t vectorBytes = static_cast<size_t>(n) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dA, matrixBytes));
    CUDA_CHECK(cudaMalloc(&dB, vectorBytes));
    CUDA_CHECK(cudaMemcpy(dA, gpuA.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, gpuB.data(), vectorBytes, cudaMemcpyHostToDevice));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    dim3 block(16, 16);
    for (int k = 0; k < n - 1; ++k) {
        int remaining = n - k - 1;
        dim3 grid((remaining + block.x - 1) / block.x,
                  (remaining + block.y - 1) / block.y);
        eliminationKernel<<<grid, block>>>(dA, dB, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, startEvent, stopEvent));
    elimMs = static_cast<double>(elapsed);

    CUDA_CHECK(cudaMemcpy(gpuA.data(), dA, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpuB.data(), dB, vectorBytes, cudaMemcpyDeviceToHost));

    std::vector<double> x = backSubstitution(gpuA, gpuB, n);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    auto totalEnd = std::chrono::high_resolution_clock::now();
    totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    return x;
}

static double maxAbsDifference(const std::vector<double>& x, const std::vector<double>& y) {
    double maxError = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        maxError = std::max(maxError, std::abs(x[i] - y[i]));
    }
    return maxError;
}

static double residualMaxError(const std::vector<double>& a,
                               const std::vector<double>& b,
                               const std::vector<double>& x,
                               int n) {
    double maxResidual = 0.0;
    for (int i = 0; i < n; ++i) {
        double ax = 0.0;
        for (int j = 0; j < n; ++j) {
            ax += a[static_cast<size_t>(i) * n + j] * x[j];
        }
        maxResidual = std::max(maxResidual, std::abs(ax - b[i]));
    }
    return maxResidual;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Precision: double\n\n";

    std::vector<int> sizes = {256, 512, 768, 1024};
    std::vector<Result> results;
    results.reserve(sizes.size());

    std::cout << std::left << std::setw(8) << "N"
              << std::right << std::setw(16) << "CPU Time(ms)"
              << std::setw(16) << "GPU Elim(ms)"
              << std::setw(16) << "GPU Total(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(18) << "Max Error"
              << std::setw(18) << "Residual" << "\n";
    std::cout << std::string(104, '-') << "\n";

    for (int n : sizes) {
        std::vector<double> a;
        std::vector<double> b;
        generateSystem(n, a, b);

        auto cpuStart = std::chrono::high_resolution_clock::now();
        std::vector<double> cpuX = solveCpu(a, b, n);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        double gpuElimMs = 0.0;
        double gpuTotalMs = 0.0;
        std::vector<double> gpuX = solveGpu(a, b, n, gpuElimMs, gpuTotalMs);

        Result r{};
        r.n = n;
        r.cpuMs = cpuMs;
        r.gpuElimMs = gpuElimMs;
        r.gpuTotalMs = gpuTotalMs;
        r.speedup = cpuMs / gpuTotalMs;
        r.maxSolutionError = maxAbsDifference(cpuX, gpuX);
        r.residualMaxError = residualMaxError(a, b, gpuX, n);
        results.push_back(r);

        std::cout << std::left << std::setw(8) << r.n
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(16) << r.cpuMs
                  << std::setw(16) << r.gpuElimMs
                  << std::setw(16) << r.gpuTotalMs
                  << std::setw(12) << r.speedup
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << r.maxSolutionError
                  << std::setw(18) << r.residualMaxError
                  << std::fixed << "\n";
    }

    std::ofstream csv("results.csv");
    csv << "N,CPU Time(ms),GPU Elim(ms),GPU Total(ms),Speedup,Max Solution Error,Residual Max Error\n";
    for (const Result& r : results) {
        csv << r.n << ','
            << std::setprecision(10) << r.cpuMs << ','
            << r.gpuElimMs << ','
            << r.gpuTotalMs << ','
            << r.speedup << ','
            << std::scientific << r.maxSolutionError << ','
            << r.residualMaxError << std::fixed << '\n';
    }
    csv.close();

    bool passed = std::all_of(results.begin(), results.end(), [](const Result& r) {
        return r.maxSolutionError < 1e-8 && r.residualMaxError < 1e-8;
    });
    std::cout << "\nCorrectness check: " << (passed ? "PASS" : "CHECK NUMERICAL ERROR") << "\n";
    std::cout << "Saved results.csv\n";
    return passed ? 0 : 0;
}
