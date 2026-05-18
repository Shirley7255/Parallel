#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <thread>


// 辅助函数


// 内联函数：一维数组模拟二维矩阵的元素访问
// A[i][j] 映射一维偏移量 mat[i * N + j]
inline int IDX(int i, int j, int N) {
    return i * N + j;
}

// 初始化矩阵

void init_matrix(float* mat, int N) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(1.0f, 100.0f);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dist(gen);
    }
}


// ---------------------------------------------------------
// 1. 串行高斯消去法（无主元选取）
// ---------------------------------------------------------
void gauss_serial(float* mat, int N) {
    for (int k = 0; k < N; ++k) {
        // 将第 k 行以下的行进行消去
        for (int i = k + 1; i < N; ++i) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            // 对当前行 k 列之后的元素进行更新计算
            for (int j = k + 1; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            // 当前消去的这个元素直接置0化为上三角
            mat[IDX(i, k, N)] = 0.0f;
        }
    }
}


// ---------------------------------------------------------
// 2. 带有手写插桩的的串行
// ---------------------------------------------------------
void gauss_serial_profile(float* mat, int N) {
    long long time_factor = 0;
    long long time_update = 0;
    
    for (int k = 0; k < N; ++k) {
        for (int i = k + 1; i < N; ++i) {
            // 探针：外层算factor与置零时间
            auto t1 = std::chrono::high_resolution_clock::now();
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            mat[IDX(i, k, N)] = 0.0f;
            auto t2 = std::chrono::high_resolution_clock::now();
            
            // 探针：内层高频读写计算时间
            for (int j = k + 1; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            
            time_factor += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            time_update += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
    }
    std::cout << "[Profiling Report for N=" << N << "]\n";
    std::cout << "外层除法与赋值耗时: " << time_factor / 1e6 << " ms\n";
    std::cout << "内层矩阵消去更新耗时: " << time_update / 1e6 << " ms\n";
}


// ---------------------------------------------------------
// 3. AVX2 单线程高斯消去
// 使用 SIMD 寄存器（__m256）单指令多数据流，一周期并行8 个 float
// ---------------------------------------------------------
void gauss_avx2(float* mat, int N) {
    for (int k = 0; k < N; ++k) {
        for (int i = k + 1; i < N; ++i) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            __m256 vec_factor = _mm256_set1_ps(factor); // 广播乘子
            
            int j = k + 1;
            //每次处理 8 个浮点数
            for (; j + 7 < N; j += 8) {
                __m256 vec_mat_kj = _mm256_loadu_ps(&mat[IDX(k, j, N)]);
                __m256 vec_mat_ij = _mm256_loadu_ps(&mat[IDX(i, j, N)]);
                
                __m256 vec_mul = _mm256_mul_ps(vec_factor, vec_mat_kj);
                __m256 vec_res = _mm256_sub_ps(vec_mat_ij, vec_mul);
                
                _mm256_storeu_ps(&mat[IDX(i, j, N)], vec_res);
            }
            
            // 剩下的不能被 8 整除的标量数据做尾部回退处理
            for (; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            mat[IDX(i, k, N)] = 0.0f;
        }
    }
}


// ---------------------------------------------------------
// 4. OpenMP 多线程高斯消去
// ---------------------------------------------------------
void gauss_openmp(float* mat, int N, int num_threads) {
    for (int k = 0; k < N; ++k) {
    
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = k + 1; i < N; ++i) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            for (int j = k + 1; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            mat[IDX(i, k, N)] = 0.0f;
        }
    }
}



// 辅助：Pthreads 工具宏
pthread_barrier_t barrier;

struct PthreadData {
    float* mat;
    int N;
    int num_threads;
    int thread_id;
};


// ---------------------------------------------------------
// 5. Pthread高斯消去
// ---------------------------------------------------------
void* gauss_pthread_func(void* arg) {
    PthreadData* data = (PthreadData*)arg;
    float* mat = data->mat;
    int N = data->N;
    int num_threads = data->num_threads;
    int thread_id = data->thread_id;

    for (int k = 0; k < N; ++k) {
        // 交替划分：线程0算1,5,9行..线程1算2,6,10行
        for (int i = k + 1 + thread_id; i < N; i += num_threads) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            for (int j = k + 1; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            mat[IDX(i, k, N)] = 0.0f;
        }
        // 当轮次 k 完全跑完，所有线程在屏障前会和，等待下一步工作
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void gauss_pthread(float* mat, int N, int num_threads) {
    pthread_barrier_init(&barrier, nullptr, num_threads);
    pthread_t* threads = new pthread_t[num_threads];
    PthreadData* thread_data = new PthreadData[num_threads];
    
    // 一次性挂载好所有计算线程
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].mat = mat;
        thread_data[i].N = N;
        thread_data[i].num_threads = num_threads;
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], nullptr, gauss_pthread_func, (void*)&thread_data[i]);
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    delete[] threads;
    delete[] thread_data;
    pthread_barrier_destroy(&barrier);
}


// ---------------------------------------------------------
// 6. Pthread + AVX2
// ---------------------------------------------------------
void* gauss_pthread_avx2_func(void* arg) {
    PthreadData* data = (PthreadData*)arg;
    float* mat = data->mat;
    int N = data->N;
    int num_threads = data->num_threads;
    int thread_id = data->thread_id;

    for (int k = 0; k < N; ++k) {
        // 第一维并行：将消去行任务均匀分发给各个物理线程
        for (int i = k + 1 + thread_id; i < N; i += num_threads) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            
            // 为第二维并行做准备：将计算好标量乘子广播到 256 位宽的寄存器中
            __m256 vec_factor = _mm256_set1_ps(factor);
            
            int j = k + 1;
            // 第二维并行：在单核内部，每次通过 SIMD 并行 8 个浮点数
            for (; j + 7 < N; j += 8) {
                // 1. 加载：从内存中直接拉取 8 个主元行元素和 8 个目标行元素到向量寄存器
                __m256 vec_mat_kj = _mm256_loadu_ps(&mat[IDX(k, j, N)]);
                __m256 vec_mat_ij = _mm256_loadu_ps(&mat[IDX(i, j, N)]);
                
                // 2. 计算：在 ALU 内单周期完成 8 路乘法和 8 路减法
                __m256 vec_mul = _mm256_mul_ps(vec_factor, vec_mat_kj);
                __m256 vec_res = _mm256_sub_ps(vec_mat_ij, vec_mul);
                
                // 3. 写回：将这 8 个计算完毕的元素写回内存
                _mm256_storeu_ps(&mat[IDX(i, j, N)], vec_res);
            }
            
            // 尾部标量处理：如果矩阵维数导致尾部不足 8 个元素，降级为普通标量串行处理
            for (; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            // 当前行的主元列下方元素硬性置零，化为上三角
            mat[IDX(i, k, N)] = 0.0f;
        }
        // 等待所有物理核都完成了这轮 k 下属于自己的所有的 i 行消去
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void gauss_pthread_avx2(float* mat, int N, int num_threads) {
    pthread_barrier_init(&barrier, nullptr, num_threads);
    pthread_t* threads = new pthread_t[num_threads];
    PthreadData* thread_data = new PthreadData[num_threads];
    
    // 初始化并启动线程
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].mat = mat;
        thread_data[i].N = N;
        thread_data[i].num_threads = num_threads;
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], nullptr, gauss_pthread_avx2_func, (void*)&thread_data[i]);
    }
    
    // 回收线程
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    delete[] threads;
    delete[] thread_data;
    pthread_barrier_destroy(&barrier);
}


// ---------------------------------------------------------
// 7. Pthread + AVX2 + Loop Unrolling 
// ---------------------------------------------------------
void* gauss_pthread_avx2_unroll_func(void* arg) {
    PthreadData* data = (PthreadData*)arg;
    float* mat = data->mat;
    int N = data->N;
    int num_threads = data->num_threads;
    int thread_id = data->thread_id;

    for (int k = 0; k < N; ++k) {
        for (int i = k + 1 + thread_id; i < N; i += num_threads) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            __m256 vec_factor = _mm256_set1_ps(factor);
            
            int j = k + 1;
            // 循环展开：每次处理 32 个元素 (4 x 8)
            for (; j + 31 < N; j += 32) {
                __m256 vec_mat_kj_0 = _mm256_loadu_ps(&mat[IDX(k, j, N)]);
                __m256 vec_mat_kj_1 = _mm256_loadu_ps(&mat[IDX(k, j + 8, N)]);
                __m256 vec_mat_kj_2 = _mm256_loadu_ps(&mat[IDX(k, j + 16, N)]);
                __m256 vec_mat_kj_3 = _mm256_loadu_ps(&mat[IDX(k, j + 24, N)]);

                __m256 vec_mat_ij_0 = _mm256_loadu_ps(&mat[IDX(i, j, N)]);
                __m256 vec_mat_ij_1 = _mm256_loadu_ps(&mat[IDX(i, j + 8, N)]);
                __m256 vec_mat_ij_2 = _mm256_loadu_ps(&mat[IDX(i, j + 16, N)]);
                __m256 vec_mat_ij_3 = _mm256_loadu_ps(&mat[IDX(i, j + 24, N)]);

                vec_mat_ij_0 = _mm256_sub_ps(vec_mat_ij_0, _mm256_mul_ps(vec_factor, vec_mat_kj_0));
                vec_mat_ij_1 = _mm256_sub_ps(vec_mat_ij_1, _mm256_mul_ps(vec_factor, vec_mat_kj_1));
                vec_mat_ij_2 = _mm256_sub_ps(vec_mat_ij_2, _mm256_mul_ps(vec_factor, vec_mat_kj_2));
                vec_mat_ij_3 = _mm256_sub_ps(vec_mat_ij_3, _mm256_mul_ps(vec_factor, vec_mat_kj_3));

                _mm256_storeu_ps(&mat[IDX(i, j, N)], vec_mat_ij_0);
                _mm256_storeu_ps(&mat[IDX(i, j + 8, N)], vec_mat_ij_1);
                _mm256_storeu_ps(&mat[IDX(i, j + 16, N)], vec_mat_ij_2);
                _mm256_storeu_ps(&mat[IDX(i, j + 24, N)], vec_mat_ij_3);
            }
            
            // 处理剩下的 8 路对齐的部分
            for (; j + 7 < N; j += 8) {
                __m256 vec_mat_kj = _mm256_loadu_ps(&mat[IDX(k, j, N)]);
                __m256 vec_mat_ij = _mm256_loadu_ps(&mat[IDX(i, j, N)]);
                __m256 vec_res = _mm256_sub_ps(vec_mat_ij, _mm256_mul_ps(vec_factor, vec_mat_kj));
                _mm256_storeu_ps(&mat[IDX(i, j, N)], vec_res);
            }
            
            // 尾部标量处理
            for (; j < N; ++j) {
                mat[IDX(i, j, N)] -= factor * mat[IDX(k, j, N)];
            }
            mat[IDX(i, k, N)] = 0.0f;
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void gauss_pthread_avx2_unroll(float* mat, int N, int num_threads) {
    pthread_barrier_init(&barrier, nullptr, num_threads);
    pthread_t* threads = new pthread_t[num_threads];
    PthreadData* thread_data = new PthreadData[num_threads];
    
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].mat = mat;
        thread_data[i].N = N;
        thread_data[i].num_threads = num_threads;
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], nullptr, gauss_pthread_avx2_unroll_func, (void*)&thread_data[i]);
    }
    
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    delete[] threads;
    delete[] thread_data;
    pthread_barrier_destroy(&barrier);
}



// C++ 标准库多线程版本
void gauss_std_thread(float* mat, int N, int num_threads) {
    for (int k = 0; k < N; ++k) {
        // 1. 串行除法步骤（主线程完成）
        for (int j = k + 1; j < N; ++j) {
            mat[k * N + j] = mat[k * N + j] / mat[k * N + k];
        }
        mat[k * N + k] = 1.0f;

        // 2. 并行消去步骤
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            // 使用 Lambda 表达式创建线程
            threads.emplace_back([=, &mat]() {
                // 使用交替（Cyclic）划分
                for (int i = k + 1 + t; i < N; i += num_threads) {
                    for (int j = k + 1; j < N; ++j) {
                        mat[i * N + j] -= mat[i * N + k] * mat[k * N + j];
                    }
                    mat[i * N + k] = 0.0f;
                }
            });
        }
        
        // 3. 等待所有线程完成 (相当于隐式的 Barrier)
        for (auto& th : threads) {
            th.join();
        }
    }
}
// OpenMP 卸载到 GPU 加速器版本
void gauss_omp_target(float* mat, int N) {
    // 将整个矩阵在 k 循环开始前，一次性拷贝到 GPU 显存中，计算结束后再拷贝回内存
    #pragma omp target data map(tofrom: mat[0:N*N])
    {
        for (int k = 0; k < N; ++k) {
            // 将内层的消去任务（i 和 j 循环）打包成 Kernel 发射给 GPU
            // 将任务分配给 GPU 的不同流多处理器 (SM / 线程块)
            // 在每个 SM 内部使用 GPU 线程束 (Warp) 并行计算
            #pragma omp target teams distribute parallel for
            for (int i = k + 1; i < N; ++i) {
                // 以下代码全部在 GPU 内部的高速显存和流处理器中执行
                float factor = mat[i * N + k] / mat[k * N + k];
                for (int j = k + 1; j < N; ++j) {
                    mat[i * N + j] -= factor * mat[k * N + j];
                }
                mat[i * N + k] = 0.0f;
            }
        }
    }
}
int main() {
    
    std::vector<int> sizes = {256, 512, 1024, 2048};
    std::vector<int> thread_counts = {2, 4, 8};
    
    std::cout << "Algorithm,N,Threads,Time(ms)" << std::endl;
    
    for (int N : sizes) {
        float* mat = new float[N * N];
        
        // ---------------------------------------------------------
        // 1. 单线程基准
        // ---------------------------------------------------------
        init_matrix(mat, N);
        auto t_start = std::chrono::high_resolution_clock::now();
        gauss_serial(mat, N);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d_serial = t_end - t_start;
        std::cout << "Serial," << N << ",1," << d_serial.count() << std::endl;
        
        // 当 N=2048 等超大规模可以跑一次插桩
        if (N == 2048) {
            init_matrix(mat, N);
            gauss_serial_profile(mat, N);
        }

        // ---------------------------------------------------------
        // 2. AVX2 单核
        // ---------------------------------------------------------
        init_matrix(mat, N);
        t_start = std::chrono::high_resolution_clock::now();
        gauss_avx2(mat, N);
        t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d_avx = t_end - t_start;
        std::cout << "AVX2," << N << ",1," << d_avx.count() << std::endl;
        
        
        // ---------------------------------------------------------
        // 3. 多核
        // ---------------------------------------------------------
        if (N >= 512) {
            for (int t : thread_counts) {
                // OpenMP 测试
                init_matrix(mat, N);
                auto t_start_omp = std::chrono::high_resolution_clock::now();
                gauss_openmp(mat, N, t);
                auto t_end_omp = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> d_omp = t_end_omp - t_start_omp;
                std::cout << "OpenMP," << N << "," << t << "," << d_omp.count() << std::endl;
                
                // Pthread 测试
                init_matrix(mat, N);
                auto t_start_pth = std::chrono::high_resolution_clock::now();
                gauss_pthread(mat, N, t);
                auto t_end_pth = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> d_pth = t_end_pth - t_start_pth;
                std::cout << "Pthread," << N << "," << t << "," << d_pth.count() << std::endl;
                
                // C++ std::thread 测试
                init_matrix(mat, N);
                auto t_start_std = std::chrono::high_resolution_clock::now();
                gauss_std_thread(mat, N, t);
                auto t_end_std = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> d_std = t_end_std - t_start_std;
                std::cout << "std::thread," << N << "," << t << "," << d_std.count() << std::endl;
            }
        }
        
        // ---------------------------------------------------------
        // 4. Pthread + AVX2 
        // ---------------------------------------------------------
        if (N == 2048) {
            for (int t : thread_counts) {
                init_matrix(mat, N);
                auto t_start_final = std::chrono::high_resolution_clock::now();
                gauss_pthread_avx2(mat, N, t);
                auto t_end_final = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> d_final = t_end_final - t_start_final;
                std::cout << "Pthread+AVX2," << N << "," << t << "," << d_final.count() << std::endl;
            }
        }
        
        // ---------------------------------------------------------
        // 5. Pthread + AVX2 + Loop Unroll 
        // ---------------------------------------------------------
        if (N == 2048) {
            for (int t : thread_counts) {
                init_matrix(mat, N);
                auto t_start_unroll = std::chrono::high_resolution_clock::now();
                gauss_pthread_avx2_unroll(mat, N, t);
                auto t_end_unroll = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> d_unroll = t_end_unroll - t_start_unroll;
                std::cout << "Pthread+AVX2_Unroll," << N << "," << t << "," << d_unroll.count() << std::endl;
            }
        }
        
        delete[] mat; // 清理缓存
    }
    
    return 0;
}