#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <stdlib.h>
#include <arm_neon.h>
#include <omp.h>
#include <pthread.h>


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
// 3. NEON 单线程高斯消去
// ---------------------------------------------------------
void gauss_neon(float* mat, int N) {
    for (int k = 0; k < N; ++k) {
        for (int i = k + 1; i < N; ++i) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            float32x4_t vec_factor = vdupq_n_f32(factor); // 改成arm的vdupq_n_f32
            
            int j = k + 1;
            // 每次处理 4 个浮点数
            for (; j + 3 < N; j += 4) {
                float32x4_t vec_mat_kj = vld1q_f32(&mat[IDX(k, j, N)]); // 改成arm的vld1q_f32
                float32x4_t vec_mat_ij = vld1q_f32(&mat[IDX(i, j, N)]); // 改成arm的vld1q_f32
                
                float32x4_t vec_mul = vmulq_f32(vec_factor, vec_mat_kj); // 改成arm的vmulq_f32
                float32x4_t vec_res = vsubq_f32(vec_mat_ij, vec_mul);    // 改成arm的vsubq_f32
                
                vst1q_f32(&mat[IDX(i, j, N)], vec_res); // 改成arm的vst1q_f32
            }
            
            // 剩下的不能被 4 整除的标量数据做尾部回退处理
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
// 6. Pthread + NEON
// ---------------------------------------------------------
void* gauss_pthread_neon_func(void* arg) {
    PthreadData* data = (PthreadData*)arg;
    float* mat = data->mat;
    int N = data->N;
    int num_threads = data->num_threads;
    int thread_id = data->thread_id;

    for (int k = 0; k < N; ++k) {
        // 第一维并行：将消去行任务均匀分发给各个物理线程
        for (int i = k + 1 + thread_id; i < N; i += num_threads) {
            float factor = mat[IDX(i, k, N)] / mat[IDX(k, k, N)];
            
            // 为第二维并行做准备：将计算好标量乘子广播到 128 位宽的寄存器中
            float32x4_t vec_factor = vdupq_n_f32(factor); // 改成arm的vdupq_n_f32
            
            int j = k + 1;
            // 第二维并行：在单核内部，每次通过 SIMD 并行 4 个浮点数
            for (; j + 3 < N; j += 4) {
                // 1. 加载
                float32x4_t vec_mat_kj = vld1q_f32(&mat[IDX(k, j, N)]); // 改成arm的vld1q_f32
                float32x4_t vec_mat_ij = vld1q_f32(&mat[IDX(i, j, N)]); // 改成arm的vld1q_f32
                
                // 2. 计算
                float32x4_t vec_mul = vmulq_f32(vec_factor, vec_mat_kj); // 改成arm的vmulq_f32
                float32x4_t vec_res = vsubq_f32(vec_mat_ij, vec_mul);    // 改成arm的vsubq_f32
                
                // 3. 写回
                vst1q_f32(&mat[IDX(i, j, N)], vec_res); // 改成arm的vst1q_f32
            }
            
            // 尾部标量处理：如果矩阵维数导致尾部不足 4 个元素
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

void gauss_pthread_neon(float* mat, int N, int num_threads) {
    pthread_barrier_init(&barrier, nullptr, num_threads);
    pthread_t* threads = new pthread_t[num_threads];
    PthreadData* thread_data = new PthreadData[num_threads];
    
    // 初始化并启动线程
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].mat = mat;
        thread_data[i].N = N;
        thread_data[i].num_threads = num_threads;
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], nullptr, gauss_pthread_neon_func, (void*)&thread_data[i]);
    }
    
    // 回收线程
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    delete[] threads;
    delete[] thread_data;
    pthread_barrier_destroy(&barrier);
}



int main() {
    
    std::vector<int> sizes = {256, 512, 1024, 2048};
    std::vector<int> thread_counts = {2, 4, 8};
    
    std::cout << "Algorithm,N,Threads,Time(ms)" << std::endl;
    
    for (int N : sizes) {
        float* mat;
        posix_memalign((void**)&mat, 32, N * N * sizeof(float));
        
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
        // 2. NEON 单核
        // ---------------------------------------------------------
        init_matrix(mat, N);
        t_start = std::chrono::high_resolution_clock::now();
        gauss_neon(mat, N);
        t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d_neon = t_end - t_start;
        std::cout << "NEON," << N << ",1," << d_neon.count() << std::endl;
        
        
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
            }
        }
        
        // ---------------------------------------------------------
        // 4. Pthread + NEON 
        // ---------------------------------------------------------
        if (N >= 512) {
            for (int t : thread_counts) {
                init_matrix(mat, N);
                auto t_start_final = std::chrono::high_resolution_clock::now();
                gauss_pthread_neon(mat, N, t);
                auto t_end_final = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> d_final = t_end_final - t_start_final;
                std::cout << "Pthread+NEON," << N << "," << t << "," << d_final.count() << std::endl;
            }
        }
        
        free(mat); // 清理缓存
    }
    
    return 0;
}
