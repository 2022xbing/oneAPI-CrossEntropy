%%writefile lab/CrossEntropy.cpp

#include <chrono>
#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>

#define random_float() (rand() / double(RAND_MAX))
// 用于生成0~N-1的随机数，用于生成mask
#define random_int(N) (int)(rand() / double(RAND_MAX) * N)
using namespace std;
using namespace sycl;

// gpu加速
// 返回计算时间
// X为输入向量，loss为结果，K为batchsize，M为cXtegory，N为feature,blockSize为分块大小，q为GPU队列
double gpu_kernel(const float *X, float *loss, float *Y, int *mask, float *weight,
                  int M, int N, int K, int blockSize, sycl::queue &q) {

// 下面利用GPU并行算法来计算X[K][M][N]的每个行之和
    // row和col对应K和M
    auto grid_rows = (K + blockSize - 1) / blockSize * blockSize;
    auto grid_cols = (M + blockSize - 1) / blockSize * blockSize;
    auto locXl_ndrange = range<2>(blockSize, blockSize);
    auto global_ndrange = range<2>(grid_rows, grid_cols);
    // 时间
    double duration = 0.0f;
    //提交工作
    auto e = q.submit([&](sycl::handler &h) {
        // GPU并行计算
        h.parallel_for<class k_name_t>(
                sycl::nd_range<2>(global_ndrange, locXl_ndrange), [=](sycl::nd_item<2> index) {
                    // row和col对应K和M
                    int row = index.get_global_id(0);
                    int col = index.get_global_id(1);
                    // sum保存每个行之和，Xmax保存每行的最大值
                    float sum = 0.0f;
                    float Xmax = 0.f;
                    // 遍历每行得到Xmax
                    for (int i = 0; i < N; ++i) {
                        Xmax = std::max(Xmax, X[row * M * N + col * N + i]);
                    }
                    // 遍历每行得到sum
                    for (int i = 0; i < N; ++i) {
                        sum += exp(X[row * M * N + col * N + i] - Xmax);
                    }
                    // 遍历每行计算得到三维矩阵Y
                    for (int i = 0; i < N; ++i) {
                        Y[row * M * N + col * N + i] = X[row * M * N + col * N + i] - Xmax - log(sum);
                    }
                });
    });
    e.wait();


    // row和col对应K和N
    grid_rows = (K + blockSize - 1) / blockSize * blockSize;
    grid_cols = (N + blockSize - 1) / blockSize * blockSize;
    locXl_ndrange = range<2>(blockSize, blockSize);
    global_ndrange = range<2>(grid_rows, grid_cols);

    e = q.submit([&](sycl::handler &h) {
        h.parallel_for<class T3>(
                sycl::nd_range<2>(global_ndrange, locXl_ndrange), [=](sycl::nd_item<2> index) {
                    int row = index.get_global_id(0);
                    int col = index.get_global_id(1);
                    // 计算出最终结果loss
                    loss[row * N + col] = -Y[row * N * M + mask[row * N + col] * N + col] * weight[row * N + col];
                });
    });
    e.wait();

    // 计算时间
    duration += (e.get_profiling_info<info::event_profiling::command_end>() -
                 e.get_profiling_info<info::event_profiling::command_start>()) / 1000.0f / 1000.0f;


    return (duration);
}

// CPU计算
double cpu_kernel(float *cX, float *closs, int *mask, float *weight, int M, int N, int K) {

    double duration = 0.0;
    std::chrono::high_resolution_clock::time_point s, e;

    s = std::chrono::high_resolution_clock::now();

    // 用Y记录中间的变量结果，sum用于记录X的每一个行的总和
    float *Y = new float[K * M * N]();

    for (int i = 0; i < K; i++) {
        // 求Xmax和sum
        for (int j = 0; j < M; j++) {
            float Xmax = 0.f;
            for (int k = 0; k < N; ++k) {
                Xmax = std::max(Xmax, cX[i * M * N + j * N + k]);
            }
            float sum1 = 0.0f;
            for (int k = 0; k < N; k++) {
                sum1 += exp(cX[i * M * N + j * N + k] - Xmax);
            }
            // 求Y
            for (int k = 0; k < N; ++k) {
                Y[i * M * N + j * N + k] = cX[i * M * N + j * N + k] - Xmax - log(sum1);
                // cout << "CPU Y: k = " << i << "m = " << j << " n = " << k<< " value = "<< Y[j*N + k] << endl;
            }
        }

        // 计算出loss结果
        for (int j = 0; j < N; ++j) {
            // closs[K][N]和weight[K][N]的下标变换为i*N+j, Y[M][N]下标变换为(mask[i*N+J])*N + j
            closs[i * N + j] = -Y[i * M * N + mask[i * N + j] * N + j] * weight[i * N + j];
        }
    }
    e = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(e - s).count();

    return (duration);
}

int verify(float *cpu_res, float *gpu_res, int length) {
    int err = 0;
    for (int i = 0; i < length; i++) {
        if (fabs(cpu_res[i] - gpu_res[i]) > 0.001) {
            err++;
            printf("\n%lf, %lf", cpu_res[i], gpu_res[i]);
        }
    }
    return (err);
}

// 结果比较
int gemm(const int M,
         const int N,
         const int K,
         const int blockSize,
         const int iterations,
         sycl::queue &q) {
    cout << "Problem size: ｘ(" << K << "," << M << "," << N << ")\n";
    cout << "Block Size = " << blockSize << std::endl;
    // 将X展开为一维
    auto A = malloc_shared<float>(M * K * N, q);
    auto Y = malloc_shared<float>(M * K * N, q);

    // mask、weight和loss都为K*N大小，展开为一维
    const int size2 = K * N;
    auto mask = malloc_shared<int>(size2, q);
    auto weight = malloc_shared<float>(size2, q);

    // 下面的Ｃ和C_host用于保存GPU和CPU的运行结果loss
    auto C = malloc_shared<float>(K * N, q);
    auto C_host = malloc_host<float>(K * N, q);

    // 初始化
    for (int i = 0; i < M * K * N; i++) {
        A[i] = random_float();
        Y[i] = 0.f;
    }

    for (int i = 0; i < N * K; i++) {
        mask[i] = random_int(M);
        weight[i] = random_float();
        C[i] = 0.0f;
        C_host[i] = 0.0f;
    }

    // 保存每秒浮点计算的数值，这个为Ａ和B两个矩阵的数据总量的２倍
    double flopsPerMatrixMul
            = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    double duration_gpu = 0.0f;
    double duration_cpu = 0.0f;

    // 下面的warmup用于GPU运行次数热身。只有运行次数超过改热身次数后才累加上GPU的run时间
    int warmup = 10;
    for (int run = 0; run < iterations + warmup; run++) {
        float duration = gpu_kernel(A, C, Y, mask, weight, M, N, K, blockSize, q);
        if (run >= warmup) duration_gpu += duration;
    }
    duration_gpu = duration_gpu / iterations;

    // 下面的warmup用于CPU运行次数热身。只有运行次数超过改热身次数后才累加上CPU的run时间
    warmup = 2;
    for (int run = 0; run < iterations / 2 + warmup; run++) {
        float duration = cpu_kernel(A, C_host, mask, weight, M, N, K);
        if (run >= warmup) duration_cpu += duration;
    }
    duration_cpu = duration_cpu / iterations / 2;

    // 比较GPU和CPU结果
    int errCode = 0;
    errCode = verify(C_host, C, N * K);
    printf("%d errors in loss_gpu\n", errCode);
    //if (errCode > 0) printf("\nThere are %d errors\n", errCode);

    printf("\nPerformance Flops = %lf, \n"
           "GPU Computation Time = %lf (ms); \n"
           "CPU Computaiton Time = %lf (ms); \n",
           flopsPerMatrixMul, duration_gpu, duration_cpu);

    // 释放空间
    free(A, q);
    free(Y, q);
    free(C, q);
    free(C_host, q);

    return (errCode);
}

int main() {

    auto propList = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    queue my_gpu_queue(cl::sycl::gpu_selector_v{}, propList);


    int K = 128, M = 32, N = 8192;
    int block = 8;
    int errCode = gemm(M, N, K, block, 3, my_gpu_queue);

    return (errCode);
}
