/*
 * matmul_kernels.cu
 * ─────────────────────────────────────────────────────────────────
 * Project 1: Custom Matrix Multiplication that beats cuBLAS
 * on small-to-medium matrix shapes.
 *
 * Three kernels implemented:
 *   1. naive_matmul      – one thread per output element, global mem only
 *   2. tiled_matmul      – shared memory tiling (TILE x TILE blocks)
 *   3. tiled_matmul_db   – double-buffered tiles (hides memory latency)
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 --maxrregcount=64 matmul_kernels.cu \
 *        -lcublas -o matmul_bench
 *
 * Run:
 *   ./matmul_bench
 * ─────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ── tuneable constants ─────────────────────────────────────── */
#define TILE      16          /* shared-memory tile width/height  */
#define TILE_DB   16          /* double-buffer tile size           */

/* ── error-checking macros ──────────────────────────────────── */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do {                                                                \
        cublasStatus_t st = (call);                                     \
        if (st != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",           \
                    __FILE__, __LINE__, (int)st);                       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)


/* ═══════════════════════════════════════════════════════════════
   KERNEL 1 – Naive
   One thread computes one element of C by looping over K.
   No data reuse → every element fetches N floats from global mem.
   ═══════════════════════════════════════════════════════════════ */
__global__ void naive_matmul(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}


/* ═══════════════════════════════════════════════════════════════
   KERNEL 2 – Tiled with shared memory
   Each block loads a TILE×TILE chunk of A and B into shared mem,
   syncs, computes partial dot products, then moves to next tile.
   Memory traffic: N/TILE times fewer global reads vs naive.
   ═══════════════════════════════════════════════════════════════ */
__global__ void tiled_matmul(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx  = threadIdx.x,  ty  = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        /* ── load tile into shared memory (boundary-safe) ─── */
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        sA[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        sB[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();   /* all threads done loading → safe to compute */

        /* ── compute partial dot product for this tile ───── */
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[ty][k] * sB[k][tx];

        __syncthreads();   /* done reading shared mem → safe to overwrite */
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}


/* ═══════════════════════════════════════════════════════════════
   KERNEL 3 – Double-buffered tiled
   While computing on tile T, we prefetch tile T+1 into the
   second shared-memory buffer, hiding global-memory latency.
   Useful when the GPU has enough registers/smem for two buffers.
   ═══════════════════════════════════════════════════════════════ */
__global__ void tiled_matmul_db(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float*       __restrict__ C,
                                 int N)
{
    /* two ping-pong buffers */
    __shared__ float sA[2][TILE_DB][TILE_DB];
    __shared__ float sB[2][TILE_DB][TILE_DB];

    int tx  = threadIdx.x,  ty  = threadIdx.y;
    int row = blockIdx.y * TILE_DB + ty;
    int col = blockIdx.x * TILE_DB + tx;

    float acc   = 0.0f;
    int num_tiles = (N + TILE_DB - 1) / TILE_DB;

    /* prefetch tile 0 into buffer 0 */
    int a_col = tx, b_row = ty;
    sA[0][ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
    sB[0][ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int t = 0; t < num_tiles - 1; ++t) {
        int cur  = t & 1;
        int next = 1 - cur;

        /* prefetch next tile while computing current */
        int na_col = (t + 1) * TILE_DB + tx;
        int nb_row = (t + 1) * TILE_DB + ty;
        sA[next][ty][tx] = (row < N && na_col < N) ? A[row * N + na_col] : 0.0f;
        sB[next][ty][tx] = (nb_row < N && col < N) ? B[nb_row * N + col] : 0.0f;

        /* compute on current buffer */
        #pragma unroll
        for (int k = 0; k < TILE_DB; ++k)
            acc += sA[cur][ty][k] * sB[cur][k][tx];

        __syncthreads();
    }

    /* compute last tile */
    int last = (num_tiles - 1) & 1;
    #pragma unroll
    for (int k = 0; k < TILE_DB; ++k)
        acc += sA[last][ty][k] * sB[last][k][tx];

    if (row < N && col < N)
        C[row * N + col] = acc;
}


/* ═══════════════════════════════════════════════════════════════
   HELPER – GPU timer using CUDA events
   Returns elapsed milliseconds.
   ═══════════════════════════════════════════════════════════════ */
static float gpu_time_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}


/* ═══════════════════════════════════════════════════════════════
   HELPER – Verify correctness against reference (cuBLAS)
   Returns max absolute error.
   ═══════════════════════════════════════════════════════════════ */
static float max_error(const float* ref, const float* cmp, int N)
{
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float e = fabsf(ref[i] - cmp[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}


/* ═══════════════════════════════════════════════════════════════
   HELPER – Compute GFLOPs for NxN matmul
   ═══════════════════════════════════════════════════════════════ */
static double gflops(int N, float ms)
{
    /* 2*N^3 floating-point ops (multiply + add per inner step) */
    return (2.0 * N * N * N) / (ms * 1e6);
}






/* ═══════════════════════════════════════════════════════════════
   BENCHMARK – one matrix size
   ═══════════════════════════════════════════════════════════════ */
static void benchmark(int N, cublasHandle_t handle,
                      cudaEvent_t ev_start, cudaEvent_t ev_stop,
                      int warmup, int reps)
{
    printf("\n── N = %4d ──────────────────────────────────────\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);

    /* ── allocate host ─────────────────────────────────────── */
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_ref  = (float*)malloc(bytes);  /* cuBLAS reference  */
    float *hC_our  = (float*)malloc(bytes);  /* our kernel result */

    /* random init */
    for (int i = 0; i < N * N; ++i) {
        hA[i] = (float)rand() / RAND_MAX - 0.5f;
        hB[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    /* ── allocate device ───────────────────────────────────── */
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    float alpha = 1.0f, beta = 0.0f;
    float ms, best_ms;

    /* ── cuBLAS (reference + baseline) ────────────────────── */
    /* Note: cuBLAS is column-major, so we compute B^T * A^T = (AB)^T,
       which gives us the correct row-major result in C. */
    for (int i = 0; i < warmup; ++i)
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha, dB, N, dA, N, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < reps; ++i)
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha, dB, N, dA, N, &beta, dC, N));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    ms = gpu_time_ms(ev_start, ev_stop) / reps;
    CUDA_CHECK(cudaMemcpy(hC_ref, dC, bytes, cudaMemcpyDeviceToHost));
    printf("  cuBLAS           : %7.3f ms   %6.1f GFLOP/s  (reference)\n",
           ms, gflops(N, ms));

    /* ── Naive kernel ──────────────────────────────────────── */
    for (int i = 0; i < warmup; ++i)
        naive_matmul<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    best_ms = 1e9f;
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < reps; ++i)
        naive_matmul<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    ms = gpu_time_ms(ev_start, ev_stop) / reps;
    CUDA_CHECK(cudaMemcpy(hC_our, dC, bytes, cudaMemcpyDeviceToHost));
    printf("  Naive kernel     : %7.3f ms   %6.1f GFLOP/s  err=%.2e\n",
           ms, gflops(N, ms), max_error(hC_ref, hC_our, N));

    /* ── Tiled kernel ──────────────────────────────────────── */
    for (int i = 0; i < warmup; ++i)
        tiled_matmul<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < reps; ++i)
        tiled_matmul<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    ms = gpu_time_ms(ev_start, ev_stop) / reps;
    CUDA_CHECK(cudaMemcpy(hC_our, dC, bytes, cudaMemcpyDeviceToHost));
    printf("  Tiled (T=%2d)     : %7.3f ms   %6.1f GFLOP/s  err=%.2e",
           TILE, ms, gflops(N, ms), max_error(hC_ref, hC_our, N));
    /* flag if we beat cuBLAS */
    {
        /* re-run cuBLAS quickly for fair compare */
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < reps; ++i)
            CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha, dB, N, dA, N, &beta, dC, N));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        float cublas_ms = gpu_time_ms(ev_start, ev_stop) / reps;
        if (ms < cublas_ms) printf("  ← BEATS cuBLAS (%.2fx faster)", cublas_ms / ms);
    }
    printf("\n");

    /* ── Double-buffered tiled kernel ──────────────────────── */
    for (int i = 0; i < warmup; ++i)
        tiled_matmul_db<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < reps; ++i)
        tiled_matmul_db<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    ms = gpu_time_ms(ev_start, ev_stop) / reps;
    CUDA_CHECK(cudaMemcpy(hC_our, dC, bytes, cudaMemcpyDeviceToHost));
    printf("  Tiled DB (T=%2d)  : %7.3f ms   %6.1f GFLOP/s  err=%.2e\n",
           TILE_DB, ms, gflops(N, ms), max_error(hC_ref, hC_our, N));

    /* ── cleanup ───────────────────────────────────────────── */
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC_ref); free(hC_our);
}


/* ═══════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════ */
int main(void)
{
    /* print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  |  SM count: %d  |  Mem: %.0f MB  |  L2: %.0f KB\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6,
           prop.l2CacheSize / 1024.0f);
    printf("Shared mem/block: %.0f KB  |  Max threads/block: %d\n",
           prop.sharedMemPerBlock / 1024.0f, prop.maxThreadsPerBlock);
    printf("Tile size: %d x %d  |  Block: %d threads\n\n",
           TILE, TILE, TILE * TILE);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    srand(42);

    /* sweep matrix sizes: small → large */
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int warmup  = 3;
    int reps    = 20;

    printf("%-18s  %10s  %14s  %s\n",
           "Kernel", "Time (ms)", "GFLOP/s", "Notes");
    printf("────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < (int)(sizeof(sizes)/sizeof(sizes[0])); ++i)
        benchmark(sizes[i], handle, ev_start, ev_stop, warmup, reps);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    printf("\nDone. Copy the table above into your README.\n");
    return 0;
}
