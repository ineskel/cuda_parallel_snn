%%writefile fully_fused_corrected_profiled.cu
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>  // std::min, std::max

#define INPUT_SIZE 32
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 1
#define BATCH_SIZE 256
#define EPOCHS 100
#define TPB 64
#define THREADS 256
#define LR 0.002f

// ----------------- PROFILING CONTROLS -----------------
#define PROFILE 1                 // 1=enable profiling prints, 0=disable
#define PROFILE_EPOCHS 5          // profile first N epochs (keeps overhead low)
#define PROFILE_WARMUP_EPOCHS 1   // skip first epoch from profiling (warmup)
// ------------------------------------------------------

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    printf("CUDA error at %s:%d -> %s\n",__FILE__,__LINE__,cudaGetErrorString(x)); exit(1); }

static inline void CUDA_KERNEL_CHECK(){
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){
        printf("Kernel launch error -> %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

/* ===================== PROFILER HELPERS ===================== */

struct Prof {
    double fwd1_ms = 0.0;   // XW1 + ReLU
    double fwd2_ms = 0.0;   // Z1W2
    double mse_ms  = 0.0;   // mse_grad
    double gw2_ms  = 0.0;   // grad_W2
    double up2_ms  = 0.0;   // sgd(W2)
    double dz1_ms  = 0.0;   // relu_backprop_output1
    double gw1_ms  = 0.0;   // grad_W1
    double up1_ms  = 0.0;   // sgd(W1)
    double total_ms = 0.0;

    long long samples_seen = 0;

    void add_total() {
        total_ms = fwd1_ms + fwd2_ms + mse_ms + gw2_ms + up2_ms + dz1_ms + gw1_ms + up1_ms;
    }
};

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer(){ cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    float ms(){
        float out = 0.f;
        cudaEventElapsedTime(&out, a, b);
        return out;
    }
};

#if PROFILE
    #define TICK(timer) cudaEventRecord((timer).a, 0)
    #define TOCK_ACC(timer, acc_ms) do { \
        cudaEventRecord((timer).b, 0); \
        cudaEventSynchronize((timer).b); \
        (acc_ms) += (timer).ms(); \
    } while(0)
#else
    #define TICK(timer) do {} while(0)
    #define TOCK_ACC(timer, acc_ms) do {} while(0)
#endif

/* ===================== TILED KERNELS ===================== */

// -------- TILED MATMUL + RELU (FORWARD) --------
__global__ void matmul_relu_tiled(float* A, float* B, float* C,
                                 int M, int K, int N)
{
    __shared__ float As[TPB][TPB];
    __shared__ float Bs[TPB][TPB];

    int row = blockIdx.y * TPB + threadIdx.y;
    int col = blockIdx.x * TPB + threadIdx.x;

    float sum = 0.f;

    for(int t = 0; t < (K + TPB - 1) / TPB; t++){
        if(row < M && t*TPB + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row*K + t*TPB + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if(col < N && t*TPB + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TPB + threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        for(int k = 0; k < TPB; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if(row < M && col < N)
        C[row*N + col] = fmaxf(0.f, sum); // ReLU fused
}

// -------- TILED MATMUL (GENERIC) --------
__global__ void matmul_tiled(float* A, float* B, float* C,
                            int M, int K, int N)
{
    __shared__ float As[TPB][TPB];
    __shared__ float Bs[TPB][TPB];

    int row = blockIdx.y * TPB + threadIdx.y;
    int col = blockIdx.x * TPB + threadIdx.x;

    float sum = 0.f;

    for(int t = 0; t < (K + TPB - 1) / TPB; t++){
        if(row < M && t*TPB + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row*K + t*TPB + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if(col < N && t*TPB + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TPB + threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        for(int k = 0; k < TPB; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if(row < M && col < N)
        C[row*N + col] = sum;
}

/* ===================== FIXED BACKPROP KERNELS (OUTPUT_SIZE=1) ===================== */

// dW2[j] = sum_b Z1[b,j] * dY[b]
__global__ void grad_W2(float* Z1, float* dY, float* dW2, int batch, int hidden)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < hidden){
        float sum = 0.f;
        for(int b = 0; b < batch; b++)
            sum += Z1[b*hidden + j] * dY[b];
        dW2[j] = sum;
    }
}

// dZ1[b,j] = (Z1[b,j] > 0) ? dY[b] * W2[j] : 0
__global__ void relu_backprop_output1(float* dY, float* W2, float* Z1, float* dZ1,
                                     int batch, int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if(idx < total){
        int b = idx / hidden;
        int j = idx - b * hidden;
        float z = Z1[idx];
        float g = dY[b] * W2[j];
        dZ1[idx] = (z > 0.f) ? g : 0.f;
    }
}

// dW1[k,j] = sum_b X[b,k] * dZ1[b,j]
__global__ void grad_W1(float* X, float* dZ1, float* dW1, int batch, int in, int hidden)
{
    int k = blockIdx.y * blockDim.y + threadIdx.y; // input index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // hidden index

    if(k < in && j < hidden){
        float sum = 0.f;
        for(int b = 0; b < batch; b++)
            sum += X[b*in + k] * dZ1[b*hidden + j];
        dW1[k*hidden + j] = sum;
    }
}

// -------- MSE GRAD --------
__global__ void mse_grad(float* Yp, float* Y, float* dY, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        dY[i] = 2.f * (Yp[i] - Y[i]) / n;
}

// -------- SGD --------
__global__ void sgd(float* W, float* dW, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        W[i] -= LR * dW[i];
}

/* ===================== CSV LOADER ===================== */

void load_csv(const char* filename, std::vector<float>& X,
              std::vector<float>& Y, int& samples)
{
    std::ifstream file(filename);
    std::string line;
    samples = 0;

    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string val;
        int col = 0;
        while(std::getline(ss, val, ',')){
            float f = std::stof(val);
            if(col < INPUT_SIZE) X.push_back(f);
            else Y.push_back(f);
            col++;
        }
        samples++;
    }
}

// Random initialization for weights
void gpu_rand(float* d, int n){
    float* h = (float*)malloc(n * sizeof(float));
    for(int i=0;i<n;i++) h[i] = rand()/(float)RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

/* ===================== MAIN ===================== */

int main(int argc, char** argv)
{
    if(argc != 2){
        printf("Usage: %s data.csv\n", argv[0]);
        return 0;
    }

    std::vector<float> Xh, Yh;
    int samples;
    load_csv(argv[1], Xh, Yh, samples);

    printf("Loaded %d samples from %s\n", samples, argv[1]);
    printf("Model: INPUT=%d, HIDDEN=%d, OUTPUT=%d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Training: epochs=%d, batch=%d, LR=%.6f, tile(TPB)=%d\n", EPOCHS, BATCH_SIZE, (double)LR, TPB);

    // Device info (useful for paper "Experimental Setup")
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s | SMs=%d | clock=%.2f GHz | globalMem=%.2f GB\n",
           p.name, p.multiProcessorCount, p.clockRate/1e6, p.totalGlobalMem/1e9);

    long long approx_forward_flops_per_sample =
        2LL*INPUT_SIZE*HIDDEN_SIZE + 2LL*HIDDEN_SIZE*OUTPUT_SIZE;
    printf("Approx forward FLOPs/sample (mul+add): %lld (excludes backprop)\n",
           approx_forward_flops_per_sample);

    float *X, *Y, *W1, *W2, *Z1, *Yp, *dY, *dZ1, *dW1, *dW2;

    CUDA_CHECK(cudaMalloc(&X, samples*INPUT_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Y, samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W1, INPUT_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W2, HIDDEN_SIZE*sizeof(float)));                 // (256)
    CUDA_CHECK(cudaMalloc(&Z1, BATCH_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Yp, BATCH_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, BATCH_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, BATCH_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, INPUT_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, HIDDEN_SIZE*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(X, Xh.data(), samples*INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Y, Yh.data(), samples*sizeof(float), cudaMemcpyHostToDevice));

    // Initialize weights (reproducible)
    srand(1);
    gpu_rand(W1, INPUT_SIZE * HIDDEN_SIZE);
    gpu_rand(W2, HIDDEN_SIZE); // OUTPUT_SIZE = 1, so W2 is just 256

    dim3 block2d(16,16);
    dim3 gridH((HIDDEN_SIZE+TPB-1)/TPB, (BATCH_SIZE+TPB-1)/TPB);
    dim3 gridO((OUTPUT_SIZE+TPB-1)/TPB, (BATCH_SIZE+TPB-1)/TPB);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Profiling accumulators
    GpuTimer t;
    Prof prof_epoch, prof_all;
    const int profile_epochs = std::min(EPOCHS, PROFILE_EPOCHS);

#if PROFILE
    printf("\n[PROFILE] Enabled. Profiling first %d epoch(s), skipping %d warmup epoch(s).\n",
           profile_epochs, PROFILE_WARMUP_EPOCHS);
    printf("[PROFILE] Per-epoch phase breakdown will print for profiled epochs.\n");
#endif

    for(int e=0;e<EPOCHS;e++){
        prof_epoch = Prof(); // reset per-epoch accumulators

        for(int i=0;i<samples;i+=BATCH_SIZE){
            int batch = ((i+BATCH_SIZE)>samples) ? (samples-i) : BATCH_SIZE;

            const bool do_profile =
#if PROFILE
                (e < profile_epochs) && (e >= PROFILE_WARMUP_EPOCHS);
#else
                false;
#endif

            // Forward: Z1 = ReLU(XW1)
            if(do_profile) TICK(t);
            matmul_relu_tiled<<<gridH, block2d>>>(X+i*INPUT_SIZE, W1, Z1,
                                                  batch, INPUT_SIZE, HIDDEN_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.fwd1_ms);

            // Forward: Yp = Z1 W2
            if(do_profile) TICK(t);
            matmul_tiled<<<gridO, block2d>>>(Z1, W2, Yp,
                                             batch, HIDDEN_SIZE, OUTPUT_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.fwd2_ms);

            // dY = d/dYp MSE
            if(do_profile) TICK(t);
            mse_grad<<<(batch+255)/256,256>>>(Yp, Y+i, dY, batch);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.mse_ms);

            // dW2 and update W2
            if(do_profile) TICK(t);
            grad_W2<<<(HIDDEN_SIZE+255)/256,256>>>(Z1, dY, dW2, batch, HIDDEN_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.gw2_ms);

            if(do_profile) TICK(t);
            sgd<<<(HIDDEN_SIZE+255)/256,256>>>(W2, dW2, HIDDEN_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.up2_ms);

            // dZ1 = (dY * W2) * ReLU'(Z1)
            int total = batch * HIDDEN_SIZE;
            if(do_profile) TICK(t);
            relu_backprop_output1<<<(total+255)/256,256>>>(dY, W2, Z1, dZ1, batch, HIDDEN_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.dz1_ms);

            // dW1 and update W1
            dim3 gridW1((HIDDEN_SIZE+15)/16, (INPUT_SIZE+15)/16);
            if(do_profile) TICK(t);
            grad_W1<<<gridW1, block2d>>>(X+i*INPUT_SIZE, dZ1, dW1, batch, INPUT_SIZE, HIDDEN_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.gw1_ms);

            if(do_profile) TICK(t);
            sgd<<<(INPUT_SIZE*HIDDEN_SIZE+255)/256,256>>>(W1, dW1, INPUT_SIZE*HIDDEN_SIZE);
            CUDA_KERNEL_CHECK();
            if(do_profile) TOCK_ACC(t, prof_epoch.up1_ms);

            if(do_profile){
                prof_epoch.samples_seen += batch;
            }
        }

#if PROFILE
        if(e < profile_epochs && e >= PROFILE_WARMUP_EPOCHS){
            prof_epoch.add_total();
            double secs = prof_epoch.total_ms / 1000.0;
            double throughput = (secs > 0) ? (prof_epoch.samples_seen / secs) : 0.0;

            printf("\n[PROFILE] Epoch %d summary (samples=%lld):\n", e, prof_epoch.samples_seen);
            printf("  fwd1 (XW1+ReLU): %8.3f ms\n", prof_epoch.fwd1_ms);
            printf("  fwd2 (Z1W2)    : %8.3f ms\n", prof_epoch.fwd2_ms);
            printf("  mse_grad       : %8.3f ms\n", prof_epoch.mse_ms);
            printf("  grad_W2        : %8.3f ms\n", prof_epoch.gw2_ms);
            printf("  sgd_W2         : %8.3f ms\n", prof_epoch.up2_ms);
            printf("  dZ1 (ReLU bp)  : %8.3f ms\n", prof_epoch.dz1_ms);
            printf("  grad_W1        : %8.3f ms\n", prof_epoch.gw1_ms);
            printf("  sgd_W1         : %8.3f ms\n", prof_epoch.up1_ms);
            printf("  ---------------------------------\n");
            printf("  TOTAL          : %8.3f ms | Throughput: %.2f samples/s\n", prof_epoch.total_ms, throughput);

            // accumulate into global profile
            prof_all.fwd1_ms += prof_epoch.fwd1_ms;
            prof_all.fwd2_ms += prof_epoch.fwd2_ms;
            prof_all.mse_ms  += prof_epoch.mse_ms;
            prof_all.gw2_ms  += prof_epoch.gw2_ms;
            prof_all.up2_ms  += prof_epoch.up2_ms;
            prof_all.dz1_ms  += prof_epoch.dz1_ms;
            prof_all.gw1_ms  += prof_epoch.gw1_ms;
            prof_all.up1_ms  += prof_epoch.up1_ms;
            prof_all.samples_seen += prof_epoch.samples_seen;

            // Optional: CSV-like line you can paste into LaTeX/Excel
            printf("[PROFILE_CSV] epoch,%d,fwd1_ms,%.3f,fwd2_ms,%.3f,mse_ms,%.3f,gw2_ms,%.3f,up2_ms,%.3f,dz1_ms,%.3f,gw1_ms,%.3f,up1_ms,%.3f,total_ms,%.3f\n",
                   e,
                   prof_epoch.fwd1_ms, prof_epoch.fwd2_ms, prof_epoch.mse_ms,
                   prof_epoch.gw2_ms, prof_epoch.up2_ms, prof_epoch.dz1_ms,
                   prof_epoch.gw1_ms, prof_epoch.up1_ms, prof_epoch.total_ms);
        }
#endif
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("\n====================================\n");
    printf("FUSED + TILED training time: %.4f s\n", ms/1000.f);
    printf("Epochs: %d | Samples: %d | Batch: %d\n", EPOCHS, samples, BATCH_SIZE);
    printf("====================================\n");

#if PROFILE
    int prof_epochs_done = std::max(0, profile_epochs - PROFILE_WARMUP_EPOCHS);
    if(prof_epochs_done > 0){
        prof_all.add_total();
        double secs = prof_all.total_ms / 1000.0;
        double throughput = (secs > 0) ? (prof_all.samples_seen / secs) : 0.0;

        printf("\n================ PROFILING AVERAGES ================\n");
        printf("Profiled epochs: %d (warmup skipped: %d)\n", prof_epochs_done, PROFILE_WARMUP_EPOCHS);
        printf("Total profiled samples: %lld\n", prof_all.samples_seen);

        auto avg = [&](double x){ return x / prof_epochs_done; };
        printf("Avg per epoch (ms):\n");
        printf("  fwd1 (XW1+ReLU): %8.3f ms\n", avg(prof_all.fwd1_ms));
        printf("  fwd2 (Z1W2)    : %8.3f ms\n", avg(prof_all.fwd2_ms));
        printf("  mse_grad       : %8.3f ms\n", avg(prof_all.mse_ms));
        printf("  grad_W2        : %8.3f ms\n", avg(prof_all.gw2_ms));
        printf("  sgd_W2         : %8.3f ms\n", avg(prof_all.up2_ms));
        printf("  dZ1 (ReLU bp)  : %8.3f ms\n", avg(prof_all.dz1_ms));
        printf("  grad_W1        : %8.3f ms\n", avg(prof_all.gw1_ms));
        printf("  sgd_W1         : %8.3f ms\n", avg(prof_all.up1_ms));
        printf("  ---------------------------------\n");
        printf("  TOTAL          : %8.3f ms\n", avg(prof_all.total_ms));
        printf("Overall profiled throughput: %.2f samples/s\n", throughput);

        double T = prof_all.total_ms;
        if(T > 0){
            printf("\nPercent of total time (profiled window):\n");
            printf("  fwd1: %5.1f%% | fwd2: %5.1f%% | mse: %5.1f%% | dW2: %5.1f%% | W2upd: %5.1f%% | dZ1: %5.1f%% | dW1: %5.1f%% | W1upd: %5.1f%%\n",
                100.0*prof_all.fwd1_ms/T, 100.0*prof_all.fwd2_ms/T, 100.0*prof_all.mse_ms/T,
                100.0*prof_all.gw2_ms/T,  100.0*prof_all.up2_ms/T,  100.0*prof_all.dz1_ms/T,
                100.0*prof_all.gw1_ms/T,  100.0*prof_all.up1_ms/T
            );
        }
        printf("====================================================\n");
    }
#endif

    cudaDeviceReset();
    return 0;
}
