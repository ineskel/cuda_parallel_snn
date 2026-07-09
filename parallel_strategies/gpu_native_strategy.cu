%%writefile gpu_end_to_end.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 32
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 1
#define BATCH_SIZE 256
#define EPOCHS 100
#define THREADS 256
#define LEARNING_RATE 0.002f

#define CHECK(call) \
    if ((call) != cudaSuccess) { \
        printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    }

// ================= KERNELS =================

__global__ void matmul(float *A, float *B, float *C, int M, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[row * K + i] * B[i * N + col];
        C[idx] = sum;
    }
}

__global__ void relu(float *A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        A[i] = fmaxf(0.f, A[i]);
}

__global__ void mse_grad(float *pred, float *y, float *grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        grad[i] = 2.f * (pred[i] - y[i]) / n;
}

__global__ void sgd(float *W, float *dW, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        W[i] -= LEARNING_RATE * dW[i];
}

// ================= HOST =================

int main() {
    int samples = 4096;

    float *X, *Y;
    CHECK(cudaMallocManaged(&X, samples * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&Y, samples * sizeof(float)));

    for (int i = 0; i < samples * INPUT_SIZE; i++) X[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < samples; i++) Y[i] = rand() / (float)RAND_MAX;

    float *W1, *W2, *Z1, *Yp, *dZ2, *dW1, *dW2;
    CHECK(cudaMallocManaged(&W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&W2, HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&Z1, samples * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&Yp, samples * sizeof(float)));
    CHECK(cudaMallocManaged(&dZ2, samples * sizeof(float)));
    CHECK(cudaMallocManaged(&dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&dW2, HIDDEN_SIZE * sizeof(float)));

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) W1[i] = 0.01f;
    for (int i = 0; i < HIDDEN_SIZE; i++) W2[i] = 0.01f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int e = 0; e < EPOCHS; e++) {
        matmul<<<(samples * HIDDEN_SIZE + THREADS - 1) / THREADS, THREADS>>>
            (X, W1, Z1, samples, INPUT_SIZE, HIDDEN_SIZE);

        relu<<<(samples * HIDDEN_SIZE + THREADS - 1) / THREADS, THREADS>>>
            (Z1, samples * HIDDEN_SIZE);

        matmul<<<(samples + THREADS - 1) / THREADS, THREADS>>>
            (Z1, W2, Yp, samples, HIDDEN_SIZE, 1);

        mse_grad<<<(samples + THREADS - 1) / THREADS, THREADS>>>
            (Yp, Y, dZ2, samples);

        matmul<<<(HIDDEN_SIZE + THREADS - 1) / THREADS, THREADS>>>
            (Z1, dZ2, dW2, HIDDEN_SIZE, samples, 1);

        sgd<<<(HIDDEN_SIZE + THREADS - 1) / THREADS, THREADS>>>
            (W2, dW2, HIDDEN_SIZE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU end-to-end training time: %.4f s\n", ms / 1000.f);

    cudaDeviceReset();
    return 0;
}
