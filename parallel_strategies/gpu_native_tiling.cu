%%writefile gpu_defused_tiled_csv.cu
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 32
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 1
#define BATCH_SIZE 256
#define EPOCHS 100
#define TPB 64
#define LR 0.002f

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    printf("CUDA error at %s:%d -> %s\n",__FILE__,__LINE__,cudaGetErrorString(x)); exit(1); }

/* ===================== TILED MATMUL ===================== */

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

/* ===================== ACTIVATIONS ===================== */

__global__ void relu(float* Z, int M, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < M && c < N)
        Z[r*N + c] = fmaxf(0.f, Z[r*N + c]);
}

__global__ void relu_grad(float* dZ, float* Z, int M, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < M && c < N)
        dZ[r*N + c] = (Z[r*N + c] > 0.f) ? dZ[r*N + c] : 0.f;
}

/* ===================== LOSS ===================== */

__global__ void mse_grad(float* Yp, float* Y, float* dY, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        dY[i] = 2.f * (Yp[i] - Y[i]) / n;
}

/* ===================== SGD ===================== */

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

/* ===================== INIT ===================== */

void gpu_rand(float* d, int n)
{
    std::vector<float> h(n);
    for(int i=0;i<n;i++) h[i] = rand()/(float)RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
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

    float *X, *Y, *W1, *W2, *Z1, *Yp;
    float *dY, *dZ1, *dW1, *dW2;

    CUDA_CHECK(cudaMalloc(&X, samples*INPUT_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Y, samples*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W1, INPUT_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W2, HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z1, BATCH_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Yp, BATCH_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, BATCH_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, BATCH_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, INPUT_SIZE*HIDDEN_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, HIDDEN_SIZE*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(X, Xh.data(), samples*INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Y, Yh.data(), samples*sizeof(float), cudaMemcpyHostToDevice));

    srand(1);
    gpu_rand(W1, INPUT_SIZE*HIDDEN_SIZE);
    gpu_rand(W2, HIDDEN_SIZE);

    dim3 block(32, 32);
    dim3 gridH((HIDDEN_SIZE+TPB-1)/TPB, (BATCH_SIZE+TPB-1)/TPB);
    dim3 gridO((OUTPUT_SIZE+TPB-1)/TPB, (BATCH_SIZE+TPB-1)/TPB);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int e=0;e<EPOCHS;e++){
        for(int i=0;i<samples;i+=BATCH_SIZE){
            int batch = ((i+BATCH_SIZE)>samples)?(samples-i):BATCH_SIZE;

            /* ---------- FORWARD (DEFUSED) ---------- */
            matmul_tiled<<<gridH, block>>>(X+i*INPUT_SIZE, W1, Z1,
                                           batch, INPUT_SIZE, HIDDEN_SIZE);
            relu<<<gridH, block>>>(Z1, batch, HIDDEN_SIZE);

            matmul_tiled<<<gridO, block>>>(Z1, W2, Yp,
                                           batch, HIDDEN_SIZE, OUTPUT_SIZE);

            /* ---------- BACKWARD (DEFUSED) ---------- */
            mse_grad<<<(batch+255)/256,256>>>(Yp, Y+i, dY, batch);

            matmul_tiled<<<gridO, block>>>(Z1, dY, dW2,
                                           HIDDEN_SIZE, batch, OUTPUT_SIZE);
            sgd<<<(HIDDEN_SIZE+255)/256,256>>>(W2, dW2, HIDDEN_SIZE);

            matmul_tiled<<<gridH, block>>>(dY, W2, dZ1,
                                           batch, OUTPUT_SIZE, HIDDEN_SIZE);
            relu_grad<<<gridH, block>>>(dZ1, Z1, batch, HIDDEN_SIZE);

            matmul_tiled<<<gridH, block>>>(X+i*INPUT_SIZE, dZ1, dW1,
                                           INPUT_SIZE, batch, HIDDEN_SIZE);
            sgd<<<(INPUT_SIZE*HIDDEN_SIZE+255)/256,256>>>(W1, dW1,
                                                         INPUT_SIZE*HIDDEN_SIZE);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("====================================\n");
    printf("DE-FUSED + TILED training time: %.4f s\n", ms/1000.f);
    printf("Epochs: %d | Samples: %d | Batch: %d\n",
           EPOCHS, samples, BATCH_SIZE);
    printf("====================================\n");

    cudaDeviceReset();
    return 0;
}
