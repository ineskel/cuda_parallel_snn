# Performance Analysis of CUDA-Based Parallelization Strategies for a Shallow Neural Network

> A systematic study of CUDA optimization techniques — tiled matrix multiplication, full GPU offloading, and kernel fusion — applied to training a shallow neural network, achieving up to **35× speedup** over a hybrid CPU/GPU baseline.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Network Architecture](#network-architecture)
- [Parallelization Strategies](#parallelization-strategies)
  - [Baseline: Reference CUDA Implementation](#baseline-reference-cuda-implementation)
  - [Optimization 1: Tiled Matrix Multiplication](#optimization-1-tiled-matrix-multiplication)
  - [Optimization 2: Full GPU Offloading](#optimization-2-full-gpu-offloading)
  - [Optimization 3: Kernel Fusion](#optimization-3-kernel-fusion)
  - [Combined: Tiling + Fusion + GPU-Native](#combined-tiling--fusion--gpu-native)
- [Results](#results)
- [Experimental Setup](#experimental-setup)
- [Authors](#authors)
- [References](#references)

---

## Overview

Matrix multiplication dominates both forward and backward propagation in shallow neural networks. This project analyzes how CUDA kernel design — specifically thread mapping, memory access patterns, and operation fusion — impacts training performance across small, medium, and large datasets.

The reference implementation maps each output matrix element to a single thread with no memory reuse. Three complementary optimizations are then layered on top, each targeting a specific bottleneck identified through NVIDIA Nsight Systems profiling.

---

## Repository Structure

```
.
├── paper/
│   ├── figures/                  # Profiling screenshots and result plots
│   ├── main.tex                  # LaTeX source of the paper
│   └── Parallel_Optimization_Strategies_CUDA_SNN.pdf
├── parallel_strategies/
│   ├── fully_fused.cu            # Kernel fusion + tiling (combined final strategy)
│   ├── gpu_native_strategy.cu    # Full GPU offloading (no host-device mid-loop transfers)
│   └── gpu_native_tiling.cu     # GPU-native execution with tiled matrix multiplication
├── reference/                    # Baseline hybrid CPU/GPU implementation
├── .gitignore
└── LICENSE
```

---

## Network Architecture

| Parameter        | Value                        |
|------------------|------------------------------|
| Input features   | 32                           |
| Hidden layer     | 256 neurons, ReLU activation |
| Output layer     | 1 neuron (regression)        |
| Loss function    | Mean Squared Error (MSE)     |
| Optimizer        | Mini-batch SGD               |
| Batch size       | 256                          |
| Learning rate    | 0.002                        |
| Epochs           | 100                          |

Training data is synthetically generated: inputs sampled uniformly from [−1, 1] with additive Gaussian noise on targets, ensuring reproducibility and controlled comparison across strategies.

---

## Parallelization Strategies

### Baseline: Reference CUDA Implementation

Each element of the output matrix is assigned to one CUDA thread. Threads are organized in 2D blocks matching the output dimensions.

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < A_rows && col < B_cols) {
    float value = 0.0f;
    for (int k = 0; k < A_cols; k++)
        value += A[row * A_cols + k] * B[k * B_cols + col];
    C[row * B_cols + col] = value;
}
```

**Profiling revealed:** the kernel itself accounts for only 0.343 s of the 11.4 s total training time on the large dataset. Over **85%** of CUDA API time is consumed by `cudaMalloc`, `cudaFree`, `cudaMemcpy`, and `cudaDeviceSynchronize` — all called at every iteration.

| Dataset | Batches | Pthreads (s) | CUDA (s) |
|---------|---------|--------------|----------|
| Small   | 1       | 0.83         | 0.45     |
| Medium  | 10      | 7.44         | 2.59     |
| Large   | 100     | 74.00        | 22.22    |

---

### Optimization 1: Tiled Matrix Multiplication

**File:** `parallel_strategies/gpu_native_tiling.cu`

Partitions input matrices into `T × T` tiles loaded cooperatively into shared memory. Each thread block computes one output tile, reusing data from shared memory for all partial dot products — eliminating redundant global memory fetches.

```
C_ij = Σ_{t} Σ_{k} A[i, tT+k] · B[tT+k, j]
```

Two `__syncthreads()` barriers per tile iteration ensure correctness. Tile size is a tunable parameter with hardware limit of 1024 threads/block.

**Impact by tile size (execution time in ms):**

| Dataset | Tile 8  | Tile 16 | Tile 32 | Tile 64 |
|---------|---------|---------|---------|---------|
| Small   | **2321**| 2323    | 2330    | —       |
| Medium  | —       | 23068   | 23514   | **22722**|
| Large   | —       | 230885  | 232759  | **229877**|

Key insight: optimal tile size scales with dataset — `T=8` wins for small inputs, `T=64` for large ones where memory bandwidth is the dominant constraint.

---

### Optimization 2: Full GPU Offloading

**File:** `parallel_strategies/gpu_native_strategy.cu`

Moves the entire training loop to the GPU. In the hybrid baseline, activations, loss, and weight updates ran on the CPU, forcing intermediate results to cross the PCIe bus on every iteration. This strategy allocates all tensors once in device memory and keeps them there for all 100 epochs.

- Data transfers occur **twice total**: once at init, once at the end
- Forward pass, backward pass, SGD updates — all CUDA kernels
- No `cudaMemcpy` inside the training loop

| Dataset | Samples | Hybrid (s) | GPU-Native (s) | Speedup |
|---------|---------|------------|----------------|---------|
| Small   | 256     | 0.3151     | 0.0187         | **16.85×** |
| Medium  | 2,560   | 3.6767     | 0.1752         | **20.98×** |
| Large   | 25,600  | 7.5937     | 1.3527         | **5.62×**  |

---

### Optimization 3: Kernel Fusion

**File:** `parallel_strategies/fully_fused.cu` (fusion component)

Merges consecutive dependent operations into a single kernel to eliminate intermediate global memory writes and reduce kernel launch overhead.

- **Forward pass:** `matmul` + `ReLU` fused — each thread computes its dot product and immediately applies activation before writing to global memory
- **Backward pass:** ReLU gradient computed inline during the weight-gradient matmul

| Dataset | Unfused (s) | Fused (s) | Speedup |
|---------|-------------|-----------|---------|
| Small   | 0.01827     | 0.0145    | 1.26×   |
| Medium  | 0.1752      | 0.1045    | 1.68×   |
| Large   | 1.3527      | 0.7267    | **1.86×** |

---

### Combined: Tiling + Fusion + GPU-Native

**File:** `parallel_strategies/fully_fused.cu`

The final implementation stacks all three optimizations:

```
Algorithm: GPU-Resident Training with Tiling and Kernel Fusion

Initialize W1, W2 on GPU
for each epoch:
  for each batch:
    Forward:  matmul_relu_tiled(X_batch, W1, Z1)
              matmul_tiled(Z1, W2, Y_hat)
    Backward: relu_backprop_output(dY, W2, Z1, dZ1)
              grad_W1(X_batch, dZ1, dW1)
              grad_W2(Z1, dY, dW2)
    Update:   sgd(W1, dW1), sgd(W2, dW2)
```

End-to-end training time with `T=16`, batch size 256, 100 epochs:

| Dataset | Samples | Time (s) |
|---------|---------|----------|
| Small   | 256     | 0.0106   |
| Medium  | 2,560   | 0.1045   |
| Large   | 25,600  | 0.8401   |

Memory transfer overhead is negligible: only 4 host-to-device transfers totalling 3.41 MB and 334 µs over the full training run.

---

## Results

Final speedup of the optimized implementation over the hybrid baseline:

| Dataset | Baseline (s) | Optimized (s) | Speedup     |
|---------|--------------|---------------|-------------|
| Small   | 0.3151       | 0.0106        | **~29.7×**  |
| Medium  | 3.6767       | 0.1045        | **~35.2×**  |
| Large   | 7.5937       | 0.8401        | **~9.0×**   |

**GPU kernel time breakdown (large dataset, final implementation):**

| Kernel               | Time (%) |
|----------------------|----------|
| `grad_W1`            | 40.3     |
| `matmul_tiled`       | 27.3     |
| `grad_W2`            | 11.4     |
| Activation backprop  | 4.1      |
| SGD update           | 3.9      |
| MSE computation      | 1.9      |

> **Note on combined optimizations:** Tiling + fusion interact non-trivially. For the largest dataset, GPU-native + fusion (without tiling) produced the best wall-clock time due to increased register/shared-memory pressure from tiling reducing warp occupancy on the Tesla T4. Optimal strategy is shape- and hardware-dependent.

---

## Experimental Setup

| Component            | Specification                  |
|----------------------|-------------------------------|
| Platform             | Google Colab (KVM VM)          |
| CPU                  | Intel Xeon @ 2.00 GHz, 2 vCPUs|
| RAM                  | 12 GB                          |
| GPU                  | NVIDIA Tesla T4                |
| GPU Memory           | 15 GB                          |
| CUDA Version         | 12.8                           |
| GPU Driver           | 580.82.07                      |
| Max threads per block| 1024                           |

---

## Authors

- Bouameur Besmala Aicha
- Djafour Touria
- Hammou Manel
- Kellou Lylia Ines
- Messaoud Amal

---

## References

1. Brouthen, K., & Akeb, A. (2025). *Exploring parallelization of shallow neural network using CUDA.* Student report, ESI.
2. NVIDIA Corporation. (2023). *CUDA C++ Programming Guide.* https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. NVIDIA Corporation. (2023). *NVIDIA Nsight Systems User Guide.* https://docs.nvidia.com/nsight-systems/
