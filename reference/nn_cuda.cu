// Parallel matrix multiplication using CUDA
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ! Network Parameters
#define INPUT_SIZE      32      // Number of input features
#define HIDDEN_SIZE     256     // Number of neurons in the hidden layer
#define OUTPUT_SIZE     1      // Number of output neurons
#define EPOCHS          100   // Number of training epochs
#define LOG_EVERY_EPOCH 1     // Log loss every n epochs
#define LEARNING_RATE   0.002
#define BATCH_SIZE      256     // Batch size for SGD
#define THREADS_PER_BLOCK 16

// ! Data Structures
// Structure for matrix
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

// ! Memory Management
// Function to allocate a matrix (contiguous memory allocation)
Matrix* allocate_matrix(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(rows * cols * sizeof(float));  // Contiguous memory
    return m;
}

// Function to free a matrix
void free_matrix(Matrix *m) {
    free(m->data);  // Free contiguous memory
    free(m);
}

// ! Matrix Operations
// Function to initialize matrix with random values using He initialization
void random_init(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            // Initialize with random values between 0 and 1
            m->data[i * m->cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

// ! Matrix Operations (GPU version)
__global__ void mat_mult_kernel(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float value = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            value += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = value;
    }
}

// Function to multiply matrices on the GPU
Matrix* mat_mult(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Incompatible matrices for multiplication.\n");
        exit(1);
    }

    Matrix *C = allocate_matrix(A->rows, B->cols);

    float *d_A, *d_B, *d_C;
    size_t sizeA = A->rows * A->cols * sizeof(float);
    size_t sizeB = B->rows * B->cols * sizeof(float);
    size_t sizeC = C->rows * C->cols * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, A->data, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((B->cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (A->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->rows, A->cols, B->cols);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(C->data, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

// Matrix subtraction: C = A - B
Matrix* mat_sub(Matrix *A, Matrix *B) {
    if(A->rows != B->rows || A->cols != B->cols) {
        printf("Incompatible matrices for subtraction.\n");
        exit(1);
    }
    Matrix *C = allocate_matrix(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->cols; j++)
            C->data[i * A->cols + j] = A->data[i * A->cols + j] - B->data[i * A->cols + j];
    return C;
}

// Matrix scalar multiplication: A = A * scalar
void mat_scalar_mult(Matrix *A, float scalar) {
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->cols; j++)
            A->data[i * A->cols + j] *= scalar;
}

// ! Activation Functions
// Function to apply ReLU activation
void relu(Matrix *m) {
    for(int i = 0; i < m->rows; i++)
        for(int j = 0; j < m->cols; j++)
            m->data[i * m->cols + j] = fmaxf(0, m->data[i * m->cols + j]);
}

// Function to compute derivative of ReLU
Matrix* relu_derivative(Matrix *m) {
    Matrix *derivative = allocate_matrix(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++)
        for(int j = 0; j < m->cols; j++)
            derivative->data[i * m->cols + j] = (m->data[i * m->cols + j] > 0) ? 1 : 0;
    return derivative;
}

// ! Loss Functions
// Function to compute Mean Squared Error
float mean_squared_error(Matrix *Y_pred, Matrix *Y_true) {
    float mse = 0.0f;
    for(int i = 0; i < Y_pred->rows; i++)
        for(int j = 0; j < Y_pred->cols; j++)
            mse += pow(Y_pred->data[i * Y_pred->cols + j] - Y_true->data[i * Y_true->cols + j], 2);
    return mse / Y_pred->rows;
}

// ! Optimization
// Function to update weights: W = W - learning_rate * grad
void update_weights(Matrix *W, Matrix *grad, float learning_rate) {
    for(int i = 0; i < W->rows; i++)
        for(int j = 0; j < W->cols; j++)
            W->data[i * W->cols + j] -= learning_rate * grad->data[i * grad->cols + j];
}

// Function to perform backpropagation and update weights
void backpropagation(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1, Matrix *Y_pred, Matrix *W1, Matrix *W2, int batch_size) {
    // Compute dZ2 = Y_pred - Y_batch
    Matrix *dZ2 = mat_sub(Y_pred, Y_batch);
    mat_scalar_mult(dZ2, 2.0f / batch_size);

    // Compute dW2 = Z1^T * dZ2
    Matrix *Z1_T = allocate_matrix(Z1->cols, Z1->rows);
    for(int i = 0; i < Z1->rows; i++) {
        for(int j = 0; j < Z1->cols; j++) {
            Z1_T->data[j * Z1->rows + i] = Z1->data[i * Z1->cols + j];
        }
    }
    Matrix *dW2 = mat_mult(Z1_T, dZ2);
    update_weights(W2, dW2, LEARNING_RATE);
    free_matrix(dW2);
    free_matrix(Z1_T);

    // Compute dZ1 = dZ2 * W2^T
    Matrix *W2_T = allocate_matrix(W2->cols, W2->rows);
    for(int i = 0; i < W2->rows; i++) {
        for(int j = 0; j < W2->cols; j++) {
            W2_T->data[j * W2->rows + i] = W2->data[i * W2->cols + j];
        }
    }
    Matrix *dZ1 = mat_mult(dZ2, W2_T);

    // Apply ReLU derivative
    Matrix *dZ1_derivative = relu_derivative(Z1);
    for(int i = 0; i < dZ1->rows; i++) {
        for(int j = 0; j < dZ1->cols; j++) {
            dZ1->data[i * dZ1->cols + j] *= dZ1_derivative->data[i * dZ1_derivative->cols + j];
        }
    }
    free_matrix(dZ1_derivative);
    free_matrix(W2_T);

    // Compute dW1 = X_batch^T * dZ1
    Matrix *X_batch_T = allocate_matrix(X_batch->cols, X_batch->rows);
    for(int i = 0; i < X_batch->rows; i++) {
        for(int j = 0; j < X_batch->cols; j++) {
            X_batch_T->data[j * X_batch->rows + i] = X_batch->data[i * X_batch->cols + j];
        }
    }
    Matrix *dW1 = mat_mult(X_batch_T, dZ1);
    update_weights(W1, dW1, LEARNING_RATE);
    free_matrix(dW1);
    free_matrix(X_batch_T);

    // Free allocated matrices
    free_matrix(dZ2);
    free_matrix(dZ1);
}

// ! Batch Processing
// Function to get a batch from the dataset
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch, int batch_start, int batch_size) {
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < INPUT_SIZE; j++)
            X_batch->data[i * INPUT_SIZE + j] = X->data[(batch_start + i) * INPUT_SIZE + j];
        Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
    }
}

// ! Data Loading
// Function to load CSV and populate X and Y, Assuming the last column is Y
int load_csv(const char *filename, Matrix **X, Matrix **Y, int *num_samples) {
    FILE *file = fopen(filename, "r");
    if(!file) {
        printf("Failed to open file.\n");
        return -1;
    }
    char line[1024];
    int count = 0;
    // First pass to count samples
    while(fgets(line, sizeof(line), file)) count++;
    *num_samples = count;
    rewind(file);
    // Allocate X and Y
    *X = allocate_matrix(count, INPUT_SIZE);
    *Y = allocate_matrix(count, OUTPUT_SIZE);
    int i = 0;
    while(fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        int j = 0;
        while(token) {
            if(j < INPUT_SIZE) {
                (*X)->data[i * INPUT_SIZE + j] = atof(token);
            } else {
                (*Y)->data[i * OUTPUT_SIZE] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(file);
    return 0;
}

// Main function
int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s <data.csv>\n", argv[0]);
        return -1;
    }

    double start_time, end_time;

    Matrix *X, *Y;
    int num_samples;
    if(load_csv(argv[1], &X, &Y, &num_samples) != 0)
        return -1;

    // Allocate and initialize weights
    Matrix *W1 = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix *W2 = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
    random_init(W1);
    random_init(W2);

    // Start measuring time
    start_time = omp_get_wtime();

    // Training loop
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        for(int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
            int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
            int batch_size = batch_end - batch_start;

            // Extract batch
            Matrix *X_batch = allocate_matrix(batch_size, INPUT_SIZE);
            Matrix *Y_batch = allocate_matrix(batch_size, OUTPUT_SIZE);
            get_batch(X, Y, X_batch, Y_batch, batch_start, batch_size);

            // Forward pass: X -> Hidden Layer -> ReLU -> Output Layer
            Matrix *Z1 = mat_mult(X_batch, W1);

            // // print both matrices to validate correctness
            // printf("Matrix X:\n");
            // for (int i = 0; i < X_batch->rows; i++) {
            //     for (int j = 0; j < X_batch->cols; j++) {
            //         printf("%.2lf ", X_batch->data[i * X_batch->cols + j]);
            //     }
            //     printf("\n");
            // }

            // printf("Matrix W1:\n");
            // for (int i = 0; i < W1->rows; i++) {
            //     for (int j = 0; j < W1->cols; j++) {
            //         printf("%.2lf ", W1->data[i * W1->cols + j]);
            //     }
            //     printf("\n");
            // }

            // printf("Matrix Z1:\n");
            // for (int i = 0; i < Z1->rows; i++) {
            //     for (int j = 0; j < Z1->cols; j++) {
            //         printf("%.2lf ", Z1->data[i * Z1->cols + j]);
            //     }
            //     printf("\n");
            // }

            // exit(0);

            relu(Z1);
            Matrix *Y_pred = mat_mult(Z1, W2);

            // Compute loss
            float loss = mean_squared_error(Y_pred, Y_batch);
            if((batch_start == 0) && ((epoch % LOG_EVERY_EPOCH == 0 && epoch != 0) || epoch == 1 || epoch == EPOCHS - 1))
                printf("Epoch %d, MSE: %f\n", epoch, loss);

            // Backward pass
            backpropagation(X_batch, Y_batch, Z1, Y_pred, W1, W2, batch_size);

            // Free allocated matrices
            free_matrix(Z1);
            free_matrix(Y_pred);
            free_matrix(X_batch);
            free_matrix(Y_batch);
        }
    }

    // Stop measuring time
    end_time = omp_get_wtime();

    printf("Training time: %.4f seconds\n", end_time - start_time);

    // Cleanup
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(X);
    free_matrix(Y);

    return 0;
}
