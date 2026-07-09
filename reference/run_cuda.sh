nvcc -o nn_cuda nn_cuda.cu -Xcompiler -fopenmp

# ./nn_cuda ./data/synthetic_convex_small.csv
# ./nn_cuda ./data/synthetic_convex_medium.csv
./nn_cuda ./data/synthetic_convex_large.csv