gcc -o nn_pthreads nn_pthreads.c -lm -pthread -pg -fopenmp

# ./nn_pthreads ./data/synthetic_convex_small.csv
# ./nn_pthreads ./data/synthetic_convex_medium.csv
./nn_pthreads ./data/synthetic_convex_large.csv