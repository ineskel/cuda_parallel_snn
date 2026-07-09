gcc -o nn nn.c -lm -pg -fopenmp

# ./nn ./data/synthetic_convex_small.csv
# ./nn ./data/synthetic_convex_medium.csv
./nn ./data/synthetic_convex_large.csv

# ./nn ./data/synthetic_convex_large.csv > log/log.txt