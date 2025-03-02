import os
import numpy as np
import random
import time

PARAM_ABS_MAX = 10000

def gen_linear_three_diagonal_system(n: int ):
    A = np.zeros((n, n), dtype=int)
    b = np.zeros((n), dtype=int)
    
    for i in range(0, n):
        b[i] = np.random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
        
        A[i, i] = PARAM_ABS_MAX
        A[i, i - 1] = (PARAM_ABS_MAX - 1) / -2 
        if i + 1 < n:
            A[i, i + 1] = (PARAM_ABS_MAX - 1) / -2 
    
    return (A, b)
    

def gen_linear_system(n: int):
    random.seed(time.time())
    np.random.seed(int(time.time()))

    A = np.zeros((n, n), dtype=int)
    b = np.zeros((n), dtype=int)
    
    for i in range(0, n):
        b[i] = np.random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
        while A[i,i] == 0:
            A[i, i] = np.random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
        new_max = int(abs(A[i, i]) / (n - 1) - 1)

        for j in range(n):
            if i != j:
                A[i, j] = random.randint(-new_max, new_max)  
    
    return (A, b)

def write_linear_system_to_file(A, b, linsys_path, prefix=""):
    Ab = np.hstack((A, b.reshape(-1, 1)))
    with open(linsys_path, "w") as f:
        for j in range(0, b.size):
            f.write(prefix + (" ".join(map(str, Ab[j].tolist()))) + "\n")


def main():
    n = 200
    A, b = gen_linear_three_diagonal_system(n)
    write_linear_system_to_file(A,b, f"./linsys/{n}.txt")

if __name__ == "__main__":
    main()