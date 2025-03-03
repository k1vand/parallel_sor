import os
import numpy as np
import random
import time

PARAM_ABS_MAX = 1000000

def gen_linear_three_diagonal_system(n: int, rel: float ):
    A = np.zeros((n, n), dtype=int)
    b = np.zeros((n), dtype=int)
    
    for i in range(0, n):
        b[i] = np.random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
        
        A[i, i] = PARAM_ABS_MAX
        nd = (rel * A[i,i]) // 2
        if i != 0:
            A[i, i - 1] = -nd
        if i + 1 < n:
            A[i, i + 1] = -nd 
    
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
    n = 2000
    A, b = gen_linear_three_diagonal_system(n, 0.00001)
    write_linear_system_to_file(A,b, f"./linsys/{n}.txt")

    n = 100
    A, b = gen_linear_three_diagonal_system(n, 0.99999)
    write_linear_system_to_file(A,b, f"./linsys/{n}.txt")

if __name__ == "__main__":
    main()