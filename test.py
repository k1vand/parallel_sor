import os
import subprocess
import random
import numpy as np

PARAM_ABS_MAX = 100


w = 1.5
e = 0.0001
ns = [3, 6, 12] # A sizes
instance_nums = [1, 2, 4] # threads / procs num

def module_path():
    return os.path.dirname(os.path.realpath(__file__))

def gen_linear_system(n: int):
    A = np.zeros((n, n), dtype=int)
    b = np.zeros((n), dtype=int)
    
    for i in range(0, n):
        b[i] = np.random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
        A[i, i] = np.random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
        new_max = int(abs(A[i, i]) / (n - 1) - 1)

        for j in range(n):
            if i != j:
                A[i, j] = random.randint(-new_max, new_max)  
    
    return (A, b)
    

def main():
    linsys_dir = os.path.join(module_path(), "linsys")
    outs_dir = os.path.join(module_path(), "outs")
    
    algs_paths = {
        "c_pthread": os.path.join(module_path(), "c_pthreads", "build", "sor")
    }

    os.makedirs(linsys_dir, exist_ok=True)
    os.makedirs(outs_dir, exist_ok=True)

    np.random.seed(1234567890)
    linsys = list()
    for i in range (0, len(ns)):
        n = ns[i]
        A, b = gen_linear_system(n)
        linsys.append((A, b, np.linalg.solve(A, b)))
        
        b = b.reshape(-1, 1)
        Ab = np.hstack((A, b))
        linsys_path = os.path.join(linsys_dir, f"{n}.txt");
        with open(linsys_path, "w") as f:
            for j in range(0, n):
                f.write(" ".join(map(str, Ab[j].tolist())) + "\n")
        
        print(f"A|b = \n{Ab}")
        print(f"n = {n}")
        for instance_num in instance_nums:
            print(f"instance num = {instance_num}")

            out_path = os.path.join(outs_dir, f"c_pthreads_{n}_{instance_num}.txt")
            cmd = [algs_paths['c_pthread'], "-c", linsys_path, "-o", out_path, "-n", n, "-e", e, "-w", w]
            result = subprocess.run(map(str, cmd), stdout=subprocess.DEVNULL)
            
            if result.returncode == 0:
                with open(out_path, "r") as f:
                    solution = [float(x) for x in f.readline().split()]
                    print(f"my solution \n{solution}")
                    print(f"true solution \n{linsys[i][2].tolist()}")

                    l2_norm = np.linalg.norm(solution - linsys[i][2], ord=2)
                    relative_error = l2_norm / np.linalg.norm(linsys[i][2], ord=2)

                    print(f"l2_norm = {l2_norm}, relative_error = {relative_error}\n")


if __name__ == "__main__":
    main()