import os
import subprocess
import random
import numpy as np
from string import Template
from typing import Dict, Union, List
import math
import csv
import time

from gen_linear_system import gen_linear_three_diagonal_system


w = 1.5
e = 0.000000000001
ns = [(2000, 0.00001)]  # A sizes
instance_nums = [1, 2, 3, 4, 6, 8]  # threads / procs num


def module_path():
    return os.path.dirname(os.path.realpath(__file__))


algs_paths = {
    "c_pthread": os.path.join(module_path(), "c_pthreads", "build", "sor"),
    "c_omp": os.path.join(module_path(), "c_omp", "build", "sor"),
    "c_mpi": os.path.join(module_path(), "c_mpi", "build", "sor"),
    "python_mpi": os.path.join(module_path(), "python_mpi", "python_mpi.py"),
}

algs: List[Dict[str, Union[str, Template]]] = [
    {
        "name": "c_pthreads",
        "cmd": Template(
            f"{algs_paths['c_pthread']} -c $linsys_path -o $out_path -n $n -t $t -e $e -w $w"
        ),
    },
    {
        "name": "c_omp",
        "cmd": Template(
            f"{algs_paths['c_omp']} -c $linsys_path -o $out_path -n $n -t $t -e $e -w $w"
        ),
    },
    {
        "name": "c_mpi",
        "cmd": Template(
            f"mpirun --use-hwthread-cpus -np $t {algs_paths['c_mpi']} -c $linsys_path -o $out_path -n $n -e $e -w $w"
        ),
    },
    {
        "name": "python_mpi",
        "cmd": Template(
            f"mpirun --use-hwthread-cpus -np $t /usr/bin/python3 {algs_paths['python_mpi']} -c $linsys_path -o $out_path -n $n -e $e -w $w"
        ),
    },
]


def main():
    result = list()
    linsys_dir = os.path.join(module_path(), "linsys")
    outs_dir = os.path.join(module_path(), "outs")
    test_out = os.path.join(module_path(), "test_out.csv")
    csv_writer = csv.DictWriter(
        open(test_out, "w"),
        fieldnames=["alg", "instances", "n", "elapsed_ms", "l2_norm", "relative_error"],
    )
    csv_writer.writeheader()

    os.makedirs(linsys_dir, exist_ok=True)
    os.makedirs(outs_dir, exist_ok=True)

    linsys = list()
    for i in range(0, len(ns)):
        n, rel = ns[i]
        A, b = gen_linear_three_diagonal_system(n, rel)
        sol = np.linalg.solve(A, b)
        linsys.append((A, b, np.round(sol, decimals=int(abs(math.log10(e))))))

        b = b.reshape(-1, 1)
        Ab = np.hstack((A, b))
        linsys_path = os.path.join(linsys_dir, f"{n}.txt")
        with open(linsys_path, "w") as f:
            for j in range(0, n):
                f.write(" ".join(map(str, Ab[j].tolist())) + "\n")

        print(f"A|b = \n{Ab}")
        print(f"n = {n}\n")
        for instance_num in instance_nums:
            if instance_num > n:
                continue
            print(f"instance num = {instance_num}\n")

            for alg in algs:
                out_path = os.path.join(
                    outs_dir, f"{alg['name']}_{n}_{instance_num}.txt"
                )
                cmd = alg["cmd"].substitute(
                    linsys_path=linsys_path,
                    out_path=out_path,
                    n=n,
                    t=instance_num,
                    e=e,
                    w=w,
                )

                print(f"Alg: {alg['name']}\n{cmd}")
                start_timestamp = time.time()
                result = subprocess.run(
                    map(str, cmd.split()), stdout=subprocess.DEVNULL
                )
                elapsed_ms = (time.time() - start_timestamp) * 1000
                if result.returncode == 0:
                    with open(out_path, "r") as f:
                        solution = [float(x) for x in f.readline().split()]
                        # print(f"{os.path.basename(out_path)}, solution \n{solution}")
                        # print(f"true solution \n{linsys[i][2].tolist()}")

                        l2_norm = np.linalg.norm(solution - linsys[i][2], ord=2)
                        relative_error = l2_norm / np.linalg.norm(linsys[i][2], ord=2)

                        print(
                            f"l2_norm = {l2_norm}, relative_error = {relative_error}\n"
                        )

                        csv_writer.writerow(
                            {
                                "alg": alg["name"],
                                "instances": instance_num,
                                "n": n,
                                "elapsed_ms": elapsed_ms,
                                "l2_norm": l2_norm,
                                "relative_error": relative_error,
                            }
                        )


if __name__ == "__main__":
    main()
