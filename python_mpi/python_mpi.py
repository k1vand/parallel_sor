import os 
import argparse
import random
from mpi4py import MPI
import numpy as np 
import math

PARAM_ABS_MAX = 100
E_TAG = 1
GI_TAG = 2

class SORTask:    
    def populate_ab_from_file(self, path):
        try:
            with open(path, "r") as f:
                data = np.loadtxt(f, dtype=int)
                print(data[:].size)
                for i in range(0, self.n):
                    for j in range(0, self.n):
                        self.A[i][j] = data[i, j]
                    self.b[i] = data[i, -1]
        except FileNotFoundError:
            print("Failed to open file")
            return -1

        return 0
        
    def populate_ab(self):
        random.seed(1234567890)

        for i in range(self.n):
            self.b[i] = random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
            while self.A[i][i] == 0:
                self.A[i][i] = random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)

            new_max = int(abs(self.A[i][i]) / (self.n - 1) - 1) 

            for j in range(0, self.n):
                if i != j:
                    self.A[i][j] = random.randint(-new_max, new_max)
        return 0


    def __init__(self, size: int, n: int = 8, w: float = 1.5, max_e: float = 0.0001, ):
        self.n = n
        self.w = w
        self.max_e = max_e

        self.A = np.zeros((self.n, self.n), dtype=int)
        self.b = np.zeros(self.n, dtype=int) 
        self.X = np.zeros(self.n, dtype=float)
        self.e = np.zeros(self.n, dtype=float)

        self.i = np.ones(size, dtype=int)

def sor(t: SORTask):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  
    size = comm.Get_size()  

    own_rows_num = t.n // size + (1 if rank + 1 <= t.n % size else 0)

    print(f"Worker {rank} start, own rows num {own_rows_num}")

    while t.i[rank] > 0:
        print(f"[{rank}] iteration №{t.i[rank]}")
        for row_i in range(0, own_rows_num):
            row =  rank + size * row_i
            old_X = t.X[row]
            fpart = (1 - t.w) * old_X

            spart = t.b[row] 
            for i in range(row  + 1, t.n):
                spart -= t.A[row][i] * t.X[i]

            for i in range(0, row):
                if row - i < size:  
                    row_owner_rank =  i % size
                    if row_owner_rank != rank:
                        # print(f"[{rank}] Get X{i} from {row_owner_rank}")
                        comm.Bcast(t.X[i: i+1], row_owner_rank)

                spart -= t.A[row][i] * t.X[i]
            
            spart = t.w * (spart / t.A[row][row])
            t.X[row] = fpart + spart

            # print(f"[{rank}] Cast x{row}: {t.X[row]}")
            comm.Bcast(t.X[row:row + 1], rank)
            t.e[row] = abs(old_X - t.X[row])

            if rank != 0:
                # print(f"[{rank}] Send e{row}: {t.e[row]}")
                comm.Isend(t.e[row: row+1], 0, E_TAG)

            # Get the last X's that are not ours
            if row_i + 1 == own_rows_num:
                for i in range(row + 1, t.n):
                    row_owner = i % size
                    # print(f"[{rank}] Get last x{i} from {row_owner}")
                    comm.Bcast(t.X[i: i+1], row_owner)
        
        if rank == 0:
            for i in range(0, t.n):
                row_owner = i % size 
                if row_owner != rank:
                    comm.Irecv(t.e[i: i+1], row_owner, E_TAG)
            cur_max_e = max(t.e)

            if cur_max_e <= t.max_e:
                print(f"Get result for {t.i[rank]} iterations, max e {cur_max_e}")
                t.i[rank] = -1
            else:
                # print(t.X)
                # print(cur_max_e)

                t.i[rank] += 1
            
            # for i in range(1, size):
            #     comm.Send(t.i, i, GI_TAG)
        
        comm.Bcast(t.i[rank: rank + 1], 0)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  
    size = comm.Get_size()  

    

    parser = argparse.ArgumentParser(description="Process linear system parameters.")
    parser.add_argument("-c", type=str, help="File with linear system", required=False)
    parser.add_argument("-o", type=str, help="Output file for solution", required=False)
    parser.add_argument("-n", type=int, help="Matrix size", required=False)
    parser.add_argument("-w", type=float, help="Relaxation factor", required=False)
    parser.add_argument("-e", type=float, help="Tolerance", required=False)
    
    args = parser.parse_args()
    
    n = 8
    w = 1.5
    max_e = 0.0001    
    if args.n is not None:
        n = args.n
    if args.w is not None:
        w = args.w
    if args.e is not None:
        max_e = args.e

    task = SORTask(size=size, n=n, w=w, max_e=max_e)

    linear_system_path = args.c
    linear_system_solve_path = args.o

    if rank == 0:
        if linear_system_path is not None:
            task.populate_ab_from_file(linear_system_path)
        else:
            task.populate_ab()

    comm.Bcast((task.A, MPI.INT), 0)
    comm.Bcast((task.b, MPI.INT), 0)

    if rank == 0:
        print(task.A)
        print(task.b)

    sor(task)

    if rank == 0:
        print(task.X)
        
        if linear_system_solve_path is not None: 
            try:
                with open(linear_system_solve_path, "w") as f:
                    precision = abs(math.log10(task.max_e))
                    f.write(" ".join(f"{x:.{int(precision)}f}" for x in task.X) + "\n")
            except FileNotFoundError:
                print("Failed to open linear_system_solve_path")

if __name__ == "__main__":
    main()