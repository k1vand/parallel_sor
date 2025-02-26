import os 
import functools
import random
from mpi4py import MPI
import numpy as np 

PARAM_ABS_MAX = 100
E_TAG = 1

class SORTask:

    def populate_linear_system(self):
        random.seed(1234567890)

        for i in range(self.n):
            self.b[i] = random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)
            self.A[i][i] = random.randint(-PARAM_ABS_MAX, PARAM_ABS_MAX)

            new_max = int(abs(self.A[i][i]) / (self.n - 1) - 1) 

            for j in range(0, self.n):
                if i != j:
                    self.A[i][j] = random.randint(-new_max, new_max)


    def __init__(self,n: int = 8, w: float = 1.5, max_e: float = 0.0001):
        self.n = n
        self.w = w
        self.max_e = max_e

        self.A = np.zeros((self.n, self.n), dtype=int)
        self.b = np.zeros(self.n, dtype=int) 
        self.X = np.zeros(self.n, dtype=float)
        self.e = np.zeros(self.n, dtype=float)

        self.i = 1

def sor(t: SORTask):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  
    size = comm.Get_size()  

    own_rows_num = t.n // size + (1 if rank + 1 <= t.n % size else 0)

    print(f"Worker {rank} start, own rows num {own_rows_num}")

    while t.i > 0:
        for row_i in range(0, own_rows_num):
            row =  rank + size * row_i
            old_X = t.X[row]
            fpart = (1 - t.w) * old_X

            spart = t.b[row] 
            for i in range(row  + 1, t.n):
                spart -= t.A[row][i] * t.X[i]

            for i in range(0, row - 1):
                row_owner_rank =  i % size
                if row_owner_rank != rank:
                    comm.Bcast(t.X[i: i+1], row_owner_rank)
                
                spart -= t.A[row][i] * t.X[i]
            
            spart = t.w * (spart / t.A[row][row])
            t.X[row] = fpart + spart

            comm.Bcast(t.X[row:row + 1], rank)
            print(f"Cast x{row}: {t.X[row]}")
            t.e[row] = abs(old_X - t.X[row])

            if rank != 0:
                comm.Isend(t.e[row: row+1], 0, E_TAG)
                print(f"Send e{row}: {t.e[row]}")

        last_row_owner = (t.n - 1) % size
        if rank != last_row_owner:
            comm.Bcast(t.X[t.n - 1: t.n], last_row_owner)
        
        if rank == 0:
            for i in range(0, t.n):
                row_owner = i % size 
                if row_owner != rank:
                    comm.Irecv(t.e[i: i+1], row_owner, E_TAG)
            cur_max_e = max(t.e)

            if cur_max_e <= t.max_e:
                print(f"Get result for {t.i} iterations, max e {cur_max_e}")
                t.i = -1
            else:
                print(t.X)
                print(cur_max_e)

                t.i += 1
        comm.Bcast(t.i, 0)

        




def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  
    size = comm.Get_size()  

    task = SORTask()

    if rank == 0:
        task.populate_linear_system()

    comm.Bcast((task.A, MPI.INT), 0)
    comm.Bcast((task.b, MPI.INT), 0)

    if rank == 0:
        print(task.A)
        print(task.b)

    sor(task)

if __name__ == "__main__":
    main()