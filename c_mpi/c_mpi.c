#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <malloc.h>
#include <memory.h>
#include <unistd.h>
#include <stdlib.h>
#include <float.h>
#include <errno.h>
#include <mpi.h>

#define PARAM_ABS_MAX 100
#define ITERATIONS_MAX 10000

#define MPI_SEND_E 1

struct global_ctx_s {
    int32_t i;
    int *A;
    uint32_t n;
    int *b;
    double *e;
    double *X;
    double max_e;
    double w;

    int workers_num;

    MPI_Comm MPI_COMM_SPLITTED;
};

inline int next_X(int i)
{
    return i % 2;
}
inline int prev_X(int i)
{
    return (i + 1) % 2;
}

void worker(struct global_ctx_s * gctx) {
    int *b = gctx->b;
    int *A = gctx->A;
    double *X = gctx->X;
    int idx;
    int global_rank;

    MPI_Comm_rank(gctx->MPI_COMM_SPLITTED, &idx);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    int own_rows_num = gctx->n / gctx->workers_num +  (idx + 1 <= gctx->n % gctx->workers_num ? 1 : 0);
    
    printf("Workder %d/%d start, own_rows_num: %d\n", idx + 1, gctx->workers_num, own_rows_num);
    int32_t gi = gctx->i;
    while(gi > 0) {
        for (int row_i = 0; row_i < own_rows_num; row_i++)
        {
            int row = idx + gctx->workers_num * row_i;
            
            double old_X = X[row];
            double old_part = (1 - gctx->w) * old_X;
            
            double new_part = b[row];
            for (int i = gctx->n - 1; i > row; i--)
            {
                new_part -= A[row * gctx->n + i] * X[i];    
            }
            

            for (int i = row - 1; i >= 0; i--)
            {
                int root_idx = i % gctx->workers_num + 1;
                MPI_Bcast(&X[i], 1, MPI_DOUBLE, root_idx, MPI_COMM_WORLD);
                new_part -= A[row * gctx->n + i] * X[i];
            }

            new_part = gctx->w * (new_part / (double) A[row * gctx->n + row]);
            X[row] = old_part + new_part;
            MPI_Bcast(&X[row], 1, MPI_DOUBLE, idx + 1, MPI_COMM_WORLD);

            gctx->e[row] = abs(old_X - X[row]);
            printf("I am is %d, sending e %g\n", global_rank, gctx->e[row]);
            MPI_Send(&gctx->e[row], 1, MPI_DOUBLE, 0, row, MPI_COMM_WORLD);
        }

        MPI_Bcast(&gi, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    }

    printf("Worker %d is finished\n", idx);
}

int count_digits(int n) {
    if (n == 0) 
        return 1;
    
    return (int) log10(abs(n)) + 1;
}

void populate_linear_system(struct global_ctx_s *gctx) {
    uint32_t new_max;
    
    // srand(time(NULL));
    srand(1234567890);
    for (int i = 0; i < gctx->n; i++)
    {    
        gctx->b[i]=  rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;
        gctx->A[i * gctx->n + i] =  rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;
        new_max = (uint32_t) (abs(gctx->A[i * gctx->n + i]) / (double) (gctx->n - 1) - 1);

        for (int j = 0; j < gctx->n; j++)
        {
            if( i != j ) {
                gctx->A[i * gctx->n + j] =  rand() % (2 * new_max + 1) - new_max;
            }
        }
    }
}

int init_gctx(struct global_ctx_s *gctx)
{
    gctx->A = malloc(sizeof(*gctx->A) * gctx->n * gctx->n);
    if(gctx->A == NULL)
        return -ENOMEM;
    memset(gctx->A, 0, sizeof(*gctx->A) * gctx->n * gctx->n);
        
    gctx->X = calloc(sizeof(*gctx->X), gctx->n);
    if(gctx->X == NULL)
        return -ENOMEM;
    memset(gctx->X, 0, sizeof(*gctx->X) * gctx->n);
    
    gctx->b = calloc(sizeof(*gctx->b), gctx->n);
    if(gctx->b == NULL)
        return -ENOMEM;
    memset(gctx->b, 0, sizeof(*gctx->b) * gctx->n);

    gctx->e = calloc(sizeof(*gctx->e), gctx->n);
    if(gctx->e == NULL)
        return -ENOMEM;
    for (int i = 0; i < gctx->n; i++)
    {
        gctx->e[i] = DBL_MAX;
    }
}

int main(int argc, char **argv) {
    int ret;
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct global_ctx_s gctx = {
        .n = 4,
        .max_e = 0.0000001,
        .i = 1,
        .w = 1.5
    };
    
    MPI_Comm_split(MPI_COMM_WORLD, (rank == 0) ? 0 : 1, rank, &gctx.MPI_COMM_SPLITTED);
    gctx.workers_num = size - 1;

    ret = init_gctx(&gctx);
    if (ret < 0)
        return ret; 
    

    if (rank == 0) { 
        populate_linear_system(&gctx);
    }

    MPI_Bcast(gctx.A, gctx.n * gctx.n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gctx.b, gctx.n, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        printf("Linear system n = %d: \n", gctx.n);
        for (int i = 0; i < gctx.n ; i++)
        {
            for (int j = 0; j < gctx.n; j++)
            {
                printf("%*d ", count_digits(PARAM_ABS_MAX), gctx.A[i * gctx.n + j]);
            }
            printf("%*d\n", count_digits(PARAM_ABS_MAX), gctx.b[i]);
        }
        printf("max e = %g, w = %g\n", gctx.max_e, gctx.w);
    }

    if (rank != 0)
    {
        worker(&gctx);
    } else {
        while (true)
        {
            
            double cur_max_e = DBL_MIN;
            
            for (int i = 0; i < gctx.n; i++)
            {
                MPI_Bcast(&gctx.X[i], 1, MPI_DOUBLE, i % gctx.workers_num + 1, MPI_COMM_WORLD);
                printf("Get new X: %g\n", gctx.X[i]);

                MPI_Status status;
                MPI_Recv(&gctx.e[i], 1, MPI_DOUBLE, i % gctx.workers_num + 1, i, MPI_COMM_WORLD, &status);
                printf("Get an sweet e %g\n", gctx.e[i]);
                
                if(cur_max_e < gctx.e[i])
                        cur_max_e = gctx.e[i];

                // if (status.MPI_ERROR == MPI_SUCCESS)
                    
                // else { 
                //     perror("Manager faild to get e daylik\n");
                //     return status.MPI_ERROR;  
                // }  
            }
            
            if(cur_max_e < gctx.max_e) {
                printf("Get result for %d iterations, max e %g\n", gctx.i, cur_max_e);
                gctx.i = -1;
                MPI_Bcast(&gctx.i, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
                break;
            } else {
                printf("X: \n");
                for (int i = 0; i < gctx.n; i++)
                {
                    printf("%.2f ", gctx.X[i]);
                }
                printf("\n");
                printf("%g\n", cur_max_e);
                
                gctx.i += 1;
                MPI_Bcast(&gctx.i, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
            }
        }
        
        
        printf("X: \n");
        for (int i = 0; i < gctx.n; i++)
        {
            printf("%.2f ", gctx.X[i]);
        }
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}