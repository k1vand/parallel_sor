#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <malloc.h>
#include <memory.h>
#include <unistd.h>
#include <stdlib.h>
#include <float.h>


#define PARAM_ABS_MAX 100
#define ITERATIONS_MAX 10000

struct global_ctx_s {
    atomic_int i;
    int **A;
    uint64_t n;
    int *b;
    _Atomic double *e;
    _Atomic double *X;
    atomic_uint *Xi;
    double max_e;
    double w;

    uint32_t threads_num;
    pthread_t *threads;

    struct tctx_s *tctxs
};

struct  tctx_s
{
    struct global_ctx_s *gctx;
    uint32_t idx;
};

inline int next_X(int i)
{
    return i % 2;
}
inline int prev_X(int i)
{
    return (i + 1) % 2;
}

void *worker(struct tctx_s * tctx) {
    struct global_ctx_s *gctx = tctx->gctx;
    int *b = gctx->b;
    int **A = gctx->A;
    _Atomic double *X = gctx->X;
    

    int own_rows_num = gctx->n / gctx->threads_num +  (tctx->idx + 1 <= gctx->n % gctx->threads_num ? 1 : 0);
    
    printf("Workder %d start, own_rows_num: %d\n", tctx->idx, own_rows_num);
    int gi  = atomic_load(&gctx->i);
    while(gi > 0) {
        for (int row_i = 0; row_i < own_rows_num; row_i++)
        {
            int row = tctx->idx + gctx->threads_num * row_i;
            
            double old_X = X[row];
            double old_part = (1 - gctx->w) * old_X;
            
            double new_part = b[row];
            for (int i = gctx->n; i > row; i--)
            {
                new_part -= A[row][i]*X[i];    
            }

            for (int i = row - 1; i >= 0; i--)
            {
                while (atomic_load(&gctx->Xi[i]) != gi)
                {
                    sleep(0);
                }
                new_part -= A[row][i] * X[i];
            }
            
            new_part = gctx->w * (new_part / (double) A[row][row]);
            atomic_store(&X[row], old_part + new_part);
            atomic_store(&gctx->e[row], abs(old_X - X[row]));

            atomic_fetch_add(&gctx->Xi[row], 1);
        }

        while (atomic_load(&gctx->i) == gi)
        {
            sleep(0);
        }
        gi  = atomic_load(&gctx->i);
    }

    printf("Worker %d is finished\n", tctx->idx);
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
        gctx->A[i][i] =  rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;
        new_max = (uint32_t) (abs(gctx->A[i][i]) / (double) (gctx->n - 1) - 1);

        for (int j = 0; j < gctx->n; j++)
        {
            if( i != j ) {
                gctx->A[i][j] =  rand() % (2 * new_max + 1) - new_max;
            }
        }
    }
}

void init_gctx(struct global_ctx_s *gctx)
{
    gctx->threads = calloc(sizeof(*gctx->threads), gctx->threads_num);
    memset(gctx->threads, 0, sizeof(*gctx->threads) * gctx->threads_num);

    gctx->tctxs =  calloc(sizeof(*gctx->tctxs), gctx->threads_num);
    memset(gctx->tctxs, 0, sizeof(*gctx->tctxs) * gctx->threads_num);

    gctx->A = calloc(sizeof(*gctx->A), gctx->n);
    for (int i = 0; i < gctx->n; i++)
    {
        gctx->A[i]= calloc(sizeof(*gctx->A[i]), gctx->n);
        memset(gctx->A[i], 0, sizeof(*gctx->A[i]) * gctx->n);
    }
    
    gctx->X = calloc(sizeof(*gctx->X), gctx->n);
    memset(gctx->X, 0, sizeof(*gctx->X) * gctx->n);
    
    gctx->b = calloc(sizeof(*gctx->b), gctx->n);
    memset(gctx->b, 0, sizeof(*gctx->b) * gctx->n);

    gctx->e = calloc(sizeof(*gctx->e), gctx->n);
    memset(gctx->e, DBL_MAX, sizeof(*gctx->e) * gctx->n);

    gctx->Xi = calloc(sizeof(*gctx->Xi), gctx->n);
    memset(gctx->Xi, 0, sizeof(*gctx->Xi) * gctx->n);
}

int main() {
    struct global_ctx_s gctx = {
        .threads_num = 2,
        .n = 8,
        .max_e = 0.0000001,
        .i = 1,
        .w = 1.5
    };

    init_gctx(&gctx);
    populate_linear_system(&gctx);

    printf("Linear system n = %d: \n", gctx.n);
    for (int i = 0; i < gctx.n ; i++)
    {
        for (int j = 0; j < gctx.n; j++)
        {
            printf("%*d ", count_digits(PARAM_ABS_MAX), gctx.A[i][j]);
        }
        printf("%*d\n", count_digits(PARAM_ABS_MAX), gctx.b[i]);
    }
    printf("max e = %g, w = %g\n", gctx.max_e, gctx.w);


    for (int i = 0; i < gctx.threads_num; i++)
    {
        gctx.tctxs[i].gctx = &gctx; 
        gctx.tctxs[i].idx = i;
        gctx.tctxs[i].i = 0;

        pthread_create(&gctx.threads[i], NULL, worker, &gctx.tctxs[i]);
    }
    

    while (true)
    {
        double cur_max_e = DBL_MIN;
        for (int i = 0; i < gctx.n; i++)
        {
            while (atomic_load(&gctx.Xi[i]) != gctx.i)
            {
                sleep(0);
            }

            if(cur_max_e < atomic_load(&gctx.e[i]))
                cur_max_e = atomic_load(&gctx.e[i]);
        }

        if(cur_max_e < gctx.max_e) {
            printf("Get result for %d iterations, max e %g\n", gctx.i, cur_max_e);
            atomic_store(&gctx.i, -1);
            break;
        } else {
            printf("X: \n");
            for (int i = 0; i < gctx.n; i++)
            {
                printf("%.2f ", gctx.X[i]);
            }
            printf("\n");
            printf("%g\n", cur_max_e);
            atomic_fetch_add(&gctx.i, 1);
        }
    }
    
    
    printf("X: \n");
    for (int i = 0; i < gctx.n; i++)
    {
        printf("%.2f ", gctx.X[i]);
    }
    printf("\n");
    

    for (int i = 0; i < gctx.threads_num; i++)
    {
        pthread_join(gctx.threads[i], NULL);
    }
    

    return 0;
}
