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
#include <omp.h>

#define PARAM_ABS_MAX 100
#define ITERATIONS_MAX 10000

struct global_ctx_s {
    int i;
    bool run;
    int **A;
    uint64_t n;
    int *b;
    double *e;
    double *X;
    uint32_t *Xi;
    double max_e;
    double w;

    uint32_t threads_num;
};


void sor(struct global_ctx_s *gctx) {
    int *b = gctx->b;
    int **A = gctx->A;
    double *X = gctx->X;
    
    #pragma omp parallel num_threads(gctx->threads_num) shared(b, A, X, gctx)
    { 
        int gi;
        int idx = omp_get_thread_num();  

        int own_rows_num = gctx->n / gctx->threads_num +  (idx + 1 <= gctx->n % gctx->threads_num ? 1 : 0);
        
        printf("Thread %d start, own_rows_num: %d\n", idx, own_rows_num);
        #pragma omp atomic read
            gi = gctx->i;
        while(gi > 0) {
            for (int row_i = 0; row_i < own_rows_num; row_i++)
            {
                int row = idx + gctx->threads_num * row_i;
                
                double old_X = X[row];
                double fpart = (1 - gctx->w) * old_X;
                
                double spart = b[row];
                for (int i = gctx->n - 1; i > row; i--)
                {
                    spart -= A[row][i]*X[i];    
                }

                for (int i = row - 1; i >= 0; i--)
                {
                    int xii;
                    do {
                        #pragma omp atomic read
                            xii = gctx->Xi[i];
                        sleep(0);
                    } while (xii != gi);
                    
                    double xi;
                    #pragma omp atomic read
                    xi = X[i];
                    spart -= A[row][i] * xi;
                }
                
                spart = gctx->w * (spart / (double) A[row][row]);
                
                #pragma omp atomic write
                X[row] = fpart + spart;
                #pragma omp atomic
                gctx->Xi[row] += 1;

                gctx->e[row] = fabs(old_X - X[row]);
            }
            
            #pragma omp barrier
            #pragma omp single copyprivate(gi)
            {
                double cur_max_e = 0;
                for (int i = 0; i < gctx->n; i++)
                {
                    if(cur_max_e < gctx->e[i])
                        cur_max_e = gctx->e[i];
                }

                if(cur_max_e < gctx->max_e || gctx->i >= ITERATIONS_MAX) {
                    printf("Get result for %d iterations, max e %g\n", gctx->i, cur_max_e);
                    gctx->i = - 1;
                } else {
                    // printf("X: \n");
                    // for (int i = 0; i < gctx->n; i++)
                    // {
                    //     printf("%.2f ", gctx->X[i]);
                    // }
                    // printf("\n");
                    // printf("%g\n", cur_max_e);

                    gctx->i++;
                }

                gi = gctx->i;
            }
        }

        printf("Worker %d is finished\n", idx);
    }
}

int count_digits(int n) {
    if (n == 0) 
        return 1;
    
    return (int) log10(abs(n)) + 1;
}

int populate_ab_from_file(struct global_ctx_s *gctx, char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        perror("Filed to open file");
        return -1;
    }

    for (int i = 0; i < gctx->n; i++) {
        for (int j = 0; j < gctx->n; j++) {
            fscanf(f, "%d", &gctx->A[i][j]);
        }
        fscanf(f, "%d", &gctx->b[i]);
    }
    fclose(f);

    return 0;
}

int populate_ab(struct global_ctx_s *gctx) {
    uint32_t new_max;

    // srand(time(NULL));
    srand(1234567890);
    for (int i = 0; i < gctx->n; i++) {
        gctx->b[i] = rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;
        while (gctx->A[i][i] == 0)
            gctx->A[i][i] = rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;

        new_max = (uint32_t) (abs(gctx->A[i][i]) / (double) (gctx->n - 1) - 1);

        for (int j = 0; j < gctx->n; j++) {
            if (i != j) {
                gctx->A[i][j] = rand() % (2 * new_max + 1) - new_max;
            }
        }
    }

    return 0;
}

void init_gctx(struct global_ctx_s *gctx)
{
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

int main(int argc, char *argv[]) {
    int ret, opt;
    char *linear_system_path = NULL;
    char *linear_system_solve_path = NULL;
    struct global_ctx_s gctx = {
        .threads_num = 4,
        .n = 8,
        .max_e = 0.0000001,
        .i = 1,
        .w = 1.5
    };

    while ((opt = getopt(argc, argv, "hc:o:t:n:w:e:")) != -1) {
        switch (opt) {
            case 'h':
                printf(
                    "-c - file with linear system, -t - threads, -n - matrix "
                    "size, -w - relax, -e - toler\n");
                return 0;
            case 'c':
                linear_system_path = optarg;
                break;
            case 'o':
                linear_system_solve_path = optarg;
                break;
            case 't':
                gctx.threads_num = atoi(optarg);
                break;
            case 'n':
                gctx.n = atoi(optarg);
                break;
            case 'w':
                gctx.w = atof(optarg);
                break;
            case 'e':
                gctx.max_e = atof(optarg);
                break;

            default:
                return 1;
        }
    }

    omp_set_num_threads(gctx.threads_num);
    omp_set_dynamic(0);

    init_gctx(&gctx);

    if (linear_system_path != NULL)
        ret = populate_ab_from_file(&gctx, linear_system_path);
    else
        ret = populate_ab(&gctx);

    if (ret != 0) {
        perror("Failed to populate linear system\n");
        return ret;
    }

    // printf("Linear system n = %d: \n", gctx.n);
    // for (int i = 0; i < gctx.n ; i++)
    // {
    //     for (int j = 0; j < gctx.n; j++)
    //     {
    //         printf("%*d ", count_digits(PARAM_ABS_MAX), gctx.A[i][j]);
    //     }
    //     printf("%*d\n", count_digits(PARAM_ABS_MAX), gctx.b[i]);
    // }
    // printf("max e = %g, w = %g\n", gctx.max_e, gctx.w);


    sor(&gctx);
    
    
    // printf("X: \n");
    // for (int i = 0; i < gctx.n; i++)
    // {
    //     printf("%.2f ", gctx.X[i]);
    // }
    // printf("\n");

    if (linear_system_solve_path != NULL) {
        FILE *f = fopen(linear_system_solve_path, "w");
        if (f == NULL) {
            perror("Filed to open linear_system_solve_path");
            return -1;
        }

        for (int i = 0; i < gctx.n; i++) {
            fprintf(f, "%.*f ", abs(log10(gctx.max_e)), gctx.X[i]);
        }
        fclose(f);
    }
    

    return 0;
}
