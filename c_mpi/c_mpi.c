#include <errno.h>
#include <float.h>
#include <malloc.h>
#include <math.h>
#include <memory.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define PARAM_ABS_MAX 100
#define ITERATIONS_MAX 1000

#define MPI_SEND_E_TAG 1000


struct global_ctx_s {
    int32_t i;
    int *A;
    uint32_t n;
    int *b;
    double *e;
    double *X;
    double max_e;
    double w;

    int run;
};

void _debug(const char *format, ...) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    va_list args;
    va_start(args, format);

    printf("[Rank %d] ", rank);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}

double sor(struct global_ctx_s *gctx, double *solution_e) {
    int ret = 0;
    int *b = gctx->b;
    int *A = gctx->A;
    double *X = gctx->X;
    MPI_Request e_req;
    int size, rank, own_rows_num;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    own_rows_num = gctx->n / size + (rank + 1 <= gctx->n % size ? 1 : 0);
    printf("Workder %d/%d start, own_rows_num: %d\n", rank + 1, size,
           own_rows_num);

    int row;
    double old_X, fpart, spart;
    while (gctx->run) {
        for (int row_i = 0; row_i < own_rows_num; row_i++) {
            row = rank + size * row_i;

            old_X = X[row];
            fpart = (1 - gctx->w) * old_X;

            spart = b[row];
            for (int i = row + 1; i < gctx->n; i++) {
                spart -= A[row * gctx->n + i] * X[i];
            }

            for (int i = 0; i < row; i++) {
                if (row - i < size) {
                    int row_owner_rank = i % size;
                    if (row_owner_rank != rank) {
                        MPI_Bcast(&gctx->X[i], 1, MPI_DOUBLE, row_owner_rank,
                                  MPI_COMM_WORLD);
                        // _debug("Get x%d %g from %d", i, gctx->X[i],
                        //        row_owner_rank);
                    }
                }

                spart -= A[row * gctx->n + i] * X[i];
            }

            spart = gctx->w * (spart / (double)A[row * gctx->n + row]);
            X[row] = fpart + spart;

            MPI_Bcast(&X[row], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);

            gctx->e[row] = fabs(old_X - X[row]);
            if (rank != 0) {
                MPI_Isend(&gctx->e[row], 1, MPI_DOUBLE, 0, MPI_SEND_E_TAG,
                          MPI_COMM_WORLD, &e_req);
            }

            if (row_i + 1 == own_rows_num) {
                for (int i = row + 1; i < gctx->n; i++) {
                    MPI_Bcast(&gctx->X[i], 1, MPI_DOUBLE, i % size,
                              MPI_COMM_WORLD);
                    // _debug("Get last x%d %g from %d", i, gctx->X[i], i % size);
                }
            }
        }

        if (rank == 0) {
            double cur_e_max = 0;
            for (int i = 0; i < gctx->n; i++) {
                int row_owner = i % size;
                if (row_owner != rank) {
                    MPI_Irecv(&gctx->e[i], 1, MPI_DOUBLE, row_owner,
                              MPI_SEND_E_TAG, MPI_COMM_WORLD, &e_req);
                    MPI_Wait(&e_req, MPI_STATUS_IGNORE);
                }

                if (cur_e_max < gctx->e[i]) cur_e_max = gctx->e[i];
            }

            if (gctx->i >= ITERATIONS_MAX) {
                ret = -1;
                gctx->run = 0;
            } else if (cur_e_max <= gctx->max_e) {
                gctx->run = 0;
                *solution_e = cur_e_max;
            } else {
                gctx->i++;
                // printf("X: \n");
                // for (int i = 0; i < gctx->n; i++) {
                //     printf("%.2f ", gctx->X[i]);
                // }
                // printf("\n");
            }
        }

        MPI_Bcast(&gctx->run, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // if (rank == 0) _debug("Iteration: %d", gctx->i);
    }

    printf("Worker %d is finished\n", rank);
    return ret;
}

int count_digits(int n) {
    if (n == 0) return 1;

    return (int)log10(abs(n)) + 1;
}

int populate_ab_from_file(struct global_ctx_s *gctx, char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        perror("Filed to open file");
        return -1;
    }

    for (int i = 0; i < gctx->n; i++) {
        for (int j = 0; j < gctx->n; j++) {
            fscanf(f, "%d", &gctx->A[i * gctx->n + j]);
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
        gctx->A[i * gctx->n + i] = 0;
        while (gctx->A[i * gctx->n + i] == 0)
            gctx->A[i * gctx->n + i] =
                rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;
        new_max =
            (uint32_t)(abs(gctx->A[i * gctx->n + i]) / (double)(gctx->n - 1) -
                       1);

        for (int j = 0; j < gctx->n; j++) {
            if (i != j) {
                gctx->A[i * gctx->n + j] = rand() % (2 * new_max + 1) - new_max;
            }
        }
    }

    return 0;
}

int init_gctx(struct global_ctx_s *gctx) {
    gctx->A = malloc(sizeof(*gctx->A) * gctx->n * gctx->n);
    if (gctx->A == NULL) return -ENOMEM;
    memset(gctx->A, 0, sizeof(*gctx->A) * gctx->n * gctx->n);

    gctx->X = calloc(sizeof(*gctx->X), gctx->n);
    if (gctx->X == NULL) return -ENOMEM;
    memset(gctx->X, 0, sizeof(*gctx->X) * gctx->n);

    gctx->b = calloc(sizeof(*gctx->b), gctx->n);
    if (gctx->b == NULL) return -ENOMEM;
    memset(gctx->b, 0, sizeof(*gctx->b) * gctx->n);

    gctx->e = calloc(sizeof(*gctx->e), gctx->n);
    if (gctx->e == NULL) return -ENOMEM;
    for (int i = 0; i < gctx->n; i++) {
        gctx->e[i] = DBL_MAX;
    }

    return 0;
}

int main(int argc, char **argv) {
    int ret, opt, rank, size;
    char *linear_system_path = NULL;
    char *linear_system_solve_path = NULL;
    double cur_max_e;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct global_ctx_s gctx = {
        .n = 8, .max_e = 0.0000001, .i = 1, .w = 1.5, .run = 1};
    
    while ((opt = getopt(argc, argv, "hc:o:n:w:e:")) != -1) {
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

    ret = init_gctx(&gctx);
    if (ret < 0) return ret;

    if (rank == 0) {
        if (linear_system_path != NULL)
            ret = populate_ab_from_file(&gctx, linear_system_path);
        else
            ret = populate_ab(&gctx);

        if (ret != 0) {
            perror("Failed to populate linear system\n");
            return ret;
        }
    }

    MPI_Bcast(gctx.A, gctx.n * gctx.n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gctx.b, gctx.n, MPI_INT, 0, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     printf("Linear system n = %d: \n", gctx.n);
    //     for (int i = 0; i < gctx.n; i++) {
    //         for (int j = 0; j < gctx.n; j++) {
    //             printf("%*d ", count_digits(PARAM_ABS_MAX),
    //                    gctx.A[i * gctx.n + j]);
    //         }
    //         printf("%*d\n", count_digits(PARAM_ABS_MAX), gctx.b[i]);
    //     }
    //     printf("max e = %g, w = %g\n", gctx.max_e, gctx.w);
    // }

    ret = sor(&gctx, &cur_max_e);
    if (rank == 0) {
        if (ret == 0) {
            printf("Get result for %d iterations, max e %g\n", gctx.i,
                   cur_max_e);
            // printf("X: \n");
            // for (int i = 0; i < gctx.n; i++) {
            //     printf("%.*f ", abs(log10(gctx.max_e)), gctx.X[i]);
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

        } else {
            printf("Reach limit of iterations: %d", ITERATIONS_MAX);
        }
    }

    MPI_Finalize();
    return 0;
}