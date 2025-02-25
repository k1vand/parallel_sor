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
#define ITERATIONS_MAX 10000

#define MPI_SEND_E_TAG 1000
#define MPI_SEND_X_TAG 2000

#define MPI_WAIT_TIMEOUT 5

#define MPI_WAIT_DEBUG(request, status, msg)                                  \
    do {                                                                      \
        time_t start_time = time(NULL);                                       \
        int flag = 0;                                                         \
        while (!flag) {                                                       \
            MPI_Test(request, &flag, status);                                 \
            if (time(NULL) - start_time > MPI_WAIT_TIMEOUT) {                 \
                int rank;                                                     \
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);                         \
                fprintf(stderr, "%s, MPI_Wait is hanging on rank %d!\n", msg, \
                        rank);                                                \
                break;                                                        \
            }                                                                 \
        }                                                                     \
        MPI_Wait(request, status);                                            \
    } while (0)

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

void worker(struct global_ctx_s *gctx) {
    int *b = gctx->b;
    int *A = gctx->A;
    double *X = gctx->X;
    MPI_Request *x_reqs;
    int size;
    int idx;
    int global_idx;

    MPI_Comm_rank(gctx->MPI_COMM_SPLITTED, &idx);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int own_rows_num = gctx->n / gctx->workers_num +
                       (idx + 1 <= gctx->n % gctx->workers_num ? 1 : 0);
    x_reqs = calloc(sizeof(*x_reqs), gctx->n);
    for (int i = 0; i < gctx->n; i++) {
        x_reqs[i] = MPI_REQUEST_NULL;
    }

    printf("Workder %d/%d start, own_rows_num: %d\n", idx + 1,
           gctx->workers_num, own_rows_num);
    while (gctx->i > 0) {
        for (int row_i = 0; row_i < own_rows_num; row_i++) {
            int row = idx + gctx->workers_num * row_i;

            double old_X = X[row];
            double fpart = (1 - gctx->w) * old_X;

            double spart = b[row];
            for (int i = gctx->n - 1; i > row; i--) {
                spart -= A[row * gctx->n + i] * X[i];
            }

            for (int i = row - 1; i >= 0; i--) {
                int source_idx = i % gctx->workers_num + 1;

                if (source_idx != global_idx) {
                    MPI_Status status;
                    MPI_Ibcast(&X[i], 1, MPI_DOUBLE, source_idx, MPI_COMM_WORLD,
                               &x_reqs[i]);

                    MPI_WAIT_DEBUG(&x_reqs[i], &status, "before fetch");
                    _debug("Fetch x%d %g from %d\n", i, X[i], source_idx);
                    x_reqs[i] = MPI_REQUEST_NULL;
                }
                spart -= A[row * gctx->n + i] * X[i];
            }

            spart = gctx->w * (spart / (double)A[row * gctx->n + row]);
            X[row] = fpart + spart;

            if (x_reqs[row] != MPI_REQUEST_NULL) {
                MPI_WAIT_DEBUG(&x_reqs[row], MPI_STATUS_IGNORE,
                               "on sended earlier");
                x_reqs[row] = MPI_REQUEST_NULL;
            }

            MPI_Ibcast(&X[row], 1, MPI_DOUBLE, global_idx, MPI_COMM_WORLD,
                       &x_reqs[row]);

            gctx->e[row] = fabs(old_X - X[row]);

            MPI_Send(&gctx->e[row], 1, MPI_DOUBLE, 0, row, MPI_COMM_WORLD);
        }

        int last_row_idx = (gctx->n - 1) % gctx->workers_num + 1;
        if (global_idx != last_row_idx) {
            MPI_Ibcast(&X[gctx->n - 1], 1, MPI_DOUBLE, last_row_idx,
                       MPI_COMM_WORLD, &x_reqs[gctx->n - 1]);
            MPI_WAIT_DEBUG(&x_reqs[gctx->n - 1], MPI_STATUS_IGNORE,
                           "wait for last X");
            _debug("Fetch x%d %g from %d\n", gctx->n - 1, X[gctx->n - 1], last_row_idx);
        }

        MPI_Request gi_req;
        MPI_Ibcast(&gctx->i, 1, MPI_INT32_T, 0, MPI_COMM_WORLD, &gi_req);
        MPI_WAIT_DEBUG(&gi_req, MPI_STATUS_IGNORE, "wait for new gi");

        _debug("Iteration: %d", gctx->i);
    }

    printf("Worker %d is finished\n", idx);
}

int count_digits(int n) {
    if (n == 0) return 1;

    return (int)log10(abs(n)) + 1;
}

void populate_linear_system(struct global_ctx_s *gctx) {
    uint32_t new_max;

    // srand(time(NULL));
    srand(1234567890);
    for (int i = 0; i < gctx->n; i++) {
        gctx->b[i] = rand() % (2 * PARAM_ABS_MAX + 1) - PARAM_ABS_MAX;
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
    int ret;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct global_ctx_s gctx = {.n = 4, .max_e = 0.0000001, .i = 1, .w = 1.5};

    MPI_Comm_split(MPI_COMM_WORLD, (rank == 0) ? 0 : 1, rank,
                   &gctx.MPI_COMM_SPLITTED);
    gctx.workers_num = size - 1;

    ret = init_gctx(&gctx);
    if (ret < 0) return ret;

    if (rank == 0) {
        populate_linear_system(&gctx);
    }

    MPI_Bcast(gctx.A, gctx.n * gctx.n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gctx.b, gctx.n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Linear system n = %d: \n", gctx.n);
        for (int i = 0; i < gctx.n; i++) {
            for (int j = 0; j < gctx.n; j++) {
                printf("%*d ", count_digits(PARAM_ABS_MAX),
                       gctx.A[i * gctx.n + j]);
            }
            printf("%*d\n", count_digits(PARAM_ABS_MAX), gctx.b[i]);
        }
        printf("max e = %g, w = %g\n", gctx.max_e, gctx.w);
    }

    if (rank != 0) {
        worker(&gctx);
    } else {
        MPI_Request request, gi_req = MPI_REQUEST_NULL;
        MPI_Status status;
        while (true) {
            double cur_max_e = DBL_MIN;

            for (int i = 0; i < gctx.n; i++) {
                int source = i % gctx.workers_num + 1;
                MPI_Ibcast(&gctx.X[i], 1, MPI_DOUBLE, source, MPI_COMM_WORLD,
                           &request);
                MPI_WAIT_DEBUG(&request, MPI_STATUS_IGNORE, "for new X");
                _debug("Get X%d from %d: %g", i, source, gctx.X[i]);

                MPI_Recv(&gctx.e[i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD,
                         &status);
                _debug("Get e%d from %d: %g", i, source, gctx.e[i]);

                if (cur_max_e < gctx.e[i]) cur_max_e = gctx.e[i];
            }

            if (gi_req != MPI_REQUEST_NULL)
                MPI_WAIT_DEBUG(&gi_req, MPI_STATUS_IGNORE, "for sending gi");

            if (cur_max_e < gctx.max_e || gctx.i >= ITERATIONS_MAX) {
                printf("Get result for %d iterations, max e %g\n", gctx.i,
                       cur_max_e);
                gctx.i = -1;
                MPI_Ibcast(&gctx.i, 1, MPI_INT32_T, 0, MPI_COMM_WORLD, &gi_req);
                break;
            } else {
                printf("X: \n");
                for (int i = 0; i < gctx.n; i++) {
                    printf("%.2f ", gctx.X[i]);
                }
                printf("\n");
                printf("e: %g\n", cur_max_e);

                gctx.i += 1;
                MPI_Ibcast(&gctx.i, 1, MPI_INT32_T, 0, MPI_COMM_WORLD, &gi_req);
                MPI_WAIT_DEBUG(&gi_req, MPI_STATUS_IGNORE, "for sending gi");
            }
        }

        printf("X: \n");
        for (int i = 0; i < gctx.n; i++) {
            printf("%.2f ", gctx.X[i]);
        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}