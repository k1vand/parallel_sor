#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>


struct global_ctx_s {
    atomic_uint i;
    double **A;
    double *b;
    double *e;
    double **X[2];
    double max_e;
};

struct  tctx_s
{
    uint32_t idx;
    
};


int main(){
    printf("hell world\n");
    return 0;
}
