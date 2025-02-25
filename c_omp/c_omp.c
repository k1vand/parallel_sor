#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d out of %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    return 0;
}