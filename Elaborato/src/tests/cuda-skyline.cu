#include <cstddef>
#include <cstdio>
#if _XOPEN_SOURCE < 600
#include <__clang_cuda_builtin_vars.h>
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hpc.h"

#define BLOCKDIM 1024

typedef struct
{
    float *P; /* coordinates P[i][j] of point i               */
    int N;    /* Number of points (rows of matrix P)          */
    int D;    /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input(points_t *points)
{
    char buf[1024];
    int N, D;
    float *P;

    if (1 != scanf("%d", &D))
    {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin))
    { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N))
    {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float *)malloc(D * N * sizeof(*P));
    assert(P);
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < D; k++)
        {
            if (1 != scanf("%f", &(P[i * D + k])))
            {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points(points_t *points)
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 if |p| dominates |q| */
__device__ int dominates(const float *p, const float *q, int D)
{
    /* The following loops could be merged, but the keep them separated
       for the sake of readability */
    for (int k = 0; k < D; k++)
    {
        if (p[k] < q[k])
        {
            return 0;
        }
    }
    for (int k = 0; k < D; k++)
    {
        if (p[k] > q[k])
        {
            return 1;
        }
    }
    return 0;
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard ouptut. `s[i] == 1` if point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline(const points_t *points, const int *s, int r)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i = 0; i < N; i++)
    {
        if (s[i])
        {
            for (int k = 0; k < D; k++)
            {
                printf("%f ", P[i * D + k]);
            }
            printf("\n");
        }
    }
}

__constant__ int d_N;

__global__ void ker_init(int *s)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < d_N)
    {
        s[index] = 1;
    }
}

__constant__ int d_D;
__device__ int d_r = 0;
__device__ int d_its = 0;

__global__ void ker_single_skyline(float *p, int *s)
{
    __shared__ int local_r;

    const int bindex = blockIdx.x;
    const int tindex = threadIdx.x;

    int t_r = 0;
    if (tindex == 0)
    {
        // initialize the block result
        local_r = 0;
    }

    // current point of each thread
    int elem = tindex + bindex * BLOCKDIM;

    if (elem >= d_N)
    {
        return;
    }

    for (int i = 0; i < d_N; i++)
    {
        if (s[i])
        {
            if (s[elem] && i != elem && dominates(&(p[i * d_D]), &(p[elem * d_D]), d_D))
            {
                s[elem] = 0;
                t_r += 1;
            }
        }
    }

    // each thread sums his result
    atomicAdd(&local_r, t_r);
    __syncthreads();

    // the first thread of each block sums his value in the final variable
    if (tindex == 0)
    {
        atomicAdd(&d_r, local_r);
    }
}

int main(int argc, char *argv[])
{
    points_t points;
    int its = 0;

    float *d_points;
    int *d_s;

    if (argc != 1)
    {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);
    int *s = (int *)malloc(points.N * sizeof(*s));
    assert(s);

    const size_t size_points = points.D * points.N * sizeof(float);
    const size_t size_s = points.N * sizeof(int);

    cudaMalloc((void **)&d_points, size_points);

    cudaMalloc((void **)&d_s, size_s);

    // copy points to GPU memory
    cudaMemcpy(d_points, points.P, size_points, cudaMemcpyHostToDevice);

    // declare global variables
    cudaMemcpyToSymbol(d_N, &points.N, sizeof(int));
    cudaMemcpyToSymbol(d_D, &points.D, sizeof(int));

    const int blocks = (points.N + BLOCKDIM - 1) / BLOCKDIM;

    // init s array
    ker_init<<<blocks, BLOCKDIM>>>(d_s);

    const double tstart = hpc_gettime();
    ker_single_skyline<<<blocks, BLOCKDIM>>>(d_points, d_s);
    const double elapsed = hpc_gettime() - tstart;

    int r = 0;
    cudaMemcpyFromSymbol(&r, d_r, sizeof(int));
    its = r;
    r = points.N - r;

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n", r);
    fprintf(stderr, "\t%d iterations\n\n", its);
    fprintf(stderr, "Execution time (s) %f\n", elapsed);
    printf("%f", elapsed);

    cudaFree(d_points);
    cudaFree(d_s);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
