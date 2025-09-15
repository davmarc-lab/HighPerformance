/*
 * Marchetti Davide 0001021628
 */

/****************************************************************************
 *
 * cuda-skyline.cu - CUDA implementaiton of the skyline operator
 *
 * Copyright (C) 2024 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * Questo programma calcola lo skyline di un insieme di punti in D
 * dimensioni letti da standard input. Per una descrizione completa
 * si veda la specifica del progetto sulla piattaforma "Virtuale".
 *
 * Per compilare:
 *
 *      nvcc -Wno-deprecated-gpu-targets cuda-skyline.cu -o cuda-skyline -lm
 *
 * Per eseguire il programma:
 *
 *      ./cuda-skyline < input > output
 *
 ****************************************************************************/

#include <cstddef>
#include <cstdio>
#if _XOPEN_SOURCE < 600
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
__device__ int d_r;
__device__ int d_its = 0;

__global__ void ker_skyline(float *p, int *s)
{
    __shared__ int s_its[BLOCKDIM];

    const int bindex = blockIdx.x;
    const int tindex = threadIdx.x;

    s_its[tindex] = 0;

    /* current point of each thread */
    int elem = tindex + bindex * BLOCKDIM;

    if (elem >= d_N)
    {
        return;
    }

    for (int i = 0; i < d_N && s[elem]; i++)
    {
        if (dominates(&(p[i * d_D]), &(p[elem * d_D]), d_D))
        {
            s[elem] = 0;
            s_its[tindex] = 1;
        }
    }

    __syncthreads();

    /* the first thread of each block sums his value in the final variable */
    if (tindex == 0)
    {
        int local_its = 0;
        for (int i = 0; i < BLOCKDIM; i++)
        {
            local_its += s_its[i];
        }
        atomicAdd(&d_its, local_its);
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

    /* points array size */
    const size_t size_points = points.D * points.N * sizeof(float);
    /* s array size */
    const size_t size_s = points.N * sizeof(int);

    /* allocate gpu memory */
    cudaMalloc((void **)&d_points, size_points);
    cudaMalloc((void **)&d_s, size_s);

    /* copy points to GPU memory */
    cudaMemcpy(d_points, points.P, size_points, cudaMemcpyHostToDevice);

    /* declare global variables */
    cudaMemcpyToSymbol(d_N, &points.N, sizeof(int));
    cudaMemcpyToSymbol(d_D, &points.D, sizeof(int));
    cudaMemcpyToSymbol(d_r, &points.N, sizeof(int));

    /* calculate the number of blocks */
    const int blocks = (points.N + BLOCKDIM - 1) / BLOCKDIM;

    const double tstart = hpc_gettime();
    /* init s array */
    ker_init<<<blocks, BLOCKDIM>>>(d_s);

    /* exec skyline */
    ker_skyline<<<blocks, BLOCKDIM>>>(d_points, d_s);

    /* copy results */
    cudaMemcpyFromSymbol(&its, d_its, sizeof(int));
    cudaMemcpy(s, d_s, size_s, cudaMemcpyDeviceToHost);
    const double elapsed = hpc_gettime() - tstart;

    fprintf(stderr, "Its: %d\n", its);

    int r = points.N - its;
    print_skyline(&points, s, r);

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n", r);
    fprintf(stderr, "Execution time (s) %f\n", elapsed);

    cudaFree(d_points);
    cudaFree(d_s);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
