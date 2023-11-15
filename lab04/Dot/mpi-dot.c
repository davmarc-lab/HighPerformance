/****************************************************************************
 *
 * mpi-dot.c - Dot product
 *
 * Copyright (C) 2016--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
  % HPC - Dot product
  % Moreno Marzolla <moreno.marzolla@unibo.it>
  % Last updated: 2022-11-01

  The file [mpi-dot.c](mpi-dot.c) contains a MPI program that computes
  the dot product between two arrays `a[]` and `b[]` of length $n$. The
  dot product $s$ of two arrays `a[]` and `b[]` is defined as:

  $$
  s = \sum_{i = 0}^{n-1} a[i] \times b[i]
  $$

  In the provided program, the master performs the whole computation and
  is therefore not parallel. The goal of this exercise is to write a
  parallel version. Assume that, at the beginning of the program, `a[]`
  and `b[]` are known only to the master. Therefore, they must be
  distributed across the processes. Each process computes the scalar
  product of the assigned portions of the arrays; the master then uses
  `MPI_Reduce()` to sum the partial results and compute $s$.

  You may initially assume that $n$ is an exact multiple of the number
  of MPI processes $P$; then, relax this assumption and modify the
  program so that it works with any array length $n$. The simpler
  solution is to distribute the arrays using `MPI_Scatter()` and let the
  master take care of any excess data. Another possibility is to use
  `MPI_Scatterv()` to distribute the input unevenly across the
  processes.

  To compile:

  mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot -lm

  To execute:

  mpirun -n P ./mpi-dot [n]

Example:

mpirun -n 4 ./mpi-dot 1000

## Files

- [mpi-dot.c](mpi-dot.c)

 ***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot( const double* x, const double* y, int n )
{
    double s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main( int argc, char* argv[] )
{
    const double TOL = 1e-5;
    double *x = NULL, *y = NULL, result = 0.0;
    int i, n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 == my_rank ) {
        /* The master allocates the vectors */
        x = (double*)malloc( n * sizeof(*x) ); assert(x != NULL);
        y = (double*)malloc( n * sizeof(*y) ); assert(y != NULL);
        for ( i=0; i<n; i++ ) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }
    }
    /* [TODO] This is not a true parallel version, since the master
       does everything */
    /* if ( 0 == my_rank ) { */
    /*     result = dot(x, y, n); */
    /* } */

    int *sendcounts = (int*)malloc(comm_sz * sizeof(int));
    assert(sendcounts != NULL);

    int *displ = (int*)malloc(comm_sz * sizeof(int));
    assert(displ != NULL);

    // initialize scatterv parameters
    for (int i = 0; i < comm_sz; i++)
    {
        const int start = n * i / comm_sz;
        const int end = n * (i + 1) / comm_sz;

        displ[i] = start;
        sendcounts[i] = end - start;
    }

    // number of elements to receive/send
    const int local_n = sendcounts[my_rank];

    // local buffers
    double *localx = (double*)malloc(local_n * sizeof(double));
    assert(localx != NULL);
    double *localy = (double*)malloc(local_n * sizeof(double));
    assert(localy != NULL);

    MPI_Scatterv(x,
            sendcounts,
            displ,
            MPI_DOUBLE,
            localx,
            local_n,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);

    MPI_Scatterv(y,
            sendcounts,
            displ,
            MPI_DOUBLE,
            localy,
            local_n,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD); 

    double local_res = dot(localx, localy, local_n);

    MPI_Reduce(&local_res,
            &result,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            0,
            MPI_COMM_WORLD);

    if (0 == my_rank) {
        printf("Dot product: %f\n", result);
        if ( fabs(result - n) < TOL ) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);
    free(localx);
    free(localy);
    free(sendcounts);
    free(displ);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
