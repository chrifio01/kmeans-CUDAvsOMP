/****************************************************************************
 *
 * k-means.c -- Implementation of the K-Means clustering algorithm.
 *
 * Copyright (C) 2025 Moreno Marzolla <https://unibo.it/sitoweb/moreno.marzolla/>
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
 * Compile with (normal mode: stops when the centroids shift less than
 * TOL, or after MAXITER iterations):
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *
 * Run with (the number of threads is set by OMP_NUM_THREADS):
 *
 *      OMP_NUM_THREADS=T ./omp-k-means K input_file output_file
 *
 * Example:
 *
 *      OMP_NUM_THREADS=8 ./omp-k-means 5 demo.txt demo.out
 *
 * Benchmark mode: run exactly N iterations instead of stopping at the
 * tolerance, so that every version and thread count does the same
 * amount of work regardless of floating-point rounding:
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp -DFIXED_ITERS=50 omp-k-means.c -o omp-k-means
 *      OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=T ./omp-k-means K input_file /dev/null
 *
 * Validation against the CUDA version: gcc does not fuse multiply and
 * add (FMA) with these flags, so the CUDA version must be compiled
 * with --fmad=false to perform the same floating-point operations.
 *
 * Output: "Elapsed time" is the wall-clock time of the main loop only
 * (no I/O), measured with hpc_gettime() (omp_get_wtime()).
 *
 ****************************************************************************/

/* Enable POSIX extensions which are required for making function
   `clock_gettime()` (used in "hpc.h") visible. The following
   `#define` must come at the very beginning, before including
   anything else.
*/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

int n_dims;             /* number of dimensions.                */

int n_points;           /* number of data points.               */

int n_clusters;         /* number of clusters.                  */

float *data;            /* [array of length (n_points * n_dims)]
                           `&data[i*n_dims]` points to the beginning
                           of the i-th data items, which is an array
                           of `n_dims` floating-point numbers.  */

float *centroids;       /* [array of length (n_clusters * n_dims)]
                           `&centroids[j*n_dims]` points to the
                           beginning of the j-th centroid, which is an
                           array of `n_dims` floating point
                           numbers.                             */

float *new_centroids;   /* [array of length (n_clusters * n_dims)] */

int *counts;            /* [array of length n_clusters] `counts[j]`
                           is the number of points that belong to
                           cluster j.                           */

int *cluster_of;        /* [array of length n_points] `clusters_of[i]`
                           is the ID of the cluster assigned to the
                           i-th data point; cluster IDs are integer in
                           0..(n_clusters-1).                   */


/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
void vzero( float *p )
{
    for (int d=0; d<n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
void vadd( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul( float *p, float v )
{
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist( float *p1, float *p2 )
{
    float result = 0.0;
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
int IDX(int i, int d)
{
    return i*n_dims + d;
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/* Assign each data point to the nearest centroid. Updates
   the `counts` array. */
void classify( void )
{
    for (int j=0; j<n_clusters; j++) {
        counts[j] = 0;
    }

    /* Data parallelism by point: schedule(static) gives each of the T
    threads a contiguous block of about N/T points. Each thread compares
    its points, one at a time, with the K centroids.
    - cluster_of[i]: every point is written by exactly one thread, so no
     synchronization is needed; data and centroids are only read.
    - counts: each thread increments a private copy (initialized to 0);
     the reduction sums the copies into counts at the end of the loop,
     so no atomic is needed. */
    #pragma omp parallel for schedule(static) default(none) shared(data, centroids, cluster_of, n_points, n_clusters) reduction(+:counts[:n_clusters])
    for (int i=0; i<n_points; i++) {
        int nearest = 0;
        float mindist = sqdist( &data[IDX(i,0)], &centroids[IDX(nearest, 0)] );
        for (int j=1; j<n_clusters; j++) {
            const float dist = sqdist( &data[IDX(i, 0)], &centroids[IDX(j, 0)] );
            if ( dist < mindist ) {
                mindist = dist;
                nearest = j;
            }
        }
        cluster_of[i] = nearest;
        counts[nearest]++;
    }
    // implicit barrier
}


/* Update the centroids. Set the centroid of each cluster to the
   barycenter of the points. Returns the maximum shift, i.e., the
   maximum difference between the (squared) old and new position of
   all centroids. */
float update_centroids( void )
{
    for (int j=0; j<n_clusters; j++) {
        vzero( &new_centroids[IDX(j, 0)] );
    }

    /* Data parallelism by point: schedule(static) gives each of the T
    threads a contiguous block of about N/T points. Each thread adds its
    points to the row of their cluster (cluster_of[i]) in a private copy
    of new_centroids, initialized to 0 by the reduction.
    - Private copies: threads never write the same memory, so no atomic
     or critical section is needed, even when points of the same cluster
     are handled by different threads.
    - The implicit barrier at the end of the loop is followed by the
     reduction, which sums the T copies into new_centroids.
    - data and cluster_of are only read. */
    #pragma omp parallel for schedule(static) default(none) shared(data, cluster_of, n_points) reduction(+:new_centroids[:n_clusters*n_dims])
    for (int i=0; i<n_points; i++) {
        vadd( &new_centroids[IDX(cluster_of[i], 0)], &data[IDX(i, 0)] );
    }
    // implicit barrier

    /* Serial part: kept serial on purpose.
    - Limited parallelism: there is only one iteration per cluster, so at
     most K threads could work (with T > K, T - K threads would be idle
     and the speedup cannot exceed K).
    - Tiny workload: only K*D operations (e.g. 10*30 = 300), negligible
     compared with the O(N*K*D) of classify(); the cost of a parallel
     region plus a max-reduction on maxshift would exceed the gain. */
    float maxshift = 0.0f;
    for (int j=0; j<n_clusters; j++) {
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            vcopy( &new_centroids[IDX(j,0)], &centroids[IDX(j,0)] );
        } else {
            vmul( &new_centroids[IDX(j, 0)], 1.0f/counts[j] );
        }
        const float shift = sqdist( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
        if (shift > maxshift)
            maxshift = shift;
        vcopy( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
    }

    return maxshift;
}

/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. This function figures out how many numbers are in a row,
   and how many rows there are. Then, it initializes the variables
   `n_dims` and `n_points` accordingly. */
void read_input( FILE *f )
{
    /* Enough for about 5000 dimensions (a D=100 line is ~1050 chars). */
    const size_t BUFLEN = 65536;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_items % n_dims == 0);  /* If this assertion fails, then
                                       there is some line of the input
                                       file that has != n_dims
                                       items. */

    n_points = n_items / n_dims;

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j=0; j<n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d=0; d<n_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;
    const float TOL = 1e-5;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    read_input(inputf);
    fclose(inputf);

    assert(n_clusters < n_points);

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n",     n_points);
    fprintf(outputf, "# Dimensions: %d\n",      n_dims);
    fprintf(outputf, "# Clusters: %d\n",        n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

    printf("Main loop starts\n\n");

    float shift;
    int iter = 0;
    const double tstart = hpc_gettime();    // for wallclock time
    do {
        classify();
        shift = update_centroids();
        printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;
#ifdef FIXED_ITERS
    /* Benchmark mode: run exactly FIXED_ITERS iterations, so that all
       versions do the same amount of work regardless of rounding. */
    } while ( iter < FIXED_ITERS );
    (void)TOL; (void)MAXITER;
#else
    } while ( (shift > TOL) && (iter <= MAXITER) );
#endif
    const double elapsed = hpc_gettime() - tstart;

    printf("\nMain loop completed\n");
    printf("Elapsed time %.3f\n\n", elapsed);

    save_results(outputf);

    fclose(outputf);

    free(data);
    free(centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}
