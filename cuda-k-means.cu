/****************************************************************************
 *
 * cuda-k-means.cu -- CUDA implementation of the K-Means clustering algorithm.
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
 *      nvcc cuda-k-means.cu -o cuda-k-means
 *
 * Run with:
 *
 *      ./cuda-k-means K input_file output_file
 *
 * Example:
 *
 *      ./cuda-k-means 5 demo.txt demo.out
 *
 * Benchmark mode: run exactly N iterations instead of stopping at the
 * tolerance, so that every version does the same amount of work
 * regardless of floating-point rounding; -DNO_CUDA_CHECK_ERROR removes
 * the synchronization done by cudaCheckError() (see hpc.h):
 *
 *      nvcc -DFIXED_ITERS=50 -DNO_CUDA_CHECK_ERROR cuda-k-means.cu -o cuda-k-means
 *      ./cuda-k-means K input_file /dev/null
 *
 * Validation against the OpenMP version: nvcc fuses multiply and add
 * (FMA) by default, gcc does not; --fmad=false makes the GPU perform the
 * same floating-point operations as the CPU, so the distances are
 * bit-for-bit identical:
 *
 *      nvcc --fmad=false cuda-k-means.cu -o cuda-k-means
 *
 * The flags can be combined (e.g. -DFIXED_ITERS=50 --fmad=false).
 *
 * Output: "Elapsed time" is the wall-clock time of the main loop only
 * (no I/O), comparable with the OpenMP version; "Overhead time" is
 * everything the GPU needs besides the main loop (context creation,
 * kernel loading, device memory allocation, one-off host<->device
 * copies), measured as a whole; "Total time" is their sum.
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

#define BLKDIM 256      /* number of CUDA threads per block */

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

/* Copy `p2` into `p1`. */
void vcopy( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

/* GPU version of the `sqdist()` function of the serial code.
Not parallelized for openmp benchmark comparison, but it's necessary to define it to be called on a __global__ function. */
__device__ float d_sqdist( const float *p1, const float *p2, int n_dims )
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
   `data` and `centroids`. These are all matrices
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

/* Data parallelism by point: the grid has one thread per point
   (gindex); each thread compares its point with the K centroids, one at
   a time (same work as an OpenMP thread for each of its N/T points).
   - cluster_of[gindex]: every point is written by exactly one thread,
     so no synchronization is needed; data and centroids are only read.
   - counts: each block increments a private copy (local_counts, in
     shared memory, zeroed at the start of the kernel). Threads of the
     same block share it, so they use atomicAdd on shared memory (fast).
     After __syncthreads() (the barrier) each block adds its copy to
     counts with one atomicAdd per cluster: the equivalent of the OpenMP
     reduction. Integer sums are exact, so the order does not matter. */
__global__ void classify_kernel( const float *data, const float *centroids,
                                 int *cluster_of, int *counts,
                                 int n_points, int n_dims, int n_clusters )
{
    extern __shared__ int local_counts[];           // n_clusters ints
    const int lindex = threadIdx.x;                 // identify local thread index
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;   // identify global thread index hence data point

    // zero the block-local histogram
    for (int j = lindex; j < n_clusters; j += blockDim.x)
        local_counts[j] = 0;
    __syncthreads();

    if (gindex < n_points) {
        const float *p = &data[gindex * n_dims];
        int nearest = 0;
        float mindist = d_sqdist(p, &centroids[0], n_dims);
        for (int j=1; j<n_clusters; j++) {
            const float dist = d_sqdist(p, &centroids[j * n_dims], n_dims);
            if (dist < mindist) {
                mindist = dist;
                nearest = j;
            }
        }
        cluster_of[gindex] = nearest;
        atomicAdd(&local_counts[nearest], 1);
    }
    __syncthreads();                                // barrier

    // merge the block histogram into the global one
    for (int j = lindex; j < n_clusters; j += blockDim.x)
        if (local_counts[j] > 0)
            atomicAdd(&counts[j], local_counts[j]);
}


/* Step 1 of the centroids update: sum of the points of each cluster
   (the parallel loop of the OpenMP update_centroids()).

   Data parallelism by point: the grid has one thread per point
   (gindex); each thread adds its point to the row of its cluster
   (cluster_of[gindex]) in a private copy of new_centroids.
   - Private copies: one per block (local_sums, in shared memory, zeroed
     at the start of the kernel), not one per thread: K*D floats per
     thread would not fit. Threads of the same block share it, so they
     use atomicAdd on shared memory.
   - After __syncthreads() each block writes its copy to its own slice
     of `partial` (length n_blocks * K * D): blocks never write the same
     memory, so no global atomic is needed.
   - The end of the kernel is the global barrier (the implicit barrier
     of the OpenMP loop); reduce_partials_kernel() then sums the slices.
   - data and cluster_of are only read. */
__global__ void update_centroids_kernel( const float *data, const int *cluster_of,
                                         float *partial,
                                         int n_points, int n_dims, int n_clusters )
{
    extern __shared__ float local_sums[];           // n_clusters * n_dims floats
    const int lindex = threadIdx.x;
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    const int size = n_clusters * n_dims;

    // zero the block-local sums
    for (int k = lindex; k < size; k += blockDim.x)
        local_sums[k] = 0.0f;
    __syncthreads();

    if (gindex < n_points) {
        const int c = cluster_of[gindex];
        for (int d=0; d<n_dims; d++)
            atomicAdd(&local_sums[c * n_dims + d], data[gindex * n_dims + d]);
    }
    __syncthreads();                                // barrier

    // write the block sums to this block's slice
    for (int k = lindex; k < size; k += blockDim.x)
        partial[blockIdx.x * size + k] = local_sums[k];
}

/* Step 2 of the centroids update: sum the private copies (the reduction
   of the OpenMP update_centroids()).

   One thread per element k of new_centroids (size = K * D): it adds
   element k of every block's slice of `partial`, always in the order
   b = 0, 1, ..., n_blocks-1.
   - Every element is written by exactly one thread: no atomic needed.
   - The fixed order makes the result identical from run to run.
   - new_centroids is fully overwritten, so it does not need to be
     zeroed before the launch. */
__global__ void reduce_partials_kernel( const float *partial, float *new_centroids,
                                        int n_blocks, int size )
{
    const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < size) {
        float s = 0.0f;
        for (int b = 0; b < n_blocks; b++)
            s += partial[b * size + k];
        new_centroids[k] = s;
    }
}

/* Step 3 of the centroids update: set the centroid of each cluster to
   the barycenter of its points (new_centroids holds the sums) and store
   in *maxshift the maximum (squared) shift of all centroids.

   Serial part: the same loop as the serial part of the OpenMP
   update_centroids(), launched with a single thread (<<<1, 1>>>).
   - Kept serial for the same reasons as in OpenMP: at most K threads
     could work, and there are only K*D operations, negligible compared
     with the O(N*K*D) of classify_kernel().
   - Executed on the GPU and not on the CPU so that the centroids never
     leave the device: only *maxshift (4 bytes) is copied back at every
     iteration, for the termination test. */
__global__ void finalize_centroids_kernel( float *centroids, float *new_centroids,
                                           const int *counts, float *maxshift,
                                           int n_dims, int n_clusters )
{
    float ms = 0.0f;
    for (int j=0; j<n_clusters; j++) {
        float *c = &centroids[j * n_dims];
        float *nc = &new_centroids[j * n_dims];
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            for (int d=0; d<n_dims; d++)
                nc[d] = c[d];
        } else {
            const float inv = 1.0f / counts[j];
            for (int d=0; d<n_dims; d++)
                nc[d] *= inv;
        }
        const float shift = d_sqdist(c, nc, n_dims);
        if (shift > ms)
            ms = shift;
        for (int d=0; d<n_dims; d++)
            c[d] = nc[d];
    }
    *maxshift = ms;
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
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));

    init_centroids();

    printf("Main loop starts\n\n");

    /* Overhead: everything the GPU needs besides the main loop, i.e.
       context creation and kernel loading (at the first CUDA call),
       device memory allocation and the one-off copies in (here), plus
       the copies out after the loop. None of it exists in the OpenMP
       version, whose overhead is estimated as T_par(T) - T_par(1)/T.
       Kernels are loaded eagerly, at context creation, so that the
       main loop does not pay for it at the first launch. */
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    const double tovh_in = hpc_gettime();

    /* Device copies of the arrays used by the kernels. `data` never
       changes, so it is copied only once. */
    float *d_data, *d_centroids, *d_new_centroids, *d_partial, *d_shift;
    int *d_cluster_of, *d_counts;
    cudaSafeCall( cudaMalloc((void**)&d_shift,         sizeof(*d_shift)) );
    cudaSafeCall( cudaMalloc((void**)&d_data,          n_points * n_dims * sizeof(*d_data)) );
    cudaSafeCall( cudaMalloc((void**)&d_centroids,     n_clusters * n_dims * sizeof(*d_centroids)) );
    cudaSafeCall( cudaMalloc((void**)&d_new_centroids, n_clusters * n_dims * sizeof(*d_new_centroids)) );
    cudaSafeCall( cudaMalloc((void**)&d_cluster_of,    n_points * sizeof(*d_cluster_of)) );
    cudaSafeCall( cudaMalloc((void**)&d_counts,        n_clusters * sizeof(*d_counts)) );

    cudaSafeCall( cudaMemcpy(d_data, data, n_points * n_dims * sizeof(*data), cudaMemcpyHostToDevice) );
    /* initial centroids: from now on they stay on the device */
    cudaSafeCall( cudaMemcpy(d_centroids, centroids, n_clusters * n_dims * sizeof(*centroids), cudaMemcpyHostToDevice) );

    const int n_blocks = (n_points + BLKDIM - 1) / BLKDIM;  // integer division rounded up to cover every point, even when N isn't a multiple of the block size
    const int size = n_clusters * n_dims;                          // elements of new_centroids 
    const int n_blocks_reduce = (size + BLKDIM - 1) / BLKDIM;   // same logic as before
    const size_t shmem = n_clusters * sizeof(int);                // block-local histogram
    const size_t sums_shmem = size * sizeof(float);                // block-local sums

    // one slice of partial sums per block of update_centroids_kernel()
    cudaSafeCall( cudaMalloc((void**)&d_partial, (size_t)n_blocks * size * sizeof(*d_partial)) );

    // The block-local sums must fit in shared memory.
    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, 0) );
    if (sums_shmem > prop.sharedMemPerBlock) {
        fprintf(stderr, "FATAL: K*D = %d floats do not fit in shared memory (%zu bytes)\n",
                n_clusters * n_dims, prop.sharedMemPerBlock);
        return EXIT_FAILURE;
    }
    const double overhead_in = hpc_gettime() - tovh_in;

    float shift;
    int iter = 0;
    const double tstart = hpc_gettime();    // for wallclock time
    do {
        cudaSafeCall( cudaMemset(d_counts, 0, n_clusters * sizeof(*d_counts)) );
        classify_kernel<<<n_blocks, BLKDIM, shmem>>>(d_data, d_centroids, d_cluster_of, d_counts,
                                                     n_points, n_dims, n_clusters);
        cudaCheckError();    // introduce overhead compared to omp version
        update_centroids_kernel<<<n_blocks, BLKDIM, sums_shmem>>>(d_data, d_cluster_of, d_partial,
                                                                  n_points, n_dims, n_clusters);
        cudaCheckError();    // introduce overhead compared to omp version
        // kernel boundary = global barrier
        reduce_partials_kernel<<<n_blocks_reduce, BLKDIM>>>(d_partial, d_new_centroids, n_blocks, size);
        cudaCheckError();    // introduce overhead compared to omp version
        // serial step 3, on the GPU
        finalize_centroids_kernel<<<1, 1>>>(d_centroids, d_new_centroids, d_counts, d_shift,
                                            n_dims, n_clusters);
        cudaCheckError();    // introduce overhead compared to omp version
        /* Only the shift (4 bytes) comes back, for the termination test;
           the blocking copy also waits for the kernels to complete. */
        cudaSafeCall( cudaMemcpy(&shift, d_shift, sizeof(shift), cudaMemcpyDeviceToHost) );
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

    /* Overhead (continued): centroids and cluster_of are needed only
       for the output. */
    const double tovh_out = hpc_gettime();
    cudaSafeCall( cudaMemcpy(centroids, d_centroids, n_clusters * n_dims * sizeof(*centroids), cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaMemcpy(cluster_of, d_cluster_of, n_points * sizeof(*cluster_of), cudaMemcpyDeviceToHost) );
    const double overhead = overhead_in + (hpc_gettime() - tovh_out);

    printf("\nMain loop completed\n");
    printf("Elapsed time %.3f\n", elapsed);    /* main loop only, comparable with OpenMP */
    printf("Overhead time %.3f\n", overhead);
    printf("Total time %.3f\n\n", elapsed + overhead);

    save_results(outputf);

    fclose(outputf);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_partial);
    cudaFree(d_shift);
    cudaFree(d_cluster_of);
    cudaFree(d_counts);

    free(data);
    free(centroids);
    free(cluster_of);

    return EXIT_SUCCESS;
}
