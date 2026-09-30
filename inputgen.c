/****************************************************************************
 *
 * inputgen.c - Generate random input for the K-Means algorithm.
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
 * This program generates a random input for the K-Means algorithm.
 *
 * To compile:
 *
 * gcc -std=c99 -Wall -Wpedantic inputgen.c -o inputgen
 *
 * To execute:
 *
 * ./inputgen points_per_cluster n_dims n_clusters [seed]
 *
 * The program generates `n_clusters` hyper-rectangles in `n_dims`
 * dimensions with random position; each hyper-rectangle contains
 * `points_per_cluster` random points. Therefore, the total number of
 * points is` points_per_cluster * n_clusters`.
 *
 * The output is deterministic and reproducible on any platform: the
 * same arguments (and `seed`, default 17) always produce the same file.
 * Moreover:
 *
 * - the position and size of the clusters do not depend on
 *   `points_per_cluster`, so inputs of different sizes describe the
 *   same clustering problem (useful for weak scaling);
 *
 * - the points of each cluster are a prefix of those generated with a
 *   larger `points_per_cluster`.
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

/* Portable PRNG (splitmix64). Unlike rand(), whose sequence depends on
   the C library, it produces the same numbers on every platform. */
typedef struct {
    uint64_t state;
} rng_t;

uint64_t rng_next( rng_t *r )
{
    uint64_t z = (r->state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Independent stream number `stream` of the generator seeded with
   `seed`. */
rng_t rng_init( uint64_t seed, uint64_t stream )
{
    rng_t r = { seed ^ (stream * 0xD1B54A32D192ED03ULL) };
    rng_next(&r);
    return r;
}

/* Uniform random number in [0, 1). */
double rand01( rng_t *r )
{
    return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0); /* 2^53 */
}

double randab( rng_t *r, double a, double b )
{
    return a + rand01(r) * (b-a);
}

/* Generate `n` random points inside a `D`-dimensional box with center
   `center` and side `2*r`. */
void gen_points( rng_t *rng, const float *center, int D, float r, int n )
{
    for (int i=0; i<n; i++) {
        for (int d=0; d<D; d++) {
            printf("%f ", center[d] + randab(rng, -r, r));
        }
        printf("\n");
    }
}

void init_center( rng_t *rng, float *p, int D )
{
    for (int d=0; d<D; d++) {
        p[d] = randab(rng, 0, 200);
    }
}

int main( int argc, char *argv[] )
{
    int points_per_cluster, n_dims, n_clusters;
    uint64_t seed = 17;

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s points_per_cluster n_dims n_clusters [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }

    points_per_cluster = atoi(argv[1]);
    n_dims = atoi(argv[2]);
    n_clusters = atoi(argv[3]);
    if (argc == 5) {
        seed = strtoull(argv[4], NULL, 10);
    }

    if (points_per_cluster <= 0 || n_dims <= 0 || n_clusters <= 0) {
        fprintf(stderr, "FATAL: all arguments must be positive\n");
        return EXIT_FAILURE;
    }

    float *center = (float*)malloc(n_clusters * n_dims * sizeof(*center));
    double *side = (double*)malloc(n_clusters * sizeof(*side));
    assert(center != NULL && side != NULL);

    /* Stream 0: geometry of the clusters, independent of
       `points_per_cluster`. */
    rng_t geo = rng_init(seed, 0);
    for (int c=0; c<n_clusters; c++) {
        side[c] = randab(&geo, 10, 30);
        init_center(&geo, &center[c * n_dims], n_dims);
    }

    /* Stream c+1: points of cluster c. */
    for (int c=0; c<n_clusters; c++) {
        rng_t pts = rng_init(seed, (uint64_t)c + 1);
        gen_points(&pts, &center[c * n_dims], n_dims, side[c], points_per_cluster);
    }

    free(side);
    free(center);
    return EXIT_SUCCESS;
}
