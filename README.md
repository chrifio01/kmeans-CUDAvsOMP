# Parallel K-Means: OpenMP vs CUDA

Project for the course *Architectures and Platforms for Artificial Intelligence*
(University of Bologna).

K-Means (Lloyd's algorithm) parallelized with the same design on two
architectures: a multi-core CPU with OpenMP and a GPU with CUDA. Both versions
use data parallelism by point: the points are split among the workers, the
per-cluster counts and sums are accumulated in private copies and combined
after a barrier, and the final update of the centroids (O(K·D)) stays serial.
The goal is to compare the two platforms fairly, so the programs differ only
where the hardware requires it.

The serial code, `hpc.h` and the original `inputgen.c` come from the course
material by Moreno Marzolla.

## Repository

| File / folder | Content |
|---|---|
| `omp-k-means.c` | OpenMP version |
| `cuda-k-means.cu` | CUDA version |
| `inputgen.c` | input generator (portable SplitMix64 PRNG, default seed 17) |
| `hpc.h` | timing and CUDA error-checking helpers |
| `bench/` | scripts that run the programs and compute the metrics and plots |
| `results/` | metrics and plots of the experiments in the report |
| `plot_result.py` | plots the clusters found by one run |

## Requirements

- gcc with OpenMP support
- CUDA toolkit (`nvcc`) and an NVIDIA GPU, for the CUDA version
- Python 3 with matplotlib, for the benchmark scripts and the plots

## Build

```
make                 # omp-k-means, cuda-k-means, inputgen
make omp             # only the OpenMP version
make cuda            # only the CUDA version
make ITERS=100       # benchmark mode (see below)
make clean
```

`make` compiles with:

```
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
nvcc --fmad=false cuda-k-means.cu -o cuda-k-means
gcc -std=c99 -Wall -Wpedantic inputgen.c -o inputgen
```

- `--fmad=false`: no fused multiply-add on the GPU, so it performs the same
  floating-point operations as the CPU version.
- `ITERS=N` adds `-DFIXED_ITERS=N` (run exactly N iterations instead of
  stopping at the tolerance, so every run does the same work) and, for CUDA,
  `-DNO_CUDA_CHECK_ERROR` (no error check and synchronization after each
  kernel). Run `make clean` before switching between normal and benchmark mode.

## Run

```
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./omp-k-means K input_file output_file
./cuda-k-means K input_file output_file
```

Both programs print the time of the main loop (`Elapsed time`, no I/O); the
CUDA version also prints the `Overhead time` (context creation, device
allocation, host-device copies) and the `Total time`. Pinning the OpenMP
threads (`OMP_PROC_BIND`, `OMP_PLACES`) gives much more stable timings.

## Inputs

```
./inputgen points_per_cluster n_dims n_clusters [seed] > file.txt
```

The output depends only on the arguments (default seed 17), on any platform,
and the cluster layout does not depend on `points_per_cluster`: inputs of
different size describe the same problem. The inputs used in the report
(D = 100, 10 clusters):

```
mkdir -p inputs
for N in 100000 200000 400000 800000 1000000; do
    ./inputgen $((N / 10)) 100 10 > inputs/N$N.txt
done
```

## Benchmarks

The scripts in `bench/` run an already compiled program (build with
`make ITERS=100`) on every input of a folder and write `metrics.txt` and the
plots:

```
# OpenMP: 1, 2, 4, 6, ..., T threads -> time, speedup, efficiency, overhead
python bench/bench_omp.py ./omp-k-means 24 10 inputs/ /dev/null --out-dir results/omp

# CUDA: time, overhead and throughput for every input (run the program once before, as warm-up)
python bench/bench_cuda.py ./cuda-k-means 10 inputs/ /dev/null --out-dir results/cuda

# comparison: CUDA speedup over OpenMP with 1, 12, 24 threads and the best p
python bench/bench_compare.py results/omp results/cuda --out-dir results/comp
```

On a laptop, keep it plugged in with the performance power profile. For the
CUDA runs, disabling the runtime power management of the GPU avoids a ~2 s
wake-up cost in the measured overhead:

```
sudo nvidia-smi -pm 1
echo on | sudo tee /sys/bus/pci/devices/<gpu-pci-address>/power/control
```
