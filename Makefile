CC      = gcc
NVCC    = nvcc
CFLAGS  = -std=c99 -Wall -Wpedantic
NVFLAGS = --fmad=false

# Benchmark mode: `make ITERS=100` runs exactly 100 iterations instead of
# stopping at the tolerance, and disables the CUDA error checks (and the
# device synchronization they perform after every kernel).
ifdef ITERS
CFLAGS  += -DFIXED_ITERS=$(ITERS)
NVFLAGS += -DFIXED_ITERS=$(ITERS) -DNO_CUDA_CHECK_ERROR
endif

EXES = omp-k-means cuda-k-means inputgen

.PHONY: all omp cuda clean

all: $(EXES)

omp: omp-k-means

cuda: cuda-k-means

omp-k-means: omp-k-means.c hpc.h
	$(CC) $(CFLAGS) -fopenmp $< -o $@

cuda-k-means: cuda-k-means.cu hpc.h
	$(NVCC) $(NVFLAGS) $< -o $@

inputgen: inputgen.c
	$(CC) -std=c99 -Wall -Wpedantic $< -o $@

clean:
	rm -f $(EXES) *.o
