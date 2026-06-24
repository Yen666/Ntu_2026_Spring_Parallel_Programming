# HW2 — Parallel SIFT Feature Detection (MPI + OpenMP)

## Architecture

Hybrid **MPI + OpenMP** parallelism:
- **MPI** (process-level): coarse-grained work distribution across nodes — distributes octaves, scales, or image regions
- **OpenMP** (thread-level): fine-grained pixel-level parallelism within each MPI process

## Task Partitioning Per Stage

| Stage | MPI Distribution | OpenMP Parallelism |
|-------|-----------------|-------------------|
| Gaussian Blur | None | 2D pixel-level (`collapse(2)`) |
| Gaussian Pyramid | Round-robin (octave, scale) | Pixel-level Gaussian blur |
| DoG Pyramid | Round-robin (octave, scale) | Pixel-level subtraction |
| Gradient Pyramid | Round-robin (octave, scale) | 2D gradient computation |
| Keypoint Detection | Round-robin (octave, scale) | Pixel-level extrema search |
| Descriptor | Round-robin (keypoint id) | Histogram computation |

### Why Round-Robin?
Avoids load imbalance caused by varying image sizes across octaves.

## Scheduling

```cpp
#pragma omp parallel for schedule(static)
```
Each iteration has equal workload → `static` scheduling avoids runtime overhead and naturally achieves load balance.

## Optimization Techniques

- **Loop interchange**: Changed to row-major order to reduce cache misses
- **SIMD**: `#pragma omp simd` for auto-vectorization in normalization and histogram smoothing
- **Halo Exchange**: Used `MPI_Sendrecv` to pass boundary rows, reducing communication volume
- **Cache optimization**: Copied columns to contiguous buffer to improve cache hit rate
- **Keypoint collection**: Thread-local buffer avoids critical sections; merged with `MPI_Allgatherv`

## Communication Pattern

- **All-to-All**: `MPI_Allgatherv` after each major stage to ensure data consistency
- **Volume**: Gaussian/DoG O(total pixels); Keypoints O(num_keypoints × 136)

## Scalability Analysis

| Resource | Scaling behavior |
|----------|-----------------|
| More nodes | Distributes work, but cross-node communication overhead increases |
| More processes/node | Reduces time up to a point; too many increases memory and communication cost |
| More CPU cores/process | Faster up to physical core count; beyond that, performance degrades |

## Difficulties & Solutions

| Problem | Solution |
|---------|----------|
| Cache miss in vertical convolution | Copied column to contiguous memory before processing |
| Load imbalance when height not divisible | Manually assigned remaining rows to last process |
