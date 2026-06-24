# NTU Parallel Programming (2025 Fall)

Homework assignments for the Parallel Programming course at National Taiwan University.

## Assignments

| HW | Topic | Technology | Notes |
|----|-------|------------|-------|
| [hw1](./hw1/) | Parallel Sokoban Solver | Pthread | A\* with Hungarian heuristic, bitset state |
| [hw2](./hw2/b12901068/) | SIFT Feature Detection | MPI + OpenMP | Hybrid parallelism across octave/scale |
| [hw3](./hw3/b12901068/) | Mandelbulb Ray Marching | CUDA | −48% runtime via memory & math optimizations |
| [hw4](./hw4/b12901068/) | SHA-256 Bitcoin Mining | CUDA | Megakernel fusion, shared-memory early exit |
| [hw5](./hw5/b12901068/) | N-Body Simulation | Multi-GPU CUDA | Adaptive persistent/loop kernel, 2-GPU task split |

## Extra: SokobanSolver

A Sokoban puzzle solver implemented with A\*, BFS, and DFS search algorithms.
See [SokobanSolver/README.md](./SokobanSolver/README.md) for details.

## Prerequisites

- GCC / G++
- NVIDIA CUDA Toolkit (hw3, hw4, hw5)
- OpenMPI (hw2, hw5)
- OpenMP (hw2)

## Build

Each assignment has its own `Makefile`. Run inside the respective folder:

```bash
make
```
