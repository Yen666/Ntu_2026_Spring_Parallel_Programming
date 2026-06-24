# NTU Parallel Programming (2025 Fall)

Homework assignments for the Parallel Programming course at National Taiwan University.

## Assignments

| HW | Topic | Technology |
|----|-------|------------|
| [hw1](./hw1/) | Mandelbrot Set | Pthread |
| [hw2](./hw2/b12901068/) | SIFT Feature Detection | OpenMP |
| [hw3](./hw3/b12901068/) | GPU Programming | CUDA |
| [hw4](./hw4/b12901068/) | SHA-256 Bitcoin Mining | CUDA |
| [hw5](./hw5/b12901068/) | Distributed Computing | MPI |

## Extra: SokobanSolver

A Sokoban puzzle solver implemented with A\*, BFS, and DFS search algorithms.
See [SokobanSolver/README.md](./SokobanSolver/README.md) for details.

## Prerequisites

- GCC / G++
- NVIDIA CUDA Toolkit (hw3, hw4)
- OpenMPI (hw5)
- OpenMP (hw2)

## Build

Each assignment has its own `Makefile`. Run inside the respective folder:

```bash
make
```
