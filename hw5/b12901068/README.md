# HW5 — N-Body Simulation on Multi-GPU

## Architecture

Multi-level parallelism with adaptive kernel design based on problem size N, managed across 2 GPUs via Task-Level Parallelism.

## Parallelism Strategy

### Small Scale (N < 100): Persistent Kernel
- Single block; all data loaded into shared memory
- All 200,000 time steps completed inside the GPU without leaving
- Only **1 kernel launch** — eliminates launch overhead entirely
- Applicable to: `b20.in` ~ `b100.in`

### Large Scale (N ≥ 100): Loop Kernel
- N blocks; each block handles one body
- **2 kernel launches per step** (nbody + check)
- Total: 200,000 × 2 = 400,000 kernel launches
- Applicable to: `b200.in` ~ `b1024.in`

## 2-GPU Resource Management

Task-Level Parallelism using `std::thread` + CUDA Streams:

| GPU | Responsibility |
|-----|----------------|
| GPU 0 | Problem 1 (minimum distance) + Problem 2 (collision detection), overlapped via Stream 0 & Stream 0b |
| GPU 1 | Problem 3 (missile simulation), multiple streams for each Gravity Device in parallel |

## Optimization Techniques

- **Branch divergence removal**: precompute effective mass into shared memory (`s_eff_m[j]`) to eliminate conditional branches on `is_device` in the inner loop
- **Fast math**: `rsqrt` instruction replaces `pow` for inverse-distance gravity calculation
  ```cpp
  double inv_dist  = rsqrt(dist_sq);
  double inv_dist3 = inv_dist * inv_dist * inv_dist;
  double f = G * mj * inv_dist3;
  ```
- **Shared memory tiling**: load position and mass data into shared memory for reuse, significantly reducing global memory bandwidth pressure
- **Compiler flags**: `-ffast-math`, `-munsafe-fp-atomics`

## Design Questions

### Q: If there are 4 GPUs instead of 2?

Mixed assignment strategy:
- **GPU 0**: Problem 1
- **GPU 1**: Problem 2
- **GPU 2 & 3**: Problem 3 split across both GPUs

Rationale: P1 and P2 have similar compute cost — one GPU each balances load. P3 is the most expensive and each Gravity Device simulation is independent (no inter-GPU communication needed), enabling near-linear speedup.

### Q: With 5 gravity devices, must we run 5 independent simulations?

No. Use a **fork-on-destruction** strategy:

1. Run a single **base simulation** assuming no device is destroyed
2. When a device is about to be destroyed at time step *t*: snapshot the current state (all positions P and velocities V) into a new memory region
3. In the fork: set that device's mass to 0, launch a new CUDA Stream to continue from step *t*
4. The base simulation continues to detect and trigger remaining destruction events

This eliminates all redundant computation before each destruction event — only the diverging post-event trajectories need to be simulated separately.
