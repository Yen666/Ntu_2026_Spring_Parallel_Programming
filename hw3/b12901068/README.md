# HW3 — CUDA Ray Marching: Mandelbulb Rendering

## Implementation Overview

Uses CUDA to parallelize ray marching for rendering Mandelbulbs.

### Device Functions

| Function | Role |
|----------|------|
| `md()` | Mandelbulb distance estimator using iterative formula v^8 + c |
| `map()` | Scene mapping with 90° rotation along X-axis |
| `trace()` | Ray marching to find surface intersection |
| `calcNor()` | Surface normal via gradient approximation |
| `softshadow()` | Soft shadow casting with secondary ray marching |
| `pal()` | Color palette generation for orbit trap coloring |

## Task Partitioning

```
Block size: 16×16 = 256 threads/block
Grid size:  ceil(width/16) × ceil(height/16)
Thread assignment: each thread processes one pixel
```

## Optimization Techniques

### Mathematical
- **Power function replacement**: custom `powi8`/`powi7` using repeated squaring → ~2–3× speedup in `md()`
- **Fast math**: `-use_fast_math` flag; uses `__cosf`, `__sinf`, `__logf` → ~20–30% gain
- **Trigonometric reuse**: precompute `cosTheta`, `sinTheta`, `cosPhi`, `sinPhi` and reuse in vec3 construction

### Memory
- **Constant memory consolidation**: packed 13 `SceneParams` into one struct → `cudaMemcpyToSymbol` from 146ms → <5ms
- **Shared memory for camera basis**: computed once per block by thread (0,0), then `__syncthreads()`
- **Pinned host memory**: `cudaMallocHost` → faster D2H transfer (~15ms vs ~23ms)

### Register
- **`__launch_bounds__(256, 2)`**: hints compiler to reduce registers from 66 → 56 per thread; improves occupancy
- **Variable reduction**: eliminated temporaries (`r2`, `r4`, `r8`) via inline functions

### PNG Encoding
- Disabled zlib compression (`btype=0`, `use_lz77=0`, `LFS_ZERO`) → encoding time 4338ms → ~300ms

## Performance Results

| Stage | Before | After |
|-------|--------|-------|
| `cudaMemcpyToSymbol` | 146 ms | <5 ms |
| Kernel launch | ~4125 ms | ~4000 ms |
| D2H memcpy | 22.9 ms | ~15 ms |
| PNG write | 4338 ms | ~300 ms |
| **Total** | **~8897 ms** | **~4584 ms (−48%)** |

## Block Size Comparison (Nsight)

| Block Size | Registers/Thread | Occupancy | Kernel Time | Notes |
|------------|-----------------|-----------|-------------|-------|
| 32×32 (1024) | Overflow | N/A | N/A | Too many threads |
| **16×16 (256)** | **56** | **62.5%** | **~192ms** | **Optimal** |
| 8×8 (64) | 52 | 75% | ~207ms | Lower parallelism |

16×16 achieves highest compute throughput (80.5%) due to balanced register/memory usage and coalesced memory access.

## Occupancy Calculation

```
Max threads/SM: 2048
Block size: 256, Registers/thread: 56
Blocks/SM: floor(65536 / (256×56)) = 4
Threads/SM: 4 × 256 = 1024
Occupancy: 1024 / 2048 = 50% (theoretical); achieved ~62.5%
```

## Difficulties

- **Register pressure vs occupancy**: 66 registers → 50% occupancy; solved with `__launch_bounds__` and inlining
- **PNG encoding bottleneck**: lodepng default settings took longer than GPU kernel; solved with zero-compression mode
- **CUDA initialization**: first `cudaMemcpyToSymbol` takes ~264ms (context creation); mitigated by consolidating 13 calls → 1, saving 140ms
