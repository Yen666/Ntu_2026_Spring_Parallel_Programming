# HW4 — CUDA SHA-256 Bitcoin Mining

## Implementation

Two-phase pipeline:

1. **CPU**: Compute Merkle root from all transactions in the block; fill into block header
2. **GPU**: Launch kernel threads, each testing a unique nonce value — compute double SHA-256 of block header, find the minimal valid nonce where hash < target

## Parallelization & Optimization Techniques

| Technique | Description |
|-----------|-------------|
| Thread-level parallelism | Each thread tests a unique nonce value |
| Single-direction nonce search | Ensures the minimal valid nonce is found |
| Shared memory flag (`s_found`) | Early exit when valid nonce found; reduces wasted computation |
| Constant memory (`k_device`) | SHA-256 round constants cached for all threads |
| Occupancy optimization | Reduced register usage to improve warp residency |
| Kernel unrolling | Fully unrolled SHA-256 transforms (megakernel fusion) |
| `__funnelshift_rc` intrinsic | Efficient bitwise rotation operations |

## Block & Thread Experiments

Tested on `case01.in`, blocks per grid = 65536:

| Threads/Block | Nonces/Thread | Time (ms) |
|--------------|--------------|-----------|
| 64 | 64 | 2138 |
| 64 | 32 | 2162 |
| 128 | 32 | 2111 |
| 128 | 64 | 2172 |
| 256 | 32 | 2086 |
| 256 | 64 | 2115 |
| **256** | **128** | **1833** |
| 512 | 256 | 2152 |

Best config: **256 threads/block, 128 nonces/thread**

## Advanced CUDA Skills

- **Megakernel SHA-256 fusion**: double SHA-256 fully unrolled and fused into a single `double_sha256_megakernel` device function — reduces kernel launch overhead and maximizes ILP
- **`__constant__` memory**: SHA-256 round constants stored for fast, broadcast-cached access across all threads
- **Shared memory early-exit flag**: once any thread in a block finds a valid nonce, the shared flag signals all other threads to stop
- **`__funnelshift_rc`**: hardware-accelerated bitwise rotation, replacing multi-instruction software rotation
