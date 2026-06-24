# HW1 — Parallel Sokoban Solver (Pthread)

## Implementation

- **Search Algorithm**: A\* with Hungarian algorithm as the heuristic function
- **State Representation**: Each state stored as a `bitset`, encoding box positions and reachable regions
- **State Expansion**: From the current state, expand by pushing each reachable box
- **Deadlock Detection**:
  - Static deadlock: precompute via BFS to mark unreachable positions
  - Corner deadlock and wall-adjacent deadlock detection

## Parallelization

- Parallelized the A\* search using **Pthread**
- Fixed race conditions on the shared priority queue using **mutex locks**

## Difficulties & Solutions

| Problem | Solution |
|---------|----------|
| Bugs accumulating after adding features | Used git version control to roll back |
| Race conditions in parallel search | Added mutex, resolved data races |
| Slow performance | Reduced BFS calls, optimized deadlock detection, minimized initialization |
| Slow path reconstruction | Used struct-based state tracking (discussed with classmates) |
| Serializing to parallel caused issues | Incremental debugging |

## Pthread vs OpenMP

### Pthread
**Pros**
- Direct control over thread lifecycle, synchronization (mutex, condition variable), scheduling
- Low overhead; no compiler abstraction hiding details
- Suitable for fine-grained custom thread pools or special scheduling

**Cons**
- Manual resource management; easy to introduce race conditions or deadlocks
- Very difficult to debug

### OpenMP
**Pros**
- Minimal code change with `#pragma omp parallel for`
- Fast to develop; ideal for loop-heavy numerical/scientific computations

**Cons**
- Low controllability; cannot fine-tune thread behavior like Pthread
- Unsuitable for complex synchronization patterns (e.g., socket servers, multi-stage pipelines)
- Auto-scheduling may be less precise than hand-tuned Pthread; watch for false sharing and thread oversubscription
