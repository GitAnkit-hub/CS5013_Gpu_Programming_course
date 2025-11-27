# BANG Exact Distance - GPU Optimized Implementation

## Project Overview

This project implements GPU optimizations for the BANG (Billion-scale Approximate Nearest Neighbor on GPUs) algorithm. The optimizations achieve **37.0% speedup on SIFT100M** and **29-35% speedup on DEEP100M** while maintaining identical recall accuracy.

### Team Information
- **Students:** Ankit Agrawal, Zaki Haseeb Kazi, Khushal Bhalia
- **Roll Numbers:** CS24MTECH12019, CS25MTECH11021, CS25MTECH14008
- **Course:** CS5013 GPU Programming
- **Institution:** Indian Institute of Technology, Hyderabad
- **Date:** November 2025

---

## Optimizations Implemented

### 1. Kernel Fusion
Combined three separate kernels into one fused kernel:
- `neighbor_filtering_new` (bloom filter)
- `compute_neighborDist_par` (L2 distance)
- `compute_BestLSets_par_sort_msort` (sort + merge)

**Benefits:**
- Eliminated 160 kernel launches per search (67% reduction)
- Data stays in shared memory instead of global memory round-trips
- Shared memory is ~100x faster than global memory (20-40 cycles vs 400-800 cycles)
- Reduced 5-10μs overhead per kernel launch, saving 1-2ms total

### 2. Adaptive Bitonic Sort
Replaced Merge sort with optimized Bitonic sort, then reduced BITONIC_SORT_SIZE from 128 to 64 elements.

**Benefits:**
- Initial bitonic sort implementation: 10.9% speedup over merge sort
- 6 sorting stages instead of 7 (25% reduction: 21 vs 28 comparison rounds)
- One fewer __syncthreads() barrier per iteration (80 barriers total saved)
- Smaller working set fits better in L1 cache (512 bytes vs 1024 bytes)
- Less wasted work on padding/sentinel values (FLT_MAX)

### 3. Binary Search Insertion for Overflow
When 65th neighbor exists (rare case when node has exactly R=64 neighbors), use binary search + shift instead of full re-sort.

**Benefits:**
- O(log n) = 6 comparisons vs 64 for another bitonic stage
- Optimizes common case (≤64 neighbors) without penalty
- No extra shared memory required
- Single-thread simplicity avoids complex parallel coordination

### 4. Block Size Tuning (256 → 128 Threads)
Reduced threads per block from 256 to 128.

**Benefits:**
- Increased occupancy: more concurrent blocks per SM improves latency hiding
- 128 threads = 4 warps matches 64-element bitonic sort perfectly (32 comparisons per stage × 4x parallelism)
- Faster __syncthreads() with fewer threads
- Optimal shared memory usage (~2KB per block)

### 5. Race Condition Fix
Fixed race condition in original implementation where `*d_nextIter = false` inside kernel caused issues with 10K concurrent blocks.

**Solution:**
- Kernel only sets `d_nextIter` to true (never false)
- Host calls `cudaMemset(d_nextIter, 0, sizeof(bool))` before each kernel launch

---

## Performance Results

### SIFT100M Dataset (D=128, L=40)

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Wall Clock Time | 22.0 ms | 13.9 ms | **36.8% faster** |
| Fused Kernel Time | 15.8 ms (3 kernels) | 10.88 ms | **31% faster** |
| Throughput | 508K QPS | 720K QPS | **42% higher** |
| Recall@10 | 90.69% | 90.69% | **Identical** |
| Iterations | 80 | 80 | Same |
| Kernel Launches | 240 | 80 | **67% reduction** |

### DEEP100M Dataset (D=96)

| L | Orig Time | Opt Time | Speedup | Orig QPS | Opt QPS | QPS Gain | Recall |
|---|-----------|----------|---------|----------|---------|----------|--------|
| 10 | 11.7 ms | 8.3 ms | **29%** | 850K | 1198K | **41%** | 72.89% |
| 20 | 16.8 ms | 12.0 ms | **29%** | 595K | 830K | **39%** | 83.86% |
| 30 | 22.9 ms | 15.8 ms | **31%** | 435K | 631K | **45%** | 88.82% |
| 40 | 27.1 ms | 19.3 ms | **29%** | 369K | 517K | **40%** | 91.72% |
| 50 | 32.5 ms | 24.8 ms | **24%** | 307K | 402K | **31%** | 93.59% |
| 60 | 39.3 ms | 25.8 ms | **35%** | 254K | 388K | **53%** | 94.84% |
| 80 | 45.0 ms | 32.7 ms | **27%** | 222K | 305K | **37%** | 96.38% |
| 100 | 57.0 ms | 36.9 ms | **35%** | 175K | 270K | **54%** | 97.30% |

**Key Observations:**
- Consistent 29-37% speedup across all L values and both datasets
- Higher gains at large L values (35% speedup, 54% throughput at L=100)
- Dataset independence: similar improvements on SIFT100M (D=128, uint8) and DEEP100M (D=96, float)
- All recall values match exactly, confirming algorithmic correctness

### Optimization Breakdown (SIFT100M, L=40)

Incremental impact of each optimization:

| Optimization | Time Saved | Cumulative Time | Speedup | QPS |
|--------------|------------|-----------------|---------|-----|
| Original (baseline) | - | 22.0 ms | - | 508K |
| + Bitonic Sort (128) | ~2.4 ms | 19.6 ms | 10.9% | 586K |
| + Kernel Fusion | ~2.6 ms | 17.0 ms | 22.7% | 700K |
| + Block Size 128 | ~2.7 ms | 14.3 ms | 35.0% | 710K |
| + Adaptive Bitonic (64) + Binary Insert | ~0.4 ms | **13.9 ms** | **36.8%** | **720K** |

---

## Directory Structure

```
BANG_Optimized/
├── fused_kernel.cu      # Fused kernel implementation
├── parANN.cu            # Modified main algorithm (add your file)
├── parANN.h             # Header with defines (add your file)
├── main.cu              # Entry point (add your file)
├── Makefile             # Build configuration
├── run_sift100m.sh      # Run script for SIFT100M
├── run_deep100m.sh      # Run script for DEEP100M
├── expected_output.txt  # Sample expected output
├── test_cases.txt       # Test cases documentation
├──	 report.pdf	      		  # Report of the project
└── README.md            # This file
```

---

## Prerequisites

- CUDA Toolkit (tested with CUDA 11.x+)
- NVIDIA GPU with compute capability sm_80 (A100, Quadro, etc.)
- GCC with OpenMP support
- Dataset files (SIFT100M or DEEP100M)

---

## Dataset Configuration

### For SIFT100M
Edit `parANN.h` lines 38-40:
```cpp
#define SIFT100M
#define L 40
#define CHUNKS 64
```

### For DEEP100M
Edit `parANN.h` lines 38-40:
```cpp
#define DEEP100M
#define L 10    // Can be 10, 20, 30, 40, 50, 60, 80, 100, 120, 160
#define CHUNKS 96
```

---

## Build Instructions

```bash
# Clean previous build
make clean

# Build optimized version
make

# Or rebuild from scratch
make rebuild
```

---

## Running the Program

### Using Run Scripts

**SIFT100M:**
```bash
chmod +x run_sift100m.sh
./run_sift100m.sh
```

**DEEP100M:**
```bash
chmod +x run_deep100m.sh
./run_deep100m.sh
```

### Manual Execution

**SIFT100M:**
```bash
./bang_exact \
    /path/to/sift100m_pq_pivots.bin \
    /path/to/sift100m_pq_compressed.bin \
    /path/to/sift100m_graph.bin \
    /path/to/sift100m_query.bin \
    /path/to/sift100m_chunk_offsets.bin \
    /path/to/sift100m_centroid.bin \
    /path/to/sift100m_gndtruth.bin \
    10000 1 256 512 256 10 64 1
```

**DEEP100M:**
```bash
./bang_exact \
    /mnt/hdd_volume2/deep100m96/deep100m_pq_pivots.bin \
    /mnt/hdd_volume2/deep100m96/deep100m_pq_compressed.bin \
    /mnt/hdd_volume2/deep100m96/deep100m_graph.bin \
    /mnt/hdd_volume2/deep100m96/deep100m_query.bin \
    /mnt/hdd_volume2/deep100m96/deep100m_chunk_offsets.bin \
    /mnt/hdd_volume2/deep100m96/deep100m_centroid.bin \
    /mnt/hdd_volume2/deep100m96/deep100m_gndtruth.bin \
    10000 1 256 512 256 10 64 0
```

---

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| num_queries | Number of queries to process | 10000 |
| K4_threads | Threads for compute_parent | 1 |
| K1_threads | Threads for populate_pqDist | 256 |
| K2_threads | Threads for neighborDist | 512 |
| K5_threads | Threads for filtering | 256 |
| recall_at | Recall@K evaluation | 10 (SIFT), 10 (DEEP) |
| cpu_threads | OpenMP threads | 64 |
| capabilities | GPU stats flag | 1 (SIFT), 0 (DEEP) |

---

## Key Code Changes

### 1. Include Fused Kernel (parANN.cu, top)
```cpp
#include "fused_kernel.cu"
```

### 2. Replace Kernel Calls (parANN.cu, ~line 490 and ~574)
```cpp
// Reset flags before kernel launch
gpuErrchk(cudaMemset(d_numNeighbors_query, 0, sizeof(unsigned)*numQueries));
gpuErrchk(cudaMemset(d_nextIter, 0, sizeof(bool)));

// FUSED KERNEL - replaces 3 separate kernels
fused_search_kernel<<<numQueries, 128, 0, streamKernels>>>(
    d_pIndex,
    d_queriesFP,
    d_processed_bit_vec,
    d_parents,
    d_BestLSets,
    d_BestLSetsDist,
    d_BestLSets_visited,
    d_BestLSets_count,
    iter,
    d_nextIter,
    d_numNeighbors_query
);
```

### 3. Bitonic Sort Configuration (fused_kernel.cu)
```cpp
#define BITONIC_SORT_SIZE 64  // 6 stages instead of 7
```

---

## Implementation Details

### Fused Kernel Structure
The fused_search_kernel consists of four phases:

1. **Phase 1 - Neighbor Filtering:** Bloom filter check using hashFn1_fused(), stores filtered neighbors in shm_neighbors[]

2. **Phase 2 - Distance Computation:** 8 threads per neighbor compute L2 distance with warp shuffle reduction

3. **Phase 3 - Sorting:** 6-stage bitonic sort for 64 elements, binary search insertion for 65th if needed

4. **Phase 4 - Merge & Parent Selection:** Parallel merge with BestLSets, select next unvisited parent

### Adaptive Bitonic Sort
- BITONIC_SORT_SIZE = 64 (previously 128)
- 6 outer stages (sort_len = 0 to 5) instead of 7
- Comparison index calculation: `ind = t + (t & bitonic_len_mask)`
- Ascending/descending determined by: `descending = (t >> sort_len) & 0x1`

### Binary Search Insertion
When numNeighbors > 64, the 65th element (index 64) is inserted using upper-bound binary search:
1. Binary search to find insertion position in O(log 64) = 6 comparisons
2. Shift elements from [insertPos...63] to [insertPos+1...64] (backwards iteration to prevent overwrites)
3. Place the new element at insertPos

---

## Output Interpretation

```
iterations = 80                              # Search iterations
(2) avg. time_fused = 0.14 ms               # Average fused kernel time
(3) total time_fused = 10.88 ms             # Total fused kernel time
(7) total transfer_time = 0.69 ms           # CPU-GPU transfer time
Wall Clock Time = 13876 microsec            # Total search time (13.9ms)
Throughput = 720667 QPS                     # Queries per second
Recall@10 = 90.69%                          # Search accuracy
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Illegal memory access" | Ensure `cudaMemset(d_nextIter, 0, ...)` is called before each kernel launch |
| Low throughput on first run | First run is warmup; press 'y' for subsequent runs |
| Recall mismatch | Verify L ≥ recall_at in configuration |
| Compilation errors | Check dataset define (SIFT100M/DEEP100M) matches data path |
| "fused_kernel.cu not found" | Ensure fused_kernel.cu is in same directory as parANN.cu |

---

## Future Work

- **CUDA Graphs:** Capture the entire search as a graph to further reduce launch overhead
- **Persistent Kernels:** Keep kernels running across iterations to eliminate all launch latency
- **Multi-Query Per Block:** Process multiple queries in one block for better resource utilization
- **Warp-Level Bitonic:** For ≤32 elements, use warp shuffle instead of shared memory

---

## References

1. BANG: Billion-scale Approximate Nearest Neighbor Search on GPUs
2. DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
3. NVIDIA CUDA C++ Programming Guide - Shared Memory and Synchronization
4. Batcher, K.E. - Sorting Networks and Their Applications (Bitonic Sort)
5. CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search