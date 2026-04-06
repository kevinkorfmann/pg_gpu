"""
Benchmark ibs_longest_segment CUDA kernel.

Compares:
  1. pg_gpu CUDA kernel (ibs_longest_segment)
  2. CuPy parallel-prefix baseline
  3. Naive NumPy reference (small sizes only, for correctness)

Usage:
    pixi run python debug/bench_ibs_longest.py
"""

import json
import time
import numpy as np
import cupy as cp

from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.relatedness import ibs_longest_segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_haplotypes(n_haps, n_snps, freq=0.3, seed=42):
    """Generate random founder-model haplotype matrix."""
    rng = np.random.default_rng(seed)
    hap = (rng.random((n_haps, n_snps)) < freq).astype(np.int8)
    return hap


def naive_ibs_longest_numpy(hap):
    """Brute-force NumPy reference: O(n_pairs * n_snps)."""
    n_haps, n_snps = hap.shape
    n_pairs = n_haps * (n_haps - 1) // 2
    result = np.zeros(n_pairs, dtype=np.int32)
    idx = 0
    for i in range(n_haps):
        for j in range(i + 1, n_haps):
            max_run = 0
            cur_run = 0
            for s in range(n_snps):
                if hap[i, s] == hap[j, s]:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0
            result[idx] = max_run
            idx += 1
    return result


def cupy_parallel_prefix_ibs_longest(hap_gpu):
    """CuPy-based parallel-prefix longest IBS segment.

    For each pair, computes element-wise equality, then finds the
    longest run of True values using a reduce approach.
    This is a vectorized baseline but uses O(n_pairs * n_snps) memory.
    """
    n_haps, n_snps = hap_gpu.shape
    n_pairs = n_haps * (n_haps - 1) // 2

    # Build pair indices
    ii, jj = [], []
    for i in range(n_haps):
        for j in range(i + 1, n_haps):
            ii.append(i)
            jj.append(j)
    ii = cp.array(ii, dtype=cp.int32)
    jj = cp.array(jj, dtype=cp.int32)

    # Element-wise IBS match matrix: (n_pairs, n_snps)
    matches = (hap_gpu[ii] == hap_gpu[jj]).astype(cp.int32)

    # Longest run via cumulative approach:
    # At each mismatch, reset counter. Use iterative doubling.
    # Simple approach: scan with running counter per row.
    # CuPy lacks a direct "longest run" primitive, so use a sequential
    # kernel-friendly approach: cumsum-based segment detection.

    # Mark segment boundaries (0 at mismatches)
    # cumsum of matches, reset at zeros
    # longest run = max segment length per row
    # This is tricky to vectorize fully; use a simple iterative approach.
    max_runs = cp.zeros(n_pairs, dtype=cp.int32)
    cur_runs = cp.zeros(n_pairs, dtype=cp.int32)
    for s in range(n_snps):
        col = matches[:, s]
        cur_runs = cp.where(col > 0, cur_runs + 1, 0)
        max_runs = cp.maximum(max_runs, cur_runs)

    return max_runs.get()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def verify_correctness():
    """Check CUDA kernel against naive NumPy for small data."""
    print("Verifying correctness (n_haps=20, n_snps=200) ...")
    hap_np = make_haplotypes(20, 200, seed=123)

    # Naive reference
    ref = naive_ibs_longest_numpy(hap_np)

    # CUDA kernel via pg_gpu
    positions = np.arange(200, dtype=np.int64)
    hm = HaplotypeMatrix(hap_np, positions)
    hm.transfer_to_gpu()
    result = ibs_longest_segment(hm)

    assert np.array_equal(ref, result), (
        f"Mismatch: max abs diff = {np.max(np.abs(ref - result))}"
    )
    print(f"  PASSED. {len(ref)} pairs, max segment = {ref.max()}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_cuda_kernel(hap_np, positions, n_reps=5):
    """Time ibs_longest_segment (CUDA kernel)."""
    hm = HaplotypeMatrix(hap_np.copy(), positions.copy())
    hm.transfer_to_gpu()
    # Warmup
    _ = ibs_longest_segment(hm)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(n_reps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = ibs_longest_segment(hm)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def bench_cupy_prefix(hap_np, n_reps=5):
    """Time CuPy parallel-prefix baseline."""
    hap_gpu = cp.asarray(hap_np, dtype=cp.int8)
    # Warmup
    _ = cupy_parallel_prefix_ibs_longest(hap_gpu)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(n_reps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = cupy_parallel_prefix_ibs_longest(hap_gpu)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def main():
    verify_correctness()

    n_haps = 100
    snp_sizes = [5_000, 10_000, 20_000, 50_000, 100_000]
    n_reps = 5
    results = {}

    print(f"\nBenchmark: n_haps={n_haps}, {n_reps} reps (median)")
    print(f"{'n_snps':>10}  {'CUDA kernel (s)':>16}  {'CuPy prefix (s)':>16}  {'speedup':>8}")
    print("-" * 60)

    for n_snps in snp_sizes:
        hap_np = make_haplotypes(n_haps, n_snps, seed=42)
        positions = np.arange(n_snps, dtype=np.int64)

        t_cuda = bench_cuda_kernel(hap_np, positions, n_reps)

        # CuPy prefix can OOM for large n_pairs * n_snps; skip if too large
        n_pairs = n_haps * (n_haps - 1) // 2
        mem_needed = n_pairs * n_snps * 4  # int32
        free_mem = cp.cuda.Device().mem_info[0]
        if mem_needed < free_mem * 0.5:
            t_prefix = bench_cupy_prefix(hap_np, n_reps)
        else:
            t_prefix = float('nan')

        speedup = t_prefix / t_cuda if not np.isnan(t_prefix) else float('nan')
        print(f"{n_snps:>10}  {t_cuda:>16.6f}  {t_prefix:>16.6f}  {speedup:>8.1f}x")

        results[str(n_snps)] = {
            "cuda_kernel_s": round(t_cuda, 6),
            "cupy_prefix_s": round(t_prefix, 6) if not np.isnan(t_prefix) else None,
            "speedup": round(speedup, 2) if not np.isnan(speedup) else None,
        }

    results["meta"] = {
        "n_haps": n_haps,
        "n_reps": n_reps,
        "snp_sizes": snp_sizes,
    }

    out_path = "debug/bench_ibs_longest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
