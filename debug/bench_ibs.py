"""Benchmark ibs() vs ibs_v2() (matmul vs fused CUDA kernel)."""

import json
import time
import numpy as np
import cupy as cp

from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.relatedness import ibs, ibs_v2


def make_haplotype_matrix(n_haps, n_snps, seed=42):
    rng = np.random.default_rng(seed)
    founders = rng.integers(0, 2, size=(5, n_snps), dtype=np.int8)
    assignments = rng.integers(0, 5, size=n_haps)
    haps = founders[assignments].copy()
    mutations = rng.random(size=(n_haps, n_snps)) < 0.02
    haps ^= mutations.astype(np.int8)
    positions = np.arange(n_snps)
    hm = HaplotypeMatrix(haps, positions)
    hm.transfer_to_gpu()
    return hm


def bench(func, hm, n_reps=5):
    func(hm)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(n_reps):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        func(hm)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def main():
    configs = [
        (100, 5000),
        (100, 10000),
        (100, 50000),
        (100, 100000),
        (200, 10000),
        (200, 50000),
        (200, 100000),
    ]
    n_reps = 5

    # Correctness check
    print("Correctness check (n_haps=100, n_snps=5000)")
    hm = make_haplotype_matrix(100, 5000, seed=99)
    r_orig = ibs(hm)
    r_v2 = ibs_v2(hm)
    max_diff = np.max(np.abs(r_orig - r_v2))
    print(f"  max absolute diff: {max_diff:.2e}")
    print(f"  PASS" if max_diff < 1e-10 else f"  FAIL")
    print()

    # Benchmarks
    print(f"{'n_haps':>7} {'n_snps':>8} | {'ibs (ms)':>10} {'ibs_v2 (ms)':>12} {'speedup':>8}")
    print("-" * 55)

    results = []
    for n_haps, n_snps in configs:
        hm = make_haplotype_matrix(n_haps, n_snps)

        t_orig = bench(ibs, hm, n_reps)
        t_v2 = bench(ibs_v2, hm, n_reps)
        speedup = t_orig / t_v2

        print(f"{n_haps:>7} {n_snps:>8} | {t_orig*1000:>10.2f} {t_v2*1000:>12.2f} {speedup:>7.1f}x")

        results.append({
            "n_haps": n_haps, "n_snps": n_snps,
            "ibs_ms": round(t_orig * 1000, 2),
            "ibs_v2_ms": round(t_v2 * 1000, 2),
            "speedup": round(speedup, 2),
        })

    with open("debug/bench_ibs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to debug/bench_ibs_results.json")


if __name__ == "__main__":
    main()
