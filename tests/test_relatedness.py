"""Tests for pg_gpu.relatedness (GRM, IBS, and IBS longest segment)."""

import time
import numpy as np
import pytest
from pg_gpu import HaplotypeMatrix, relatedness


@pytest.fixture
def small_haplotype_matrix():
    """Small test dataset: 6 haplotypes (3 diploids), 20 variants."""
    np.random.seed(42)
    hap = np.random.randint(0, 2, size=(6, 20)).astype(np.int8)
    positions = np.arange(20) * 1000
    return HaplotypeMatrix(hap, positions)


def _reference_grm(hap):
    """Compute GRM from haplotypes using numpy (reference implementation)."""
    n_ind = hap.shape[0] // 2
    g = hap[:n_ind, :] + hap[n_ind:, :]
    g = g.astype(float)
    p = g.mean(axis=0) / 2
    poly = (p > 0) & (p < 1)
    g_p = g[:, poly]
    p_p = p[poly]
    centered = g_p - 2 * p_p
    scale = np.sqrt(2 * p_p * (1 - p_p))
    std = centered / scale
    return (std @ std.T) / poly.sum()


def _reference_ibs(hap):
    """Compute IBS from haplotypes using numpy (reference implementation)."""
    n_ind = hap.shape[0] // 2
    g = hap[:n_ind, :] + hap[n_ind:, :]
    n_snps = g.shape[1]
    ibs_mat = np.eye(n_ind)
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            ibs_val = (2 - np.abs(g[i] - g[j])).sum() / (2 * n_snps)
            ibs_mat[i, j] = ibs_val
            ibs_mat[j, i] = ibs_val
    return ibs_mat


class TestGRM:
    def test_shape(self, small_haplotype_matrix):
        grm = relatedness.grm(small_haplotype_matrix)
        assert grm.shape == (3, 3)

    def test_symmetric(self, small_haplotype_matrix):
        grm = relatedness.grm(small_haplotype_matrix)
        np.testing.assert_allclose(grm, grm.T)

    def test_matches_reference(self, small_haplotype_matrix):
        grm_pg = relatedness.grm(small_haplotype_matrix)
        hap = small_haplotype_matrix.haplotypes
        if hasattr(hap, 'get'):
            hap = hap.get()
        grm_ref = _reference_grm(hap)
        np.testing.assert_allclose(grm_pg, grm_ref, atol=1e-10)

    def test_identical_individuals(self):
        """Two identical individuals should have GRM off-diagonal = diagonal."""
        # pg_gpu layout: [allele1_ind0, allele1_ind1, allele2_ind0, allele2_ind1]
        hap = np.array([[0, 1, 0, 1, 0],   # ind0 allele1
                         [0, 1, 0, 1, 0],   # ind1 allele1 (same as ind0)
                         [1, 0, 1, 0, 1],   # ind0 allele2
                         [1, 0, 1, 0, 1]], dtype=np.int8)  # ind1 allele2 (same)
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        grm = relatedness.grm(hm)
        np.testing.assert_allclose(grm[0, 1], grm[0, 0], atol=1e-10)

    def test_returns_numpy(self, small_haplotype_matrix):
        grm = relatedness.grm(small_haplotype_matrix)
        assert isinstance(grm, np.ndarray)


class TestIBS:
    def test_shape(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        assert ibs_mat.shape == (3, 3)

    def test_diagonal_is_one(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        np.testing.assert_allclose(ibs_mat.diagonal(), 1.0)

    def test_symmetric(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        np.testing.assert_allclose(ibs_mat, ibs_mat.T)

    def test_range(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        assert np.all(ibs_mat >= 0)
        assert np.all(ibs_mat <= 1)

    def test_matches_reference(self, small_haplotype_matrix):
        ibs_pg = relatedness.ibs(small_haplotype_matrix)
        hap = small_haplotype_matrix.haplotypes
        if hasattr(hap, 'get'):
            hap = hap.get()
        ibs_ref = _reference_ibs(hap)
        np.testing.assert_allclose(ibs_pg, ibs_ref, atol=1e-10)

    def test_identical_individuals(self):
        # pg_gpu layout: [allele1_ind0, allele1_ind1, allele2_ind0, allele2_ind1]
        hap = np.array([[0, 1, 0, 1, 0],   # ind0 allele1
                         [0, 1, 0, 1, 0],   # ind1 allele1
                         [1, 0, 1, 0, 1],   # ind0 allele2
                         [1, 0, 1, 0, 1]], dtype=np.int8)  # ind1 allele2
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        ibs_mat = relatedness.ibs(hm)
        assert ibs_mat[0, 1] == 1.0

    def test_completely_different(self):
        # ind0 = 0/0 at all sites, ind1 = 2/2 at all sites
        hap = np.array([[0, 0, 0, 0, 0],   # ind0 allele1
                         [1, 1, 1, 1, 1],   # ind1 allele1
                         [0, 0, 0, 0, 0],   # ind0 allele2
                         [1, 1, 1, 1, 1]], dtype=np.int8)  # ind1 allele2
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        ibs_mat = relatedness.ibs(hm)
        assert ibs_mat[0, 1] == 0.0

    def test_returns_numpy(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        assert isinstance(ibs_mat, np.ndarray)


def _reference_ibs_longest_segment(hap):
    """Naive NumPy reference: longest contiguous IBS per haplotype pair."""
    n_hap, n_var = hap.shape
    n_pairs = n_hap * (n_hap - 1) // 2
    result = np.zeros(n_pairs, dtype=np.int32)
    idx = 0
    for i in range(n_hap):
        for j in range(i + 1, n_hap):
            max_run = 0
            cur_run = 0
            for s in range(n_var):
                if hap[i, s] < 0 or hap[j, s] < 0:
                    cur_run = 0
                elif hap[i, s] == hap[j, s]:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0
            result[idx] = max_run
            idx += 1
    return result


class TestIBSLongestSegment:
    def test_output_shape(self, small_haplotype_matrix):
        result = relatedness.ibs_longest_segment(small_haplotype_matrix)
        n_hap = 6
        n_pairs = n_hap * (n_hap - 1) // 2
        assert result.shape == (n_pairs,)

    def test_returns_numpy_int32(self, small_haplotype_matrix):
        result = relatedness.ibs_longest_segment(small_haplotype_matrix)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32

    def test_nonnegative(self, small_haplotype_matrix):
        result = relatedness.ibs_longest_segment(small_haplotype_matrix)
        assert np.all(result >= 0)

    def test_matches_reference(self, small_haplotype_matrix):
        result = relatedness.ibs_longest_segment(small_haplotype_matrix)
        hap = small_haplotype_matrix.haplotypes
        if hasattr(hap, 'get'):
            hap = hap.get()
        ref = _reference_ibs_longest_segment(hap)
        np.testing.assert_array_equal(result, ref)

    def test_matches_reference_larger(self):
        np.random.seed(123)
        hap = np.random.randint(0, 2, size=(40, 500)).astype(np.int8)
        positions = np.arange(500) * 100
        hm = HaplotypeMatrix(hap, positions)
        result = relatedness.ibs_longest_segment(hm)
        ref = _reference_ibs_longest_segment(hap)
        np.testing.assert_array_equal(result, ref)

    def test_identical_haplotypes(self):
        hap = np.array([[0, 1, 0, 1, 1],
                         [0, 1, 0, 1, 1]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        result = relatedness.ibs_longest_segment(hm)
        assert result[0] == 5

    def test_completely_different(self):
        hap = np.array([[0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        result = relatedness.ibs_longest_segment(hm)
        assert result[0] == 0

    def test_single_match(self):
        hap = np.array([[0, 1, 0, 1, 0],
                         [1, 1, 1, 0, 1]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        result = relatedness.ibs_longest_segment(hm)
        assert result[0] == 1

    def test_known_segments(self):
        # Match at positions 2,3,4 (run of 3), then mismatch, then match at 6 (run of 1)
        hap = np.array([[0, 1, 0, 0, 0, 1, 1, 0],
                         [1, 0, 0, 0, 0, 0, 1, 1]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(8) * 100)
        result = relatedness.ibs_longest_segment(hm)
        assert result[0] == 3

    def test_missing_data_breaks_run(self):
        hap = np.array([[0, 0, 0, 0, 0],
                         [0, 0, -1, 0, 0]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        result = relatedness.ibs_longest_segment(hm)
        # Missing at position 2 breaks the run: max is 2 (positions 0-1 or 3-4)
        assert result[0] == 2

    def test_missing_data_exclude_mode(self):
        hap = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, -1, 0, 0, 0]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(6) * 100)
        result = relatedness.ibs_longest_segment(hm, missing_data='exclude')
        # After excluding site 2, remaining 5 sites all match -> run of 5
        assert result[0] == 5

    def test_multiple_pairs(self):
        hap = np.array([[0, 0, 0, 0],   # hap 0
                         [0, 0, 1, 1],   # hap 1
                         [1, 1, 1, 1]], dtype=np.int8)  # hap 2
        hm = HaplotypeMatrix(hap, np.arange(4) * 100)
        result = relatedness.ibs_longest_segment(hm)
        # Pair (0,1): match at 0,1 then differ -> 2
        # Pair (0,2): all differ -> 0
        # Pair (1,2): match at 2,3 -> 2
        assert result[0] == 2  # (0,1)
        assert result[1] == 0  # (0,2)
        assert result[2] == 2  # (1,2)


class TestIBSLongestSegmentPerformance:
    @pytest.fixture
    def large_dataset(self):
        np.random.seed(42)
        rng = np.random.default_rng(42)
        n_haps, n_snps = 100, 50000
        founders = rng.integers(0, 2, size=(5, n_snps), dtype=np.int8)
        assignments = rng.integers(0, 5, size=n_haps)
        haps = founders[assignments].copy()
        mutations = rng.random(size=(n_haps, n_snps)) < 0.02
        haps ^= mutations.astype(np.int8)
        positions = np.arange(n_snps) * 100
        return HaplotypeMatrix(haps, positions)

    def test_performance(self, large_dataset):
        # Warmup
        relatedness.ibs_longest_segment(large_dataset)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            relatedness.ibs_longest_segment(large_dataset)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        median_ms = np.median(times) * 1000

        print(f"\nibs_longest_segment: 100 haplotypes x 50k variants")
        print(f"  median: {median_ms:.2f} ms ({4950} pairs)")
        # Should be well under 100ms for this size
        assert median_ms < 100, f"Too slow: {median_ms:.1f} ms"
