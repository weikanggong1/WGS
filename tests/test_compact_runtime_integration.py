"""Population/subset compact reuse with the actual standalone reader."""
import numpy as np
import pytest

from staar_phewas.cache_runtime.portable import PortableCachedGDS
from test_cache_only_workflow import dataset


@pytest.mark.parametrize("prefetch_depth", [0, 2])
def test_standalone_compact_cold_and_warm_keep_exact_subset_values(tmp_path, prefetch_depth):
    root, container, _, _ = dataset(tmp_path)
    directory = root / ".cohort_compact_v1"
    variants = np.array([10, 1, 4, 0, 11, 3], dtype=np.int64)
    samples = np.array([6, 1, 3, 0], dtype=np.int64)

    def read():
        with PortableCachedGDS(container, device="cpu", prepared_cache_directory=directory,
                prepared_cache_max_bytes=2**20, prefetch_depth=prefetch_depth) as reader:
            blocks = list(reader.iter_minor_blocks(variants, samples, block_size=3, minimum_mac=0))
            data = [(block.trait_dense(np.arange(len(samples)))[0].copy(),
                     block.variant_indices.copy(), block.initial_mac().copy()) for block in blocks]
            return data, dict(reader.reader_metadata["analysis_cache"])

    cold, cold_metrics = read()
    warm, warm_metrics = read()
    for left, right in zip(cold, warm):
        for actual, expected in zip(left, right):
            np.testing.assert_array_equal(actual, expected)
    assert cold_metrics["cohort_compact_writes"] == 2
    assert warm_metrics["cohort_compact_hits"] == 2
    assert warm_metrics["frame_loads"] == 0
    with PortableCachedGDS(container, device="cpu", prepared_cache_directory=directory,
            prepared_cache_max_bytes=2**20, prefetch_depth=prefetch_depth) as reader:
        list(reader.iter_minor_blocks(variants, samples[::-1], block_size=3, minimum_mac=0))
        metrics = reader.reader_metadata["analysis_cache"]
        assert metrics["cohort_compact_misses"] == 2
        assert metrics["frame_loads"] > 0
