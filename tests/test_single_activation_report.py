"""Configuration requests must not claim cached Single work was executed."""
import pytest
from staar_phewas import cli


@pytest.mark.parametrize('readers,blocks,columns', [
    ([], 0, 0),
    ([{'reader_backend': 'native_sdk'}], 0, 0),
    ([{'analysis_cache': {'single_effective_blocks': 0}}], 0, 0),
    ([{'analysis_cache': {'single_effective_blocks': 2,
                         'single_effective_columns': 1500}},
      {'analysis_cache': {'single_effective_blocks': 1,
                         'single_effective_columns': 200}}], 3, 1700),
])
def test_report_distinguishes_request_from_executed_batches(readers, blocks, columns, monkeypatch):
    monkeypatch.setattr(cli, '_run_configuration',
                        lambda *args, **kwargs: {'genotype_readers': readers, 'total_seconds': 0.0})
    report = cli.run_configuration({'matmul_mode': 'tf32',
                                   'single_batch_optimization': True,
                                   'individual_effective_block_size': 1024}, device='cpu')
    execution = report['single_optimization_configuration']
    assert execution['requested_batch_optimization'] is True
    assert execution['activated'] is (blocks > 0)
    assert execution['actual_effective_blocks'] == blocks
    assert execution['actual_effective_columns'] == columns


def test_public_batch_module_reexports_the_implementation():
    from torchstaar.cache_runtime.single_batches import iter_effective_minor_blocks
    from staar_phewas.cache_runtime.single_batches import iter_effective_minor_blocks as implementation
    assert iter_effective_minor_blocks is implementation
