"""Exercise scheduled index preparation through the real CLI job loop."""
import pytest
from staar_phewas import cli
from staar_phewas.masks import NONCODING_CATEGORIES


@pytest.mark.parametrize('specs,expected', [
    ([('individual', {}), ('singlevariant', {}), ('coding', {'start': 1, 'end': 3})], []),
    ([('noncoding', {'start': 1, 'end': 3}), ('ncrna', {'start': 1, 'end': 3})], []),
    ([('ncrna', {}), ('ncrna', {})], ['ncRNA']),
    ([('noncoding', {'category': 'upstream'}), ('noncoding', {'category': 'UTR'}),
      ('noncoding', {'category': 'upstream', 'include_ncrna': True})], ['upstream', 'UTR', 'ncRNA']),
    ([('coding', {'start': 1, 'end': 3}), ('noncoding', {}), ('ncrna', {}),
      ('individual', {})], [*NONCODING_CATEGORIES, 'ncRNA']),
])
@pytest.mark.parametrize('packed_directory', [None, 'local-reader-build'])
def test_cli_prepares_only_scheduled_index_once_and_preserves_order(tmp_path, monkeypatch, specs, expected, packed_directory):
    events = []
    class Model:
        n = 3; n_pheno = 1; family = 'gaussian'; matmul_mode = 'fp64'
        def set_matmul_mode(self, mode): self.matmul_mode = mode
    class Reader:
        reader_metadata = {}
        def __init__(self, *args, **kwargs):
            assert kwargs == ({} if packed_directory is None else {'packed_reader_directory': packed_directory})
            events.append(('open',))
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Pipeline:
        def __init__(self, *args, **kwargs): events.append(('pipeline',))
        def prepare_annotation_index(self, chromosome, **kwargs):
            events.append(('index', kwargs['categories']))
            assert kwargs['include_ncrna'] is False
        def __getattr__(self, name):
            if name in {'coding', 'noncoding', 'ncrna', 'individual', 'singlevariant'}:
                def job(**kwargs):
                    events.append(('job', name, kwargs))
                    return [[]]
                return job
            raise AttributeError(name)
    monkeypatch.setattr(cli, 'GaussianNullModel', Model)
    monkeypatch.setattr(cli, 'load_null_model', lambda *args, **kwargs: Model())
    monkeypatch.setattr(cli, 'SeqArrayGDS', Reader)
    monkeypatch.setattr(cli, 'PheWASPipeline', Pipeline)
    monkeypatch.setattr(cli, '_bind_gds_samples', lambda *args: None)
    # A missing file detects accidental promoter input I/O in non-promoter jobs.
    promoter = tmp_path / 'promoters.tsv'
    if any(c.startswith('promoter_') for c in expected): promoter.write_text('1\t1\t3\n')
    jobs = [dict(kind=kind, arguments=arguments, output=str(tmp_path / f'{i}.Rdata'))
            for i, (kind, arguments) in enumerate(specs)]
    config = dict(matmul_mode='fp64', precision_control=True,
        phenotypes=[dict(name='trait', model='cache.npz')],
        chromosomes=[dict(name=1, gds='input.gds', jobs=jobs,
            annotation_index=dict(promoter_intervals_file=str(promoter), include_ncrna=True))])
    if packed_directory is not None: config['packed_reader_directory'] = packed_directory
    report = cli.run_configuration(config, device='cpu')
    assert [e for e in events if e[0] == 'index'] == ([('index', expected)] if expected else [])
    assert [e[1] for e in events if e[0] == 'job'] == [kind for kind, _ in specs]
    assert len([e for e in events if e[0] == 'open']) == 1
    assert len([e for e in events if e[0] == 'pipeline']) == 1
    assert [job['kind'] for job in report['jobs']] == [kind for kind, _ in specs]
    if not expected: assert report['index_preparation_seconds'] == 0


@pytest.mark.parametrize('value', ['', ' ', True, 0, [], {}])
def test_packed_reader_directory_rejects_invalid_configuration_before_inputs(value):
    with pytest.raises(ValueError, match='packed_reader_directory'):
        cli.run_configuration(dict(matmul_mode='fp64', precision_control=True,
            phenotypes=[], chromosomes=[], packed_reader_directory=value), device='cpu')
