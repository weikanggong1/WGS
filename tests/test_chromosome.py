"""Tutorial schedule coverage and whole-catalog preservation."""
import pytest
from staar_phewas.chromosome import chromosome_configuration, CODING_MASKS, NONCODING_MASKS


def configuration():
    return {'phenotypes': [{'name': 'continuous_trait', 'model': 'private/model.npz'}],
            'gds': 'private/chromosome.gds', 'output_directory': 'runs/full',
            'annotation_catalog': 'private/catalog.json',
            'promoter_intervals_file': 'private/promoters.tsv'}


def manifest():
    return {'chromosome': '21',
            'genes_info': [{'gene_name': 'G'+str(k), 'start': k+1, 'end': k+10} for k in range(51)],
            'ncRNA_genes': [{'gene_name': 'RNA'+str(k)} for k in range(101)],
            'start_loc': 10, 'end_loc': 20_000_010,
            'array_offsets': {'coding': 100, 'noncoding': 100, 'ncrna': 200, 'individual': 300}}


def test_complete_catalog_masks_and_global_batch_numbers():
    result = chromosome_configuration(configuration(), manifest())
    jobs = result['chromosomes'][0]['jobs']
    coding = [job for job in jobs if job['kind'] == 'coding']
    noncoding = [job for job in jobs if job['kind'] == 'noncoding']
    rna = [job for job in jobs if job['kind'] == 'ncrna']
    assert len(coding) == len(noncoding) == 51 and len(rna) == 101
    assert coding[49]['output'].endswith('_Coding_101.Rdata')
    assert coding[50]['output'].endswith('_Coding_102.Rdata')
    assert rna[99]['output'].endswith('_ncRNA_201.Rdata')
    assert rna[100]['output'].endswith('_ncRNA_202.Rdata')
    assert all(job['arguments']['category'] == 'all_categories_incl_ptv' for job in coding)
    assert all(job['layout'] == 'base' for job in jobs)
    assert len(CODING_MASKS) == 7 and len(NONCODING_MASKS) == 8


def test_exact_multiple_region_boundary_is_not_dropped():
    result = chromosome_configuration(configuration(), manifest())
    individual = [job for job in result['chromosomes'][0]['jobs'] if job['kind'] == 'individual']
    ranges = [(job['arguments']['start'], job['arguments']['end']) for job in individual]
    assert ranges == [(10, 10_000_009), (10_000_010, 20_000_009), (20_000_010, 20_000_010)]
    assert all(ranges[k][1] + 1 == ranges[k+1][0] for k in range(len(ranges)-1))
    assert individual[-1]['output'].endswith('_Individual_Analysis_303.Rdata')


def test_empty_catalog_does_not_invent_gene_jobs():
    source = manifest()
    source['genes_info'] = []; source['ncRNA_genes'] = []
    result = chromosome_configuration(configuration(), source)
    assert {job['kind'] for job in result['chromosomes'][0]['jobs']} == {'individual'}
    assert result['coverage']['coding_genes'] == result['coverage']['ncrna_genes'] == 0


def test_manifest_rejects_incomplete_catalog():
    source = manifest()
    source['number_coding_genes'] = 52
    with pytest.raises(ValueError, match='complete gene catalog'):
        chromosome_configuration(configuration(), source)


@pytest.mark.parametrize('settings', [{'family': 'binomial'}, {'joint_mode': 'ordinary'}])
def test_univariate_continuous_scope_is_explicit(settings):
    config = configuration()
    config['phenotypes'][0].update(settings)
    with pytest.raises(ValueError, match='one continuous phenotype'):
        chromosome_configuration(config, manifest())


def test_current_chromosome_plan_uses_native_unsplit_tf32():
    expanded=chromosome_configuration(configuration(),manifest())
    assert expanded['matmul_mode']=='tf32' and expanded['tf32_split_k']==0
    assert expanded['statistics_execution']=='serial'
    assert 'tf32_binned_tile_shape' not in expanded and 'tf32_binned_fused_small' not in expanded


@pytest.mark.parametrize('enabled,width', [(False, 512), (True, 1024)])
def test_single_batch_controls_reach_chromosome_execution(enabled, width, monkeypatch):
    from staar_phewas import cli
    from staar_phewas.chromosome import run_chromosome
    config = configuration()
    config.update(manifest=manifest(), single_batch_optimization=enabled,
                  individual_effective_block_size=width)
    baseline = chromosome_configuration(configuration(), manifest())
    planned = chromosome_configuration(config, config['manifest'])
    assert planned['single_batch_optimization'] is enabled
    assert planned['individual_effective_block_size'] == width
    assert planned['chromosomes'] == baseline['chromosomes']
    called = []
    def execute(settings, **kwargs):
        called.append(settings)
        return {}
    monkeypatch.setattr(cli, 'run_configuration', execute)
    run_chromosome(config, device='cuda:0')
    assert called[0]['single_batch_optimization'] is enabled
    assert called[0]['individual_effective_block_size'] == width


@pytest.mark.parametrize('obsolete', [ {'matmul_mode':'tf32_binned'}, {'matmul_mode':'tf32x3'},
    {'tf32_split_k':1024}, {'tf32_binned_fused_small':True}, {'tf32_binned_tile_shape':[32,64]} ])
def test_chromosome_plan_rejects_removed_reconstruction_controls(obsolete):
    cfg=configuration();cfg.update(obsolete)
    with pytest.raises((ValueError,TypeError)):
        chromosome_configuration(cfg,manifest())


@pytest.mark.parametrize('requested',['auto','torch','cusolver_batched'])
def test_solver_choice_survives_complete_plan_and_run_chromosome(requested,monkeypatch):
    from staar_phewas import cli
    from staar_phewas.chromosome import run_chromosome
    cfg=configuration();cfg['manifest']=manifest();cfg['weighted_eigensolver']=requested
    baseline=chromosome_configuration(configuration(),manifest())
    expanded=chromosome_configuration(cfg,cfg['manifest'])
    assert expanded['weighted_eigensolver']==requested
    assert expanded['chromosomes']==baseline['chromosomes']
    assert expanded['coverage']==baseline['coverage']
    called=[]
    def execute(configuration,**kwargs):
        called.append((configuration,kwargs))
        return {'weighted_eigensolver_execution':cli._weighted_eigensolver_settings(
            configuration,configuration['matmul_mode'],kwargs['device'])}
    monkeypatch.setattr(cli,'run_configuration',execute)
    report=run_chromosome(cfg,device='cuda:0')
    assert called[0][0]['weighted_eigensolver']==requested
    assert report['weighted_eigensolver_execution']['requested']==requested
    assert report['weighted_eigensolver_execution']['effective']==('torch' if requested=='torch' else 'cusolver_batched')
    assert report['coverage']==baseline['coverage']


@pytest.mark.parametrize('invalid',[None,True,[],'unsupported'])
def test_run_chromosome_keeps_invalid_choice_so_cli_rejects_before_analysis(invalid,monkeypatch):
    from staar_phewas import cli
    from staar_phewas.chromosome import run_chromosome
    cfg=configuration();cfg['manifest']=manifest();cfg['weighted_eigensolver']=invalid
    monkeypatch.setattr(cli,'_run_configuration',lambda *args,**kwargs:pytest.fail('invalid solver entered analysis'))
    with pytest.raises(ValueError,match='weighted_eigensolver'):
        run_chromosome(cfg,device='cuda')


def test_unset_chromosome_solver_uses_cli_auto_default(monkeypatch):
    from staar_phewas import cli
    from staar_phewas.chromosome import run_chromosome
    cfg=configuration();cfg['manifest']=manifest()
    assert 'weighted_eigensolver' not in chromosome_configuration(cfg,cfg['manifest'])
    def execute(configuration,**kwargs):
        return {'weighted_eigensolver_execution':cli._weighted_eigensolver_settings(
            configuration,configuration['matmul_mode'],kwargs['device'])}
    monkeypatch.setattr(cli,'run_configuration',execute)
    report=run_chromosome(cfg,device='cuda:0')
    assert report['weighted_eigensolver_execution']['requested']=='auto'
    assert report['weighted_eigensolver_execution']['effective']=='cusolver_batched'
