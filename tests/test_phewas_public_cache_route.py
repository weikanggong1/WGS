"""Existing population-cache routing never prepares genetic data."""
from pathlib import Path
import pytest
from fudan_wgs_toolkit.run import run_WGS_all

def test_existing_cache_routes_all_traits_to_shared_mixed_workflow(tmp_path, monkeypatch):
    from fudan_wgs_toolkit import phewas_run
    cache=tmp_path/'cache';cache.mkdir();(cache/'cache_dataset.json').write_text('{}')
    calls=[]
    def run(*args, **kwargs):
        calls.append((args,kwargs));return [dict(association_rows={'individual':2,'coding':3,'noncoding':4,'ncrna':1})
                                           for _ in kwargs['gpu_ids']]
    monkeypatch.setattr(phewas_run,'run_phewas',run)
    result=run_WGS_all('phenotypes.csv','covariates.csv',cache,output_directory=tmp_path/'out',
        gpu_ids=tuple(range(8)),memory_limit_gib=None,trait_batch_size=16)
    assert result['completed'] and len(calls)==1 and result['eligible_association_tests']==80
    assert calls[0][0]==('phenotypes.csv','covariates.csv',cache)
    assert calls[0][1]['gpu_ids']==tuple(range(8))
    assert calls[0][1]['continuous_transform']=='paper'
    assert calls[0][1]['trait_batch_size']==16
    assert (tmp_path/'out/report.private.json').is_file()

def test_two_dataset_roots_cannot_silently_change_identity(tmp_path):
    (tmp_path/'cache_dataset.json').write_text('{}');(tmp_path/'dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'):
        run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out')

def test_mixed_pipeline_does_not_silently_override_bound_profiles(tmp_path):
    (tmp_path/'cache_dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='profile'):
        run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out',phenotype_families={'example':'gaussian'})

@pytest.mark.parametrize('option,value',[
    ('memory_limit_gib',20),('memory_limit_gib',-1),('gene_variant_type','SNV'),
    ('covariance_block_size',8192),('long_mask_threshold',-1),('long_mask_rank',256),
    ('seed',0),('single_mac_cutoff',1),('single_group_variants',1024),
    ('single_region_size',1),('individual_effective_block_size',512),
    ('device_cache_bytes',0),('compact_cache_bytes',0),('metadata_cache_bytes',0)])
def test_mixed_route_rejects_ignored_scientific_or_resource_override(tmp_path,option,value,monkeypatch):
    from fudan_wgs_toolkit import phewas_run
    (tmp_path/'cache_dataset.json').write_text('{}')
    def forbidden(*args,**kwargs):raise AssertionError('job must not start for unsupported options')
    monkeypatch.setattr(phewas_run,'run_phewas',forbidden)
    with pytest.raises(ValueError):
        run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out',**{option:value})

@pytest.mark.parametrize('kwargs',[{'trait_batch_size':32},{'continuous_transform':'none'}])
def test_modern_route_rejects_ignored_mixed_workflow_options(tmp_path,kwargs):
    with pytest.raises(ValueError,match='population-cache'):
        run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out',**kwargs)

def test_cli_reports_successful_population_cache_run(tmp_path,monkeypatch,capsys):
    import json
    from fudan_wgs_toolkit import phewas_run,run
    (tmp_path/'cache_dataset.json').write_text('{}')
    monkeypatch.setattr(phewas_run,'run_phewas',lambda *a,**k:[dict(association_rows={
        'individual':2,'coding':3,'noncoding':4,'ncrna':1})])
    run.main(['p.csv','c.csv',str(tmp_path),'--output-directory',str(tmp_path/'out')])
    report=json.loads(capsys.readouterr().out)
    assert report['completed'] and report['eligible_association_tests']==10
