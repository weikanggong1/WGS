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
