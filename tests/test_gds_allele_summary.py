"""Small synthetic I/O edge units; these are not performance benchmarks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staar_phewas.gds import SeqArrayGDS, _allele_code_dtype, _allele_frequency_summary


def fake_reader(codes, layers, n_samples=None):
    codes = np.asarray(codes, dtype=np.int64)
    layers = np.asarray(layers, dtype=np.int64)
    n_samples = codes.shape[1]
    raw = np.concatenate([np.asarray([(codes[i] >> (2 * layer)) & 3
        for layer in range(int(count))], dtype=np.uint8).reshape(int(count), n_samples, 2)
        for i, count in enumerate(layers)], axis=0)
    class SDK:
        def read_flat_path(self, fileid, path, offset, count, dtype):
            return raw.ravel()[offset:offset + count].copy()
        def read_selected_rows_path(self, fileid, path, offset, rows, width, mask, dtype):
            return raw.reshape(-1, width)[offset // width:offset // width + rows, mask].copy().ravel()
    reader = SeqArrayGDS.__new__(SeqArrayGDS)
    reader.n_samples, reader.n_variants, reader.ploidy = n_samples, len(layers), 2
    reader._file = SimpleNamespace(fileid=1)
    reader._flat_reader = SDK()
    reader._genotype_steps = layers
    reader._genotype_offsets = np.r_[0, layers.cumsum()]
    reader.genotype_raw_memory_bytes = 2**20
    reader.genotype_max_gap_layers = 8
    reader._prepare_genotype_index = lambda: None
    return reader


@pytest.mark.parametrize("layers,dtype", [(0,np.int16),(7,np.int16),(8,np.int32),(15,np.int32),(16,np.int64)])
def test_signed_dtype_retains_called_codes_and_missing(layers, dtype):
    assert _allele_code_dtype(layers) == dtype
    sentinel = (1 << (2 * layers)) - 1
    called = max(0, sentinel - 1)
    reader = fake_reader([[[called, sentinel],[0,0]]], [layers])
    result = reader.read_genotype(np.array([0]), np.array([1,0]))
    assert result.dtype == dtype
    if layers:
        assert result[0,1,0] == called and result[0,1,1] == -1
    else:
        assert np.all(result == -1)


def test_odd_called_alleles_preserve_original_rounding_and_all_missing_na():
    af, miss, mac, ac, called = _allele_frequency_summary([4,0], [7,0], 6)
    assert miss[0] == 5 / 12
    assert 6 * (1 - miss[0]) == 3.4999999999999996
    assert af[0] == 4 / 7 and mac[0] == 2
    assert np.isnan(af[1]) and np.isnan(ac[1]) and np.isnan(mac[1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partial_calls_tie_and_minor_fill_follow_original_allele_summaries(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for integer GPU decoding")
    codes = np.array([[[0,3],[3,1],[3,3],[0,0],[0,1],[1,1]],
                      [[0,3],[0,1],[3,1],[0,0],[3,1],[3,3]]])
    reader = fake_reader(codes, [1,1])
    block = reader.minor_block(np.array([0,1]), np.arange(6), device=device)
    assert block.initial_mac().tolist() == [4.,2.]
    assert block.union_ref_af.tolist() == [0.5,4/7]
    assert block.allele_missing_rate().tolist() == [1/3,5/12]
    assert block.observed_mac().tolist() == [3.,1.]
    # Source tie is ALT, with no per-trait second flip; whole partial calls NA.
    expected = np.array([[np.nan,np.nan],[np.nan,1],[np.nan,np.nan],[0,0],[1,np.nan],[2,np.nan]])
    mean, maf, _, missing, alt = block.trait_dense(np.arange(6))
    expected[np.isnan(expected[:,0]),0] = 1
    expected[np.isnan(expected[:,1]),1] = .5
    np.testing.assert_array_equal(mean, expected)
    assert alt.all() and missing.tolist() == [3,4]
    base_minor = block.trait_dense(np.arange(6), "minor", frequency_mode="reference")
    assert base_minor[1].tolist() == [4/12,3/12]
    # Initial MAC and observed MAC are distinct: no rounding from MAF*2N.
    filtered = reader.minor_block(np.array([0,1]), np.arange(6), device=device, minimum_mac=3)
    np.testing.assert_array_equal(filtered.variant_indices, [0])
    assert filtered.initial_mac()[0] == 4 and filtered.observed_mac()[0] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sample_indices", [np.array([4,1]),np.array([4,3,2,1,0])])
def test_gpu_matches_cpu_for_zero_and_sixteen_layers_and_arbitrary_order(sample_indices):
    high = 2**32-2
    codes = np.array([[[0,3]]*5, [[0,3],[1,1],[0,0],[3,1],[1,0]],
                      [[high,2**32-1],[0,0],[3,0],[1,0],[0,0]],
                      [[3,3],[0,1],[1,1],[0,0],[3,1]]])
    reader = fake_reader(codes, [0,1,16,1])
    variants = np.array([3,0,2,1])
    cpu = reader.minor_block(variants, sample_indices, device="cpu")
    gpu = reader.minor_block(variants, sample_indices, device="cuda")
    for key in ("row","col","value","union_ref_af","union_initial_mac","union_missing_rate",
                "union_ref_ac","union_called_alleles","variant_indices","sample_indices"):
        np.testing.assert_array_equal(getattr(cpu,key), getattr(gpu,key))
    assert reader._minor_decode_counts == {"cpu":1,"cuda":1}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_individual_coverage_counts_only_explicit_initial_mac_calls(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    reader = fake_reader([[[0,3],[3,1],[3,3],[0,0],[0,1],[1,1]]], [1])
    reader.minor_block(np.array([0]),np.arange(6),device=device)
    assert not hasattr(reader,"_minor_decode_coverage")
    reader.minor_block(np.array([0]),np.arange(6),device=device,minimum_mac=4)
    assert reader._minor_decode_coverage == {"decoded_variants":1,"mac_eligible_variants":1,
        "half_missing_genotypes":2,"ref_af_tie_variants":1,"all_missing_variants":0,"max_bit2_layers":1}


@pytest.mark.parametrize("core_dtype", [torch.float32, torch.float64])
def test_individual_nonpositive_variance_preserves_original_ieee_values(core_dtype):
    from staar_phewas.pipeline import AnalysisOptions, PheWASPipeline
    reader = fake_reader([[[0,0],[0,1],[1,1]]]*6,[1]*6)
    block = reader.minor_block(np.arange(6),np.arange(3))
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    variance = torch.tensor([-3.,-3e-12,0.,float("nan"),float("inf"),2.],dtype=core_dtype)
    score = torch.full((6,),6.,dtype=core_dtype)
    model=SimpleNamespace(n=3,n_pheno=1,use_spa=False,device="cpu",
                         individual_score_variance=lambda g:(score,variance))
    pipeline.models=[model]
    pipeline.trait_rows=[np.arange(3)]
    pipeline.union_rows=np.arange(3)
    pipeline.position=np.arange(6)
    pipeline.options=AnalysisOptions(wrapper_semantics="base")
    pipeline._base_mask=lambda *args:np.ones(6,dtype=bool)
    pipeline.gds=SimpleNamespace(n_variants=6,iter_minor_blocks=lambda *args,**kw:iter([block]),
        read_field=lambda *args:np.full(6,"1"),read_ref_alt=lambda *args:(np.full(6,"A"),np.full(6,"G")))
    rows=list(pipeline.iter_individual_records("1",mac_cutoff=1))[0][1]
    for j in (0,1):
        assert rows[j]["Score"]==6 and rows[j]["pvalue_log10"]==0
        assert np.isnan(rows[j]["Score_se"]) and np.isnan(rows[j]["Est_se"])
        assert rows[j]["Est"]==6/float(variance[j])
    assert rows[2]["Score_se"]==0 and rows[2]["Est"]==0 and rows[2]["Est_se"]==0
    assert np.isnan(rows[3]["pvalue_log10"]) and np.isnan(rows[3]["Est"])
    assert np.isinf(rows[4]["Score_se"]) and rows[4]["Est"]==0 and rows[4]["Est_se"]==0


@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('frequency_mode', ['count', 'reference'])
@pytest.mark.parametrize('filtered', [False, True])
def test_host_fp32_dense_matches_existing_double_route(imputation, frequency_mode, filtered):
    # Read actual original decoder output: includes half-missing, AF tie,
    # all-missing and fractional mean fills, with nonphysical variant order.
    codes = np.array([[[0,3],[3,1],[3,3],[0,0],[0,1],[1,1],[0,1]],
                      [[0,3],[0,1],[3,1],[0,0],[3,1],[3,3],[0,0]],
                      [[3,3]]*7, [[0,0],[0,1],[1,1],[0,0],[1,1],[0,1],[0,0]]])
    reader = fake_reader(codes, [1]*4)
    samples = np.array([5,2,0,6,1,4,3])
    block = reader.minor_block(np.array([3,1,2,0]), samples,
                               minimum_mac=2 if filtered else None)
    orders = [np.arange(6,-1,-1)]
    if frequency_mode == 'count':
        orders += [np.array([5,0,3]), np.array([],dtype=int)]
    for rows in orders:
        old = block.trait_dense(rows, imputation, frequency_mode=frequency_mode)
        direct = block.trait_dense(rows, imputation, frequency_mode=frequency_mode, dtype=np.float32)
        assert direct[0].dtype == np.float32
        np.testing.assert_array_equal(direct[0].view(np.uint32), old[0].astype(np.float32).view(np.uint32))
        for new_metadata, original_metadata in zip(direct[1:], old[1:]):
            np.testing.assert_array_equal(new_metadata, original_metadata)
    with pytest.raises(ValueError, match='dtype'):
        block.trait_dense(orders[0], dtype=np.float16)
