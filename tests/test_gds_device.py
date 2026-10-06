"""Synthetic semantics units only, never a real-data benchmark."""
import numpy as np
import pytest
import torch
from staar_phewas.gds_cuda import native_minor_block
from tests.test_gds_flat_decode import make_reader


@pytest.mark.parametrize("multiplier,samples", [(1, [3, 0, 2, 1]), (4, [3, 0, 2, 1])])
@pytest.mark.parametrize("minimum_mac", [None, 1, 5])
def test_resident_decoder_and_trait_semantics(multiplier, samples, minimum_mac):
    reader, _, _, _ = make_reader(multiplier)
    reader.genotype_raw_memory_bytes = 4096
    variants = np.array([7, 2, 0, 3, 5, 1, 4, 6])
    samples = np.asarray(samples)
    legacy = native_minor_block(reader, variants, samples, device="cpu", minimum_mac=minimum_mac)
    resident = native_minor_block(reader, variants, samples, device="cpu", minimum_mac=minimum_mac, resident=True)
    np.testing.assert_array_equal(resident.variant_indices, legacy.variant_indices)
    np.testing.assert_array_equal(resident.sample_indices, legacy.sample_indices)
    np.testing.assert_array_equal(resident.initial_mac(), legacy.initial_mac())
    np.testing.assert_array_equal(resident.observed_mac([2, 0]), legacy.observed_mac([2, 0]))
    for rows in (np.array([3, 0, 2, 1]), np.array([2, 0]), np.array([], dtype=int)):
        for imputation in ("mean", "minor"):
            for frequency in ("count", "reference"):
                if frequency == "reference" and len(rows) != 4:
                    continue
                expected = legacy.trait_dense(rows, imputation, frequency_mode=frequency)
                actual = resident.trait_dense(rows, imputation, frequency_mode=frequency)
                np.testing.assert_array_equal(actual[0].numpy(), expected[0])
                for a, b in zip(actual[1:], expected[1:]):
                    np.testing.assert_array_equal(a, b)
                fp32 = resident.trait_dense(rows, imputation, frequency_mode=frequency, dtype=torch.float32)[0]
                np.testing.assert_array_equal(fp32.numpy(), expected[0].astype(np.float32))
    columns = np.arange(resident.shape[1])[::-1].copy()
    selected = resident.select_columns(columns)
    expected = legacy.select_columns(columns)
    np.testing.assert_array_equal(selected.trait_dense(np.arange(4))[0].numpy(), expected.trait_dense(np.arange(4))[0])
    with pytest.raises(ValueError):
        resident.trait_dense(np.arange(4), dtype=torch.float16)


def test_resident_requires_cuda_sdk_without_silent_fallback():
    reader, _, _, _ = make_reader()
    with pytest.raises(ValueError, match="requires CUDA"):
        reader.minor_block(np.array([0]), np.arange(4), device="cpu", resident=True)


def test_sample_mask_cache_owns_indices_and_evicts():
    reader, _, _, _ = make_reader(4)
    samples = np.array([3, 0, 2, 1])
    first = reader._sample_selection(samples)
    assert reader._sample_selection(samples)[0] is first[0]
    samples[0] = 5
    second = reader._sample_selection(samples)
    assert second[0] is not first[0]
    np.testing.assert_array_equal(np.flatnonzero(second[0])[::2] // 2, np.sort(samples))


def test_resident_af_tie_whole_missing_and_zero_layers():
    reader, _, _, _ = make_reader()
    reader.genotype_raw_memory_bytes = 4096
    variants, samples = np.array([0, 3, 1]), np.array([0])
    legacy = native_minor_block(reader, variants, samples, device="cpu")
    resident = native_minor_block(reader, variants, samples, device="cpu", resident=True)
    assert resident.union_ref_af[0] == 0.5
    for mode in ("count", "reference"):
        for imputation in ("mean", "minor"):
            got = resident.trait_dense(np.array([0]), imputation, frequency_mode=mode)
            wanted = legacy.trait_dense(np.array([0]), imputation, frequency_mode=mode)
            assert got[-1][0]  # ALT orientation at AF tie.
            for a, b in zip(got, wanted):
                np.testing.assert_array_equal(a.numpy() if torch.is_tensor(a) else a, b)


def small_device_block():
    from staar_phewas.gds_device import DeviceMinorBlock
    # Sentinel 3 is whole-genotype missingness. Reference allele summaries
    # deliberately differ from these minor-count sums; do not shortcut via AC.
    dosage=torch.tensor([[0,1,3,2],[2,3,3,0],[1,2,3,1]],dtype=torch.uint8)
    return DeviceMinorBlock(dosage,np.array([9,2,5]),np.array([8,3,1,7]),
        np.array([.5,.7,np.nan,.1]),np.array([3.,2.,np.nan,3.]),
        np.array([0.,1/6,1.,0.]),np.array([3.,4.,0.,1.]),np.array([6,5,0,6]))


def test_identity_selection_shares_lossless_input_but_dense_output_is_private():
    block=small_device_block();before=block.dosage.clone()
    assert block.select_columns(np.arange(4)) is block
    assert block._trait_counts(np.arange(3))[1] is block.dosage
    for dtype in (torch.float32,torch.float64):
        dense=block.trait_dense(np.arange(3),dtype=dtype)[0]
        assert dense.data_ptr()!=block.dosage.data_ptr()
        dense.zero_()
        assert torch.equal(block.dosage,before)


def test_host_count_cache_reuses_reductions_in_selected_children_and_invalidates_writes():
    from torch.utils._python_dispatch import TorchDispatchMode
    class SumCounter(TorchDispatchMode):
        def __init__(self):self.calls=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func)=='aten.sum.dim_IntList':self.calls+=1
            return func(*args,**(kwargs or {}))
    block=small_device_block();rows=np.array([2,0]);counter=SumCounter()
    with counter:
        expected=block.observed_mac(rows)
        assert counter.calls==2
        expected[0]=999  # Returned arrays never corrupt cached metadata.
        np.testing.assert_array_equal(block.observed_mac(rows),[1.,3.,0.,3.])
        block.trait_dense(rows,dtype=torch.float32)
        assert counter.calls==2
        selected=block.select_columns([3,0,1])
        np.testing.assert_array_equal(selected.observed_mac(rows),[3.,1.,3.])
        selected.trait_dense(rows,dtype=torch.float32)
        assert counter.calls==2
        block.dosage[0,0]=2
        np.testing.assert_array_equal(block.observed_mac(rows),[3.,3.,0.,3.])
        assert counter.calls==4
        block.observed_mac(rows[::-1].copy())
        assert counter.calls==6
    assert all(not torch.is_tensor(x) for x in block._counts_cache)


@pytest.mark.parametrize('rows',[np.arange(3),np.array([2,0]),np.array([],dtype=int)])
@pytest.mark.parametrize('frequency',['count','reference'])
@pytest.mark.parametrize('imputation',['mean','minor'])
def test_direct_fp32_imputation_matches_old_double_fill_cast_bits_and_all_metadata(rows,frequency,imputation):
    if frequency=='reference' and len(rows)!=3:return
    block=small_device_block();before=block.dosage.clone()
    double=block.trait_dense(rows,imputation,frequency_mode=frequency,dtype=torch.float64)
    single=block.trait_dense(rows,imputation,frequency_mode=frequency,dtype=torch.float32)
    assert torch.equal(single[0].contiguous().view(torch.int32),double[0].float().contiguous().view(torch.int32))
    for a,b in zip(single[1:],double[1:]):np.testing.assert_array_equal(a,b)
    assert torch.equal(block.dosage,before)


def test_inference_tensor_without_version_counter_bypasses_count_cache():
    block=small_device_block()
    with torch.inference_mode():block.dosage=block.dosage.clone()
    expected=block.observed_mac()
    assert block._counts_cache is None
    np.testing.assert_array_equal(block.observed_mac(),expected)


def test_empty_and_reordered_selection_preserve_sample_variant_order():
    block=small_device_block()
    columns=np.array([3,0,2]);rows=np.array([2,0])
    selected=block.select_columns(columns)
    np.testing.assert_array_equal(selected.sample_indices,block.sample_indices)
    np.testing.assert_array_equal(selected.variant_indices,block.variant_indices[columns])
    original=block.trait_dense(rows,dtype=torch.float32)
    actual=selected.trait_dense(rows,dtype=torch.float32)
    assert torch.equal(actual[0].view(torch.int32),original[0][:,columns].contiguous().view(torch.int32))
    assert block.select_columns(np.array([],dtype=int)).trait_dense(rows)[0].shape==(2,0)


@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('frequency', ['count', 'reference'])
def test_trait_summary_matches_dense_without_floating_slab(monkeypatch, imputation, frequency):
    reader, _, _, _ = make_reader()
    block = native_minor_block(reader, np.array([7,2,0,3,5,1,4,6]),
                               np.array([3,0,2,1]), device='cpu', resident=True)
    rows = np.array([2,0,3,1]) if frequency == 'reference' else np.array([2,0])
    expected = block.trait_dense(rows, imputation, frequency_mode=frequency)[1:]
    before = block.dosage.clone()
    monkeypatch.setattr(block, 'trait_dense', lambda *a, **k: pytest.fail('floating preparation before rare filter'))
    actual = block.trait_summary(rows, imputation, frequency_mode=frequency)
    for lhs, rhs in zip(actual, expected):np.testing.assert_array_equal(lhs, rhs)
    assert torch.equal(before, block.dosage)
    assert all(not isinstance(value, torch.Tensor) for value in block._counts_cache[1:])
