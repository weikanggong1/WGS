"""CPU algebra/branch contracts; these are not accuracy or speed benchmarks."""
import math
import pytest
import torch
from staar_phewas import statistics as stats
from staar_phewas._statistics_sync import quadratic_form_sf_batch


def test_native_weight_families_and_small_phred():
    f=torch.tensor([.001,.004,.007],dtype=torch.float64)
    phred=torch.tensor([[1e-8,10.],[1e-8,20.],[1e-8,30.]],dtype=torch.float64)
    burden,skat,acat=stats.annotation_weights(f,phred,dtype=torch.float32)
    assert all(x.dtype==torch.float32 for x in (burden,skat,acat))
    rank=-torch.expm1(-phred.float()*math.log(10)/10)
    torch.testing.assert_close(burden[:,4],rank[:,0])
    torch.testing.assert_close(skat[:,4].square(),rank[:,0])
    assert bool((burden[:,1]>0).all())
    reference=stats.annotation_weights(f,phred)
    for x,y in zip((burden,skat,acat),reference):
        torch.testing.assert_close(x.double(),y,rtol=4e-6,atol=2e-7)


def test_exact_proportional_reuse_and_complete_fp32_eigh(monkeypatch):
    v=torch.tensor([[2.,.25],[.25,3.]],dtype=torch.float32)
    w=torch.tensor([[1.,2.,1.,3.],[2.,4.,3.,6.]],dtype=torch.float32)
    calls=[];eigh=torch.linalg.eigvalsh
    def checked(matrix,**kwargs):
        calls.append((matrix.dtype,matrix.shape))
        assert matrix.dtype==torch.float32
        return eigh(matrix,**kwargs)
    monkeypatch.setattr(torch.linalg,'eigvalsh',checked)
    result=stats._native_weighted_spectra(v,w)
    assert sum(shape[0] for _,shape in calls)==2
    reference=torch.stack([eigh(v*row[:,None]*row[None,:]) for row in w.T]).double()
    torch.testing.assert_close(result,reference,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(result[1],result[0]*4,rtol=0,atol=0)
    torch.testing.assert_close(result[3],result[0]*9,rtol=0,atol=0)


@pytest.mark.parametrize('factor',[0.,.3,1.,2.,50.])
def test_batch_original_saddle_and_moment_branches(factor):
    spectra=torch.tensor([[.5,1.,2.],[1.,2.,4.],[1e-9,.4,3.]],dtype=torch.float64)
    q=spectra.sum(dim=1)*factor
    actual=quadratic_form_sf_batch(q,spectra)
    expected=torch.stack([stats._quadratic_form_sf_tensor(x,e,sync_light=True) for x,e in zip(q,spectra)])
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-14)


def test_batch_cct_small_normal_zero_branches():
    p=torch.tensor([[1e-20,.02,.7],[.1,.2,.5],[0.,.2,1.]],dtype=torch.float64)
    w=torch.tensor([[1.,2.,3.]]*3,dtype=torch.float32)
    actual=stats._cct_rows(p,w)
    expected=torch.stack([stats._cct_tensor(x,y,internal=True) for x,y in zip(p,w)])
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-15)
    with pytest.raises(stats.DegenerateTestError):stats._cct_rows(p[2:],torch.tensor([[0.,1.,1.]]))


@pytest.mark.parametrize('mac',[ [4.,6.,12.], [2.,4.,5.], [12.,15.,20.] ])
def test_native_staar_core_and_output_storage(monkeypatch,mac):
    products=[]
    def cpu_algebra(a,b,**kwargs):
        assert a.dtype==b.dtype==torch.float32
        products.append((a.shape,b.shape))
        return a@b
    from staar_phewas import _burden
    monkeypatch.setattr(_burden,'ieee_burden_product',cpu_algebra)
    monkeypatch.setattr(stats,'_ordered_sum',lambda _:pytest.fail('native called ordered FP64 reduction'))
    result=stats.staar_test(torch.tensor([.5,1.,.3]),torch.eye(3),[.001,.002,.004],mac,
        annotations=[[10.],[20.],[30.]],names=['a'],matmul_mode='tf32',tail_optimization=True)
    assert products and result['num_variant']==3 and result['cMAC']==sum(mac)
    probabilities=[v for k,v in result.items() if k not in ('num_variant','cMAC')]
    assert probabilities and all(isinstance(x,float) and 0<x<=1 for x in probabilities)


def test_probability_storage_avoids_fp32_underflow():
    p=stats._chi1_from_score(torch.tensor([15.],dtype=torch.float32),torch.ones(1,dtype=torch.float32))
    assert p.dtype==torch.float64 and 0<float(p[0])<1e-45


def test_near_proportional_weights_are_not_reused(monkeypatch):
    perturbed=torch.nextafter(torch.tensor(4.,dtype=torch.float32),torch.tensor(float('inf')))
    weights=torch.tensor([[1.,2.],[2.,float(perturbed)]],dtype=torch.float32)
    calls=[];eigh=torch.linalg.eigvalsh
    def checked(matrix,**kwargs):
        calls.append(len(matrix));return eigh(matrix,**kwargs)
    monkeypatch.setattr(torch.linalg,'eigvalsh',checked)
    stats._native_weighted_spectra(torch.eye(2,dtype=torch.float32),weights)
    assert sum(calls)==2


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_weight_relation_chunks_keep_exact_identity(chunk):
    perturbed = torch.nextafter(torch.tensor(4., dtype=torch.float32), torch.tensor(float('inf')))
    rows = torch.tensor([[0., 1., 2., 0.], [0., 2., 4., 0.],
                         [0., 2., float(perturbed), 0.], [1., 0., 0., 2.]], dtype=torch.float32)
    bytes_per_row = 17 * 4 * 4 + 9 * 4
    exact, pivots, related = stats._native_weight_relations(rows, scratch_limit=bytes_per_row * chunk)
    assert pivots.tolist() == [1, 1, 1, 0]
    assert related == [[True, True, False, False], [True, True, False, False],
                       [False, False, True, False], [False, False, False, True]]
    assert exact.dtype == torch.float64
    assert torch.equal(exact, rows.double())


def test_weight_relation_extreme_values_and_noncontiguous_rows():
    tiny = torch.nextafter(torch.tensor(0., dtype=torch.float32), torch.tensor(1.))
    for values in (torch.tensor([[tiny, 0., tiny], [tiny * 2, 0., tiny * 2]]),
                   torch.tensor([[3e38, 1e38], [1.5e38, 5e37]], dtype=torch.float32)):
        _, _, related = stats._native_weight_relations(values)
        assert related == [[True, True], [True, True]]
    values = torch.tensor([[1., 9., 2., 8.], [2., 7., 4., 6.]], dtype=torch.float32)[:, ::2]
    assert not values.is_contiguous()
    assert stats._native_weight_relations(values)[2] == [[True, True], [True, True]]


def test_weight_relation_rejects_degenerate_or_unbounded_work():
    with pytest.raises(stats.DegenerateTestError):
        stats._native_weight_relations(torch.zeros((2, 3), dtype=torch.float32))
    with pytest.raises(MemoryError):
        stats._native_weight_relations(torch.ones((2, 3)), scratch_limit=1)
    with pytest.raises(ValueError, match="FP32"):
        stats._native_weight_relations(torch.ones((2, 3), dtype=torch.float64))


def test_batch_retains_absolute_eigen_cutoff_and_validation():
    eig=torch.tensor([[.999e-8,1e-8,2.],[0.,0.,3.]],dtype=torch.float64)
    q=torch.tensor([2.,0.],dtype=torch.float64)
    actual=quadratic_form_sf_batch(q,eig)
    expected=torch.stack([stats._quadratic_form_sf_tensor(x,e,sync_light=True) for x,e in zip(q,eig)])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)
    with pytest.raises(stats.DegenerateTestError):quadratic_form_sf_batch(torch.ones(1),torch.zeros((1,3)))
    with pytest.raises(ValueError):quadratic_form_sf_batch(torch.tensor([-1.]),torch.ones((1,3)))
    with pytest.raises(ValueError):quadratic_form_sf_batch(torch.ones(1),torch.tensor([[float('nan')]]))
