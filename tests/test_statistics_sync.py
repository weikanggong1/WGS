"""CPU bitwise differential checks; GPU real-data checks run separately."""
import warnings
import struct
import pytest
import torch
from fudan_wgs_toolkit.statistics import _cct_tensor, _quadratic_form_sf_tensor, association_test
from fudan_wgs_toolkit._statistics_sync import bisection_root


def reference_root(scaled, q):
    lower = q.new_tensor(-.01) if bool(q > scaled.sum()) else -torch.full_like(q,scaled.numel())/(2*q)
    upper, root = q.new_tensor(.499995),q.new_tensor(0.)
    for _ in range(2048):
        if bool((upper-lower).abs() <= 1e-8):
            break
        root = (upper+lower)/2
        derivative = (scaled/(1-2*scaled*root)).sum()-q
        if bool(derivative == 0):
            break
        upper = torch.where(derivative>0,root,upper)
        lower = torch.where(derivative>0,lower,root)
    else:
        raise ArithmeticError
    return root


@pytest.mark.parametrize('length',[1,2,3,32,33,101])
def test_root_and_tail_bitwise_match_reference(length):
    generator = torch.Generator().manual_seed(152+length)
    raw = torch.rand(length, generator=generator,dtype=torch.float64)+.01
    scaled = raw/raw.max()
    for ratio in [1e-8,.01,.9,.99999,1.,1.00001,1.1,2.,10.,1e5]:
        q = scaled.sum()*ratio
        assert torch.equal(bisection_root(scaled,q,lambda value:value.sum()).view(torch.int64), reference_root(scaled,q).view(torch.int64))
        q_raw = q*raw.max()
        reference = _quadratic_form_sf_tensor(q_raw,raw)
        candidate = _quadratic_form_sf_tensor(q_raw,raw,sync_light=True)
        assert torch.equal(reference.view(torch.int64),candidate.view(torch.int64))


@pytest.mark.parametrize('p,weights,internal',[
    ([0.,.4],[-1.,0.],False),([1.,.4],None,False),
    ([0.,1.],[1.,0.],False),([.1,.2],[0.,0.],False),
    ([.1,.2],[-1.,2.],False),([.1,.2],[float('nan'),1.],False),
    ([float('nan'),.2],None,False),([-.1,.2],None,False),
    ([.3,.8],[1.,5.],True),([1e-200,1e-200],None,False),
    ([1e-17,.5,1.],[1.,2.,3.],True),([0.,.5],[1.,1.],True),
    ([0.,.5],[0.,1.],True),([.01,.1,.3],None,False),
    ([.9,.99,.7],None,False),([.3,.3],[1.,5.],False),
])
def test_cct_validation_boundary_and_bits(p,weights,internal):
    outcomes=[]
    for optimized in [False,True]:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                result=_cct_tensor(p,weights,internal=internal,sync_light=optimized)
                outcomes.append(('value',result.clone(),[str(w.message)for w in captured]))
            except Exception as error:
                outcomes.append(('error',type(error),str(error),[str(w.message)for w in captured]))
    if outcomes[0][0]=='value':
        assert outcomes[1][0]=='value'
        assert torch.equal(outcomes[0][1].view(torch.int64),outcomes[1][1].view(torch.int64))
        assert outcomes[0][2]==outcomes[1][2]
    else:
        assert outcomes[0]==outcomes[1]


def test_whole_statistics_cpu_exact_values():
    score=torch.tensor([.2,-.1,.5],dtype=torch.float64)
    covariance=torch.tensor([[2.,.1,.3],[.1,1.,.2],[.3,.2,3.]],dtype=torch.float64)
    payload=dict(score=score,covariance=covariance,maf=[.001,.002,.004],mac=[2.,4.,30.],annotations=[[10.,20.],[20.,40.],[30.,50.]])
    reference=association_test(**payload)
    actual=association_test(**payload,tail_optimization=True)
    assert list(actual)==list(reference)
    assert actual==reference
    assert all(struct.pack("d",float(actual[key]))==struct.pack("d",float(value)) for key,value in reference.items())


def test_exact_zero_derivative_freezes_first_midpoint():
    scaled=torch.tensor([1.],dtype=torch.float64)
    midpoint=torch.tensor((.499995-.01)/2,dtype=torch.float64)
    q=(scaled/(1-2*scaled*midpoint)).sum()
    assert torch.equal(reference_root(scaled,q),midpoint)
    assert torch.equal(bisection_root(scaled,q,lambda value:value.sum()),midpoint)


@pytest.mark.parametrize('q',[0.,1e-10,1.,10.,1000.])
def test_eigenvalue_cutoff_and_moment_boundary_preserved(q):
    raw=torch.tensor([-1e-9,0.,.999999999e-8,1e-8,1.000000001e-8,1.,2.],dtype=torch.float64)
    try:
        reference=_quadratic_form_sf_tensor(q,raw)
    except ArithmeticError as original_error:
        # At extremely small q, large negative brackets can stagnate above the
        # original absolute tolerance. Preserve the original failure boundary.
        with pytest.raises(ArithmeticError, match=str(original_error)):
            _quadratic_form_sf_tensor(q,raw,sync_light=True)
    else:
        candidate=_quadratic_form_sf_tensor(q,raw,sync_light=True)
        assert torch.equal(reference.view(torch.int64),candidate.view(torch.int64))


@pytest.mark.parametrize('q,raw',[
    (1e-300,[1e308,1e308]),  # finite raw inputs; scaled q underflows to zero
    (1e308,[1e-8,1e-8]),   # finite raw inputs; scaled q overflows
    (1e-300,[0.,1.,2.]),
    (1e308,[0.,1.,2.]),
])
def test_scaled_extremes_preserve_original_outcome(q,raw):
    raw=torch.tensor(raw,dtype=torch.float64)
    outcomes=[]
    for optimized in [False,True]:
        try:
            result=_quadratic_form_sf_tensor(q,raw,sync_light=optimized)
            outcomes.append(('result',result.view(torch.int64).item()))
        except Exception as error:
            outcomes.append(('error',type(error),str(error)))
    assert outcomes[0]==outcomes[1]


def test_batch_tail_groups_validation_and_branch_host_transfers(monkeypatch):
    import fudan_wgs_toolkit._statistics_sync as sync
    calls = []
    original = sync.host_flags
    def tracked(*flags):
        calls.append(len(flags))
        return original(*flags)
    monkeypatch.setattr(sync, 'host_flags', tracked)
    eigenvalues = torch.tensor([[.5, 1., 2.], [.5, 1., 2.], [.5, 1., 2.]], dtype=torch.float64)
    q = torch.tensor([0., 3.5, 7.], dtype=torch.float64)
    actual = sync.quadratic_form_sf_batch(q, eigenvalues)
    expected = torch.stack([_quadratic_form_sf_tensor(a, b) for a, b in zip(q, eigenvalues)])
    assert torch.equal(actual.view(torch.int64), expected.view(torch.int64))
    assert calls == [3, 2, 3]


@pytest.mark.parametrize('q,eigenvalues,error', [
    ([-1.], [[0.]], ValueError),
    ([float('nan')], [[0.]], ValueError),
    ([0.], [[float('inf')]], ValueError),
    ([0.], [[0.]], RuntimeError),
])
def test_batch_tail_validation_failure_priority(q, eigenvalues, error):
    from fudan_wgs_toolkit._statistics_sync import quadratic_form_sf_batch
    from fudan_wgs_toolkit.statistics import DegenerateTestError
    expected = DegenerateTestError if error is RuntimeError else error
    with pytest.raises(expected):
        quadratic_form_sf_batch(torch.tensor(q, dtype=torch.float64),
                                torch.tensor(eigenvalues, dtype=torch.float64))
