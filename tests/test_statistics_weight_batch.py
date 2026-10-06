"""Weight batching controls; synthetic fixtures here are not a benchmark."""
import math
import struct
import warnings
import pytest
import torch
import staar_phewas.statistics as statistics


@pytest.mark.parametrize('length',[3,33,131])
@pytest.mark.parametrize('group',['mixed','all_rare','all_common'])
@pytest.mark.parametrize('calibration',['chi2','gaussian_glm'])
def test_weight_batch_keeps_scalar_values_and_fields(length,group,calibration):
    generator=torch.Generator().manual_seed(442+length)
    x=torch.randn(length,4,generator=generator,dtype=torch.float64)
    covariance=x@x.T+torch.eye(length,dtype=torch.float64)
    score=torch.randn(length,generator=generator,dtype=torch.float64)*.1
    mac=torch.full((length,),2. if group=='all_rare' else 30.,dtype=torch.float64)
    if group=='mixed':mac[:length//2]=2.
    payload=dict(score=score,covariance=covariance,maf=torch.linspace(.001,.009,length,dtype=torch.float64),
                 mac=mac,annotations=torch.rand(length,3,generator=generator,dtype=torch.float64)*40,
                 acat_calibration=calibration,dof=100,tail_optimization=True)
    with warnings.catch_warnings(record=True) as original_warnings:
        warnings.simplefilter('always')
        original=statistics.staar_test(**payload)
    statistics.statistics_execution_metadata(reset=True)
    with warnings.catch_warnings(record=True) as candidate_warnings:
        warnings.simplefilter('always')
        candidate=statistics.staar_test(**payload,weight_batch_optimization=True)
    assert [(w.category,str(w.message)) for w in candidate_warnings] == [(w.category,str(w.message)) for w in original_warnings]
    assert list(candidate)==list(original)
    assert all(struct.pack('d',float(value))==struct.pack('d',float(original[field]))
               for field,value in candidate.items())
    assert statistics.statistics_execution_metadata()['weight_batch_optimization_calls']==1


def test_chi_square_tail_calls_are_batched(monkeypatch):
    original=statistics._chi1_from_score
    calls=[]
    def observe(score,variance):
        calls.append(tuple(score.shape))
        return original(score,variance)
    monkeypatch.setattr(statistics,'_chi1_from_score',observe)
    payload=dict(score=[.2,-.1,.5],covariance=[[2.,.1,.3],[.1,1.,.2],[.3,.2,3.]],
        maf=[.001,.002,.004],mac=[2.,4.,30.],annotations=[[10.,20.],[20.,40.],[30.,50.]],
        _skat_pvalues=[.1,.2,.3,.4,.5,.6],tail_optimization=True)
    statistics.staar_test(**payload)
    assert len(calls)==13
    calls.clear()
    statistics.staar_test(**payload,weight_batch_optimization=True)
    assert calls==[(1,),(6,),(6,)]
    calls.clear()
    payload['mac']=[2.,4.,6.]
    statistics.staar_test(**payload,weight_batch_optimization=True)
    assert calls==[(0,),(6,)]


@pytest.mark.parametrize('p',[[1e-200,.1,.7],[.5,.9,1.],[0.,.2,.3]])
def test_cached_cct_tangent_keeps_internal_branch(p):
    values=torch.tensor(p,dtype=torch.float64)
    transformed=torch.tan((.5-values)*math.pi)
    original=statistics._cct_tensor(values,[1.,2.,3.],internal=True,sync_light=True)
    candidate=statistics._cct_tensor(values,[1.,2.,3.],internal=True,sync_light=True,
                                      _normal_transform=transformed)
    assert torch.equal(original.view(torch.int64),candidate.view(torch.int64))
