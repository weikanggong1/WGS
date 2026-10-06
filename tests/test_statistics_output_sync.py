"""Probability-only synchronization changes; no synthetic performance claims."""
import math
import struct
import warnings
import pytest
import torch
from staar_phewas import statistics as stats


def outcome(function):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            result=function()
        except Exception as error:
            return ('error',type(error),str(error),[str(x.message) for x in captured])
        if isinstance(result,dict):
            bits=[(key,type(value),struct.pack('d',value)) for key,value in result.items()]
        else:
            bits=result.view(torch.int64).tolist()
        return ('value',bits,[str(x.message) for x in captured])


@pytest.mark.parametrize('p,w', [
    ([[.1,.3]],[[1.,2.]]),([[1e-250,1e-17,.8]],[[1.,2.,3.]]),
    ([[1e-16,math.nextafter(1e-16,0.),math.nextafter(1e-16,1.)]],[[1.,1.,1.]]),
    ([[0.,.3,1.]],[[1.,2.,3.]]),([[0.,.3,1.]],[[0.,2.,3.]]),
    ([[float('nan'),.2]],[[-1.,0.]]),([[-.1,.2]],[[0.,0.]]),
    ([[.1,.2]],[[-1.,1.]]),([[.1,.2]],[[float('nan'),0.]]),
    ([[.1,.2]],[[1e308,1e308]]),([[.1,.2]],[[0.,0.]]),
    ([[]],[[]]),([[.1,.2]],[[1.]]),
])
def test_cct_flags_preserve_bits_validation_priority_and_final_nan(p,w):
    p=torch.tensor(p,dtype=torch.float64);w=torch.tensor(w,dtype=torch.float64)
    reference=outcome(lambda:stats._cct_rows(p,w,batch_validation=False))
    candidate=outcome(lambda:stats._cct_rows(p,w,batch_validation=True))
    assert candidate==reference


def test_cct_validation_uses_one_six_flag_transfer(monkeypatch):
    import staar_phewas._statistics_sync as sync
    calls=[];original=sync.host_flags
    def tracked(*flags):calls.append(len(flags));return original(*flags)
    monkeypatch.setattr(sync,'host_flags',tracked)
    stats._cct_rows(torch.tensor([[.1,.2]],dtype=torch.float64),torch.ones((1,2)))
    assert calls==[6]


@pytest.mark.parametrize('tail',[False,True])
@pytest.mark.parametrize('special',[None,0.,1.,float('nan')])
def test_w26_all_output_keys_types_bits_and_warnings(tail,special):
    labels=[f'annotation_{i}' for i in range(12)]
    p=torch.linspace(.01,.99,78,dtype=torch.float64).reshape(3,26)
    if special is not None:p[0,0]=special
    reference=outcome(lambda:stats._staar_probability_fields(p,labels,tail_optimization=tail,batch_copy=False))
    candidate=outcome(lambda:stats._staar_probability_fields(p,labels,tail_optimization=tail,batch_copy=True))
    assert candidate==reference
    if special is None:
        fields=stats._staar_probability_fields(p,labels,tail_optimization=tail)
        expected=[]
        for method,combined in [('SKAT','STAAR-S'),('Burden','STAAR-B'),('ACAT-V','STAAR-A')]:
            for beta in ('1,25','1,1'):
                expected.extend([f'{method}({beta})']+[f'{method}({beta})-{label}' for label in labels]+[f'{combined}({beta})'])
        assert list(fields)==expected+['ACAT-O','STAAR-O']
        assert len(fields)==86 and all(type(value) is float for value in fields.values())
        assert fields['SKAT(1,25)']==float(p[0,0])
        assert fields['STAAR-S(1,25)']==stats.cct(p[0,:13],sync_light=tail)
        assert fields['STAAR-O']==stats.cct(p.reshape(-1),sync_light=tail)


@pytest.mark.parametrize('shape',[(3,0),(0,26),(3,25),(2,26)])
def test_empty_or_incomplete_probability_rows_fail(shape):
    for batch in (False,True):
        with pytest.raises(ValueError,match='complete FP64'):
            stats._staar_probability_fields(torch.zeros(shape,dtype=torch.float64),
                [f'a{i}' for i in range(12)],batch_copy=batch)


@pytest.mark.parametrize('output_batch',[False,True])
@pytest.mark.parametrize('cct_flags',[False,True])
@pytest.mark.parametrize('mac',[[2.,4.,6.],[20.,30.,40.],[2.,20.,30.]])
def test_native_w26_whole_statistics_switches_match(monkeypatch,output_batch,cct_flags,mac):
    from staar_phewas import _burden
    monkeypatch.setattr(_burden,'ieee_burden_product',lambda a,b,**kwargs:a@b)
    generator=torch.Generator().manual_seed(928)
    payload=dict(score=torch.tensor([.2,-.4,.7]),covariance=torch.eye(3),
        maf=[.001,.003,.006],mac=mac,
        annotations=10+30*torch.rand((3,12),generator=generator),
        names=[f'a{i}' for i in range(12)],matmul_mode='tf32',tail_optimization=True)
    original=stats.staar_test(**payload,output_batch_optimization=False,cct_validation_optimization=False)
    actual=stats.staar_test(**payload,output_batch_optimization=output_batch,cct_validation_optimization=cct_flags)
    assert list(actual)==list(original)
    assert len(actual)==88
    for key in actual:
        assert type(actual[key]) is type(original[key])
        assert struct.pack('d',actual[key])==struct.pack('d',original[key])
