"""CPU API/layout contracts only; actual native TF32 requires real CUDA audit."""
import pytest
import torch
from fudan_wgs_toolkit import tf32


def test_fp64_control_is_explicit_and_preserves_double_products():
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    b=a.T
    assert torch.equal(tf32.matmul(a,b),a@b)
    assert tf32.matmul(a,b,mode='fp64').dtype==torch.float64


@pytest.mark.parametrize('mode',['tf32x3','tf32_binned','allow_tf32','float16',None])
def test_obsolete_or_unrequested_precision_modes_rejected(mode):
    with pytest.raises(ValueError,match='obsolete'):tf32.validate_mode(mode)


def test_forced_native_mode_rejects_cpu_no_precision_fallback():
    a=torch.ones((2,2),dtype=torch.float64)
    with pytest.raises(ValueError,match='CUDA'):tf32.matmul(a,a,mode='tf32')
    assert tf32.execution_metadata()['fp64_gemm_fallback_count']==0


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.int64])
def test_native_rejects_other_storage_before_hardware(dtype):
    a=torch.ones((2,2),dtype=dtype)
    with pytest.raises(ValueError,match='float32/float64'):tf32.matmul(a,a,mode='tf32')


def test_sparse_operands_rejected():
    a=torch.ones((2,2)).to_sparse()
    with pytest.raises(ValueError,match='dense'):tf32.matmul(a,a,mode='tf32')


@pytest.mark.parametrize('left,right,route',[
 ((5,7),(7,3),'tf32_mma'),((1,7),(7,3),'gemv'),((5,7),(7,1),'gemv'),
 ((1,7),(7,1),'dot'),((5,1),(1,3),'outer'),((1,1),(1,3),'gemv'),((5,0),(0,3),'empty'),((0,7),(7,3),'empty')])
def test_native_matrix_vector_empty_dispatch(left,right,route):
    assert tf32._native_route(left,right)==route


@pytest.mark.parametrize('shapes',[((1,7),(7,1)),((4,7),(7,1)),((1,7),(7,4))])
def test_fp32_vector_helper_formula_shape_noncontiguous_and_no_input_mutation(shapes):
    left_shape,right_shape=shapes
    a=torch.arange(left_shape[0]*left_shape[1]*2,dtype=torch.float32).reshape(left_shape[0],-1)[:,::2]
    b=torch.arange(right_shape[0]*right_shape[1]*2,dtype=torch.float32).reshape(right_shape[0],-1)[:,::2]
    before_a=a.clone();before_b=b.clone()
    result=tf32._vector_product(a,b)
    expected=(a[:,:,None]*b[None,:,:]).sum(1)
    assert result.dtype==torch.float32 and result.shape==(left_shape[0],right_shape[1])
    torch.testing.assert_close(result,expected)
    assert torch.equal(a,before_a) and torch.equal(b,before_b)


_MMA='mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 %x;'
_RNA='cvt.rna.tf32.f32 %a, %b;'


def test_actual_ptx_validation_requires_mma_and_two_explicit_rna_conversions():
    tf32._verified_tf32_ptx(_MMA+'\n'+_RNA+'\n'+_RNA)
    with pytest.raises(RuntimeError,match='no TF32'):tf32._verified_tf32_ptx(_RNA+'\n'+_RNA)
    for missing in ('',_RNA,'// '+_RNA+'\n// '+_RNA,
                    'cvt.rn.tf32.f32 %a, %b;\ncvt.rn.tf32.f32 %c, %d;'):
        with pytest.raises(RuntimeError,match='RNA conversion'):
            tf32._verified_tf32_ptx(_MMA+'\n'+missing)
    with pytest.raises(RuntimeError,match='unsupported'):
        tf32._verified_tf32_ptx(_MMA+'\n'+_RNA+'\n'+_RNA+'\nmma.sync.aligned.f32.bf16.bf16.f32 %y;')


def test_code_verification_cache_retains_only_successful_compiled_objects():
    from types import SimpleNamespace
    tf32.execution_metadata(reset=True)
    valid=SimpleNamespace(asm={'ptx':_MMA+'\n'+_RNA+'\n'+_RNA})
    tf32._verify_compiled_tf32(valid);tf32._verify_compiled_tf32(valid)
    bad=SimpleNamespace(asm={'ptx':_MMA})
    with pytest.raises(RuntimeError):tf32._verify_compiled_tf32(bad)
    assert tf32._verified_compiled_cache[id(valid)] is valid
    assert id(bad) not in tf32._verified_compiled_cache
    metadata=tf32.execution_metadata(reset=True)
    assert metadata['ptx_specialization_inspection_count']==1
    assert metadata['ptx_verified_code_cache_hits']==1
    # CPU code inspection itself is not evidence of actual GPU execution.
    assert metadata['tf32_gemm_call_count']==metadata['ptx_verified_rna_gemm_count']==0


def test_capability_cache_uses_actual_device_index(monkeypatch):
    tf32._capability_cache.clear();queries=[]
    monkeypatch.setattr(torch.cuda,'current_device',lambda:3)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(queries.append(i) or (8,0)))
    assert tf32._device_capability(torch.device('cuda'))==(8,0)
    assert tf32._device_capability(torch.device('cuda:3'))==(8,0)
    assert tf32._device_capability(torch.device('cuda:2'))==(8,0)
    assert queries==[3,2]
    tf32._capability_cache.clear()


def test_outer_helper_preserves_fp32_formula_signedzero_strides_and_inputs():
    a=torch.tensor([0.,99.,-0.,99.,1.25,99.],dtype=torch.float32)[::2,None]
    b=torch.tensor([-2.,99.,0.,99.,3.5,99.],dtype=torch.float32)[None,::2]
    aa=a.clone();bb=b.clone()
    result=tf32._outer_product(a,b)
    expected=torch.stack([a[i,0]*b[0] for i in range(a.shape[0])])
    assert result.dtype==torch.float32 and result.shape==(3,3)
    assert torch.equal(result.view(torch.int32),expected.view(torch.int32))
    assert torch.equal(a.view(torch.int32),aa.view(torch.int32))
    assert torch.equal(b.view(torch.int32),bb.view(torch.int32))
    with pytest.raises(ValueError,match='float32'):tf32._outer_product(a.double(),b)
    with pytest.raises(ValueError,match='inner dimension'):tf32._outer_product(torch.ones(3,2),torch.ones(2,3))


@pytest.mark.parametrize('split_k',[1,32,64,1024,-1,True,0.,'0'])
def test_only_unsplit_native_product_is_supported(split_k):
    with pytest.raises(ValueError,match='split_k=0'):tf32.configure_tf32(split_k=split_k)


def test_removed_component_and_tile_controls_fail_directly():
    with pytest.raises(TypeError,match='Obsolete'):tf32.configure_tf32(binned_split_k=1024)
    with pytest.raises(TypeError):tf32.matmul(torch.ones(2,2),torch.ones(2,2),split_k=32)
    assert not hasattr(tf32,'_split_binned') and not hasattr(tf32,'_split_three')


def test_configuration_metadata_does_not_claim_actual_mma():
    try:
        tf32.configure_tf32(memory_limit_gib=12,split_k=0)
        m=tf32.execution_metadata(reset=True)
        assert m['fp32_outer_product_count']==0
        assert 'ties away from zero' in m['tf32_input_rounding']
        assert m['backend']=='not_used' and m['tf32_gemm_call_count']==m['ptx_verified_tf32_gemm_count']==0
        assert m['component_reconstruction_product_count']==m['tf32_total_logical_component_product_count']==0
        assert m['kernel_accumulator_dtype']==m['kernel_output_dtype']==m['result_dtype']=='float32'
        assert m['tf32_memory_guard']['process_allocated_limit_bytes']==12*2**30
        assert tf32.execution_metadata()['raw_tf32_default_split_k']==0
    finally:tf32.configure_tf32()


def test_outer_dispatch_counts_no_mma_and_returns_fp32_with_cpu_device_stub(monkeypatch):
    # Dispatch contract only: replacing the CUDA-device predicate makes no
    # hardware claim; the real elementwise helper still computes the product.
    monkeypatch.setattr(torch.Tensor,'is_cuda',property(lambda self:True))
    monkeypatch.setattr(tf32,'_guard_product_workspace',lambda *args,**kwargs:None)
    def forbidden(*args):raise AssertionError('K=1 outer must not use GEMM/GEMV')
    monkeypatch.setattr(tf32,'_matrix_product',forbidden)
    monkeypatch.setattr(tf32,'_vector_product',forbidden)
    tf32.execution_metadata(reset=True)
    a=torch.tensor([[1.25],[-2.]],dtype=torch.float64)
    b=torch.tensor([[3.,4.,5.]],dtype=torch.float64)
    result=tf32.matmul(a,b,mode='tf32')
    assert result.dtype==torch.float32
    assert torch.equal(result,a.float()*b.float())
    report=tf32.execution_metadata(reset=True)
    assert report['logical_product_count']==report['fp32_outer_product_count']==1
    assert report['tf32_gemm_call_count']==report['ptx_verified_tf32_gemm_count']==report['fp32_vector_product_count']==0


@pytest.mark.parametrize('m,n,k,expected_grid',[
    (256,256,1024,(4,2)),(257,257,1025,(5,3)),
    (255,256,1024,(8,4)),(256,255,1024,(8,4)),(256,256,1023,(8,4)),
])
def test_matrix_launch_geometry_uses_large_policy_only_when_all_bounds_hold(monkeypatch,m,n,k,expected_grid):
    from types import SimpleNamespace
    from contextlib import nullcontext
    launches=[]
    class Kernel:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                launches.append((grid,kwargs,args[3:6]))
                return SimpleNamespace(asm={'ptx':_MMA+'\n'+_RNA+'\n'+_RNA})
            return launch
    monkeypatch.setattr(tf32,'triton',SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b))
    monkeypatch.setattr(tf32,'_gemm',Kernel(),raising=False)
    monkeypatch.setattr(tf32,'_device_capability',lambda device:(8,0))
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    tf32.execution_metadata(reset=True)
    # CPU operands test host dispatch/launch arguments; the kernel is a stub.
    output=tf32._matrix_product(torch.empty(m,k),torch.empty(k,n))
    grid,config,dimensions=launches[0]
    assert grid==expected_grid and dimensions==(m,n,k)
    assert config['BK']==32 and config['num_warps']==4 and config['num_stages']==3
    assert grid[0]*config['BM']>=m and grid[1]*config['BN']>=n
    assert (grid[0]-1)*config['BM']<m and (grid[1]-1)*config['BN']<n
    assert output.shape==(m,n) and output.dtype==torch.float32
    report=tf32.execution_metadata(reset=True)
    assert report['output_tile_padding_elements']==grid[0]*config['BM']*grid[1]*config['BN']-m*n
    assert report['tf32_gemm_call_count']==report['ptx_verified_rna_gemm_count']==1
