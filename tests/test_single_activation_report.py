"""Single execution metrics count computed work rather than configuration flags."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import torch
from fudan_wgs_toolkit.phewas_runtime.single import execution_metadata,process_single_batches
from test_phewas_runtime_contract import block

class Profiler:
    @contextmanager
    def measure(self,*args,**kwargs):yield

class Pipeline:
    def __init__(self):
        self.options=SimpleNamespace(wrapper_semantics='base')
        self.models=[SimpleNamespace(n=3,n_pheno=1,use_spa=False,family='gaussian',matmul_mode='tf32',
            x=torch.ones((3,1),dtype=torch.float32),device='cpu',individual_score_variance=lambda:None)]
        self.union_rows=np.arange(3);self.trait_rows=[np.arange(3)];self.profiler=Profiler()
    def _prepare_individual_block(self,block,ordinal,**kwargs):return block,ordinal+len(block.variant_indices)
    def _prepare_individual_trait(self,state,*args,**kwargs):
        return {'selected':state.variant_indices,'genotype':torch.ones((3,len(state.variant_indices)))}
    def _compute_individual_core(self,prepared):
        size=len(prepared['selected'])
        return {'score':torch.ones(size,dtype=torch.float32),'variance':torch.ones(size,dtype=torch.float32)}
    def _individual_records_from_values(self,prepared,values,**kwargs):
        return [{'Score':float(row[1])} for row in values]

def test_requested_but_unready_block_does_not_claim_computation():
    execution_metadata(reset=True)
    result,ordinals=process_single_batches([Pipeline()],[None],[0],'21')
    report=execution_metadata()
    assert result==[[]] and ordinals==[0]
    assert report['calls']==1 and report['active_trait_blocks']==report['computed_trait_blocks']==0
    assert report['pointwise_tail_batches']==report['result_transfer_batches']==0

def test_computed_blocks_and_shared_tails_are_reported_separately():
    execution_metadata(reset=True)
    pipelines=[Pipeline(),Pipeline()]
    result,ordinals=process_single_batches(pipelines,[block(np.arange(3),[1,2]),block(np.arange(3),[3])],[0,5],'21')
    report=execution_metadata()
    assert list(map(len,result))==[2,1] and ordinals==[2,6]
    assert report['computed_trait_blocks']==report['active_trait_blocks']==2
    assert report['pointwise_tail_batches']==report['result_transfer_batches']==1
    assert report['result_transfer_values']==12 and report['maximum_batch_variants']==3
    assert report['covariance_shared'] is False
