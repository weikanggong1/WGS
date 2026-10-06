"""Private six-state cache reader; source28 SDK/layer decoder remains pinned."""
import ast
import hashlib
import importlib
import inspect
from pathlib import Path
import textwrap

SOURCE_GDS_CUDA_SHA='76f891b02032997ddc92e4a8eb214421e3a5ea0d9ac26edaf96e24bb6ca0f148'


def state_code(reference,called):
    """CPU scalar contract only. REF means allele code 0, not minor allele."""
    if type(reference) is not int or type(called) is not int or not 0<=reference<=called<=2:
        raise ValueError('invalid diploid integer REF/called counts')
    return 2-reference if called==2 else 3 if called==0 else 5-reference


GROUP_STATES='''
# Entire original group, no MAC eligibility or whole/half-missing masking.
with measure("gds_cache_state_encode", gpu=True):
    sample_reference = reference.sum(dim=2, dtype=torch.uint8)
    sample_called = (~missing).sum(dim=2, dtype=torch.uint8)
    dosage = torch.where(sample_called == 2, 2 - sample_reference,
                        torch.where(sample_called == 0, 3, 5 - sample_reference)).to(torch.uint8)
    if use_selected:
        if sample_order is None:
            sample_order = _device_sample_index(reader,samples,device,permutation=permutation)
        dosage = dosage.index_select(1, sample_order)
    for output, row in zip(outputs, dosage):
        dosages[int(output)] = row
    del sample_reference, sample_called, dosage
'''
TAIL='''
with measure("gds_cache_state_assemble", gpu=True):
    absent = torch.full((n,), 3, dtype=torch.uint8, device=device)
    states = (torch.stack([dosages.get(output, absent) for output in range(m)], dim=0)
              if m else torch.empty((0, n), dtype=torch.uint8, device=device))
return dict(states=states, reference_ac=reference_ac, called_alleles=called_alleles,
            half_missing=half_missing, variant_indices=variants.copy(), sample_indices=samples.copy(),
            allele_direction="original_REF", encoding="0/1/2 ALT count; 3 missing-both; 4 REF+missing; 5 nonREF+missing")
'''


def clone_decoder(original,namespace):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn=tree.body[0]
    eligibility=0
    for parent in ast.walk(fn):
        if not isinstance(parent,ast.For):continue
        for i,node in enumerate(parent.body):
            if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='eligible':
                if not isinstance(parent.body[i+1],ast.If):raise RuntimeError('eligibility seam changed')
                parent.body[i:i+2]=ast.parse(GROUP_STATES).body;eligibility+=1;break
    if eligibility!=1:raise RuntimeError('expected one original eligibility boundary')
    tail=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='summaries')
    fn.body[tail:]=ast.parse(TAIL).body
    ast.fix_missing_locations(tree)
    names={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
           for n in ast.walk(tree) if isinstance(n,ast.Call)}
    if names&{'DeviceMinorBlock','SparseMinorBlock','trait_dense','score_covariance','masked_fill_'}:
        raise RuntimeError('state reader retained dosage/minor/statistics boundary')
    scope=dict(namespace)
    exec(compile(tree,'<validated-six-state-reader>','exec'),scope)
    return scope[fn.name],hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def build_reader():
    """Explicit runtime entry. Default import/plan above is stdlib-only."""
    module=importlib.import_module('..gds_cuda', __package__)
    path=Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_GDS_CUDA_SHA:
        raise RuntimeError('Validated gds_cuda source binding mismatch')
    function,digest=clone_decoder(module.native_minor_block,vars(module))

    def read_states(reader,variants,samples,*,device,memory_limit_gib=20):
        import numpy as np
        import torch
        if reader.ploidy!=2:raise ValueError('six-state cache requires diploid original GDS')
        variants=np.asarray(variants);samples=np.asarray(samples)
        for values,size in ((variants,reader.n_variants),(samples,reader.n_samples)):
            if values.ndim!=1 or not np.issubdtype(values.dtype,np.integer):raise ValueError('require integer caller index vectors')
            if np.any(values<0) or np.any(values>=size) or len(np.unique(values))!=len(values):raise ValueError('indices must be unique and bounded')
        variants=variants.astype(np.int64,copy=False);samples=samples.astype(np.int64,copy=False)
        target=torch.device(device)
        if target.type!='cuda':raise ValueError('cache output requires CUDA ownership')
        if not 0<memory_limit_gib<=20:raise ValueError('cache process cap must be <=20GiB')
        n,m=len(samples),len(variants)
        # Conservative original decoder scratch plus per-group state+final stack.
        required=98*n*m+4*reader.genotype_raw_memory_bytes
        allocated=torch.cuda.memory_allocated(target)
        unused=max(0,torch.cuda.memory_reserved(target)-allocated)
        free,_=torch.cuda.mem_get_info(target)
        reserve=256*2**20
        if required>min(int(memory_limit_gib*2**30)-allocated,free+unused-reserve):
            raise MemoryError('six-state decoder/storage exceeds live/process budget')
        result=function(reader,variants,samples,device=target,minimum_mac=None,resident=True)
        states=result['states']
        if states.shape!=(m,n) or states.dtype!=torch.uint8 or not states.is_contiguous() or states.device!=target:
            # Resolve cuda without index against the actual allocated device.
            if not (target.index is None and states.device.type=='cuda' and states.shape==(m,n)
                    and states.dtype==torch.uint8 and states.is_contiguous()):
                raise RuntimeError('six-state output ownership/layout contract failed')
        if bool(torch.any(states > 5)):
            raise ValueError('invalid state codes 6/7 in generated cache block')
        result.update(source_gds_cuda_sha256=SOURCE_GDS_CUDA_SHA,decoder_ast_sha256=digest,
                      minimum_mac_filter_applied=False,minor_orientation_applied=False)
        return result
    read_states.decoder_ast_sha256=digest
    return read_states


def plan():
    return dict(source_gds_cuda_sha256=SOURCE_GDS_CUDA_SHA,output='CUDA uint8 [variant,sample], original caller order',
                states={'0':'two REF','1':'one REF/one nonREF','2':'two nonREF','3':'both missing',
                        '4':'one REF/one missing','5':'one nonREF/one missing'},
                minimum_mac_filter=False,minor_flip=False,GPU_execution=False)

if __name__=='__main__':
    import json
    print(json.dumps(plan()))
