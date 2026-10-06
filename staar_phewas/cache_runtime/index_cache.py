"""Private CandidateAnnotationIndex persistence; no genotype or PHRED cache."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np

FORMAT='private-candidate-annotation-index-v1'
BINDING_KEYS={'gds_stat','n_variants','source_sha256','annotation_catalog','qc_path',
              'promoter_manifest','chromosome','variant_type','categories'}
ARRAY_KEYS={'binding_json','format_json','genes','pair_gene','pair_category','offsets',
            'variant_rows','prepared_categories','promoter_signature','promoter_present'}


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def make_binding(*,gds_stat,n_variants,source_sha256,annotation_catalog,qc_path,
                 promoter_manifest,chromosome,variant_type,categories):
    """Caller supplies normalized chromosome and original source/node bindings.

    promoter_manifest binds original file SHA plus exact normalized intervals;
    use None only for an index with no promoter categories/signature.
    source_sha256 must include index, pipeline, masks and GDS read definitions.
    No sample/model binding is needed: candidates precede MAF and imputation.
    """
    result=dict(gds_stat=gds_stat,n_variants=n_variants,source_sha256=source_sha256,
        annotation_catalog=annotation_catalog,qc_path=qc_path,promoter_manifest=promoter_manifest,
        chromosome=chromosome,variant_type=variant_type,categories=sorted(set(categories)))
    return validate_binding(json.loads(_json(result)))


def validate_binding(binding):
    if set(binding)!=BINDING_KEYS:raise ValueError('explicit cache binding fields required')
    if set(binding['gds_stat'])!={'device','inode','size','mtime_ns'}:raise ValueError('complete GDS stat required')
    if any(type(v) is not int or v<0 for v in binding['gds_stat'].values()):raise ValueError('invalid GDS stat')
    if type(binding['n_variants']) is not int or binding['n_variants']<0:raise ValueError('invalid physical row count')
    source=binding['source_sha256']
    required={'annotation_index.py','pipeline.py','masks.py','gds.py'}
    if not isinstance(source,dict) or not required<=set(source):raise ValueError('incomplete source semantics binding')
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in source.values()):raise ValueError('invalid source SHA')
    if not isinstance(binding['annotation_catalog'],dict) or not isinstance(binding['qc_path'],str) or not binding['qc_path']:raise ValueError('invalid annotation/QC binding')
    if not isinstance(binding['chromosome'],str) or not binding['chromosome']:raise ValueError('normalized chromosome required')
    if binding['variant_type'] not in ('SNV','Indel','variant'):raise ValueError('invalid variant type')
    cats=binding['categories']
    if not isinstance(cats,list) or cats!=sorted(set(cats)) or any(not isinstance(c,str) or not c for c in cats):raise ValueError('invalid category coverage')
    if any(c.startswith('promoter_') for c in cats) and binding['promoter_manifest'] is None:raise ValueError('promoter reference binding required')
    promoter=binding['promoter_manifest']
    if promoter is not None:
        if not isinstance(promoter,dict) or set(promoter)!={'file_sha256','normalized_signature'}:
            raise ValueError('exact promoter manifest fields required')
        digest=promoter['file_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('invalid promoter file SHA')
        signature=promoter['normalized_signature']
        if not isinstance(signature,list) or any(not isinstance(p,list) or len(p)!=2 or any(type(x) is not int for x in p) or p[0]>p[1] for p in signature) or signature!=sorted(signature):raise ValueError('invalid normalized promoter signature')
    _json(binding)
    return binding


def capture(index):
    genes=list(index._groups)
    pair_gene=[];pair_category=[];parts=[];offsets=[0]
    for gene,groups in index._groups.items():
        for category,rows in groups.items():
            a=np.asarray(rows)
            if a.dtype!=np.int64 or a.ndim!=1 or np.any(a<0) or np.any(a[1:]<=a[:-1]):raise ValueError('index rows must be strictly increasing int64')
            pair_gene.append(gene);pair_category.append(category);parts.append(a)
            offsets.append(offsets[-1]+len(a))
    present=index.promoter_signature is not None
    signature=np.asarray(index.promoter_signature if present else [],dtype=np.int64).reshape(-1,2)
    if np.any(signature[:,0]>signature[:,1]):raise ValueError('invalid promoter signature')
    return dict(genes=np.asarray(genes,dtype=np.str_),pair_gene=np.asarray(pair_gene,dtype=np.str_),
        pair_category=np.asarray(pair_category,dtype=np.str_),offsets=np.asarray(offsets,dtype=np.int64),
        variant_rows=np.concatenate(parts) if parts else np.empty(0,dtype=np.int64),
        prepared_categories=np.asarray(sorted(index.prepared_categories),dtype=np.str_),
        promoter_signature=signature,promoter_present=np.asarray(int(present),dtype=np.uint8))


def save(path,index,binding):
    binding=validate_binding(json.loads(_json(binding)))
    if (index.chromosome,index.variant_type)!=(binding['chromosome'],binding['variant_type']) or sorted(index.prepared_categories)!=binding['categories']:
        raise ValueError('index identity/category coverage differs from binding')
    data=capture(index)
    expected_sig=binding['promoter_manifest']
    if (index.promoter_signature is None)!=(expected_sig is None) or (expected_sig is not None and data['promoter_signature'].tolist()!=expected_sig['normalized_signature']):
        raise ValueError('index promoter signature differs from manifest')
    if np.any(data['variant_rows']>=binding['n_variants']):raise ValueError('physical row exceeds GDS dimensions')
    data.update(binding_json=np.frombuffer(_json(binding).encode(),dtype=np.uint8),
                format_json=np.frombuffer(_json(FORMAT).encode(),dtype=np.uint8))
    path=Path(path)
    if path.exists():raise FileExistsError('refuse to overwrite candidate cache')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.index-',suffix='.npz',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**data);stream.flush();os.fsync(stream.fileno())
        # link is atomic and cannot replace a concurrently created target.
        os.link(name,path)
    finally:
        Path(name).unlink(missing_ok=True)
    return dict(file_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                candidate_rows=len(data['variant_rows']),gene_category_pairs=len(data['pair_gene']))


def prepare(pipe,path,binding,*,promoter_intervals=None):
    """用原pipeline准备确切类别并保存，返回(index, anonymous_report)。

    binding必须由当前源stat、真实源码SHA与注释/promoter输入构造。
    promoter_intervals保留原pipeline输入形式，不从其他索引复制候选行。
    """
    expected=validate_binding(json.loads(_json(binding)))
    index=pipe.prepare_annotation_index(expected['chromosome'],
        categories=expected['categories'],include_ncrna=False,
        promoter_intervals=promoter_intervals)
    return index,save(path,index,expected)


def restore(pipe,path,expected_binding,*,max_uncompressed_bytes=512*2**20):
    expected=validate_binding(json.loads(_json(expected_binding)))
    path=Path(path)
    with zipfile.ZipFile(path) as z:
        if set(z.namelist())!={k+'.npy' for k in ARRAY_KEYS} or len(z.infolist())!=len(ARRAY_KEYS):raise ValueError('cache archive fields mismatch')
        if sum(i.file_size for i in z.infolist())>max_uncompressed_bytes:raise MemoryError('index cache decompression budget exceeded')
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in ARRAY_KEYS}
    for name in ('binding_json','format_json'):
        if data[name].dtype!=np.uint8 or data[name].ndim!=1:raise ValueError('invalid JSON byte array')
    actual=json.loads(data['binding_json'].tobytes())
    if actual!=expected or json.loads(data['format_json'].tobytes())!=FORMAT:raise ValueError('candidate cache source/input binding mismatch')
    for name in ('genes','pair_gene','pair_category','prepared_categories'):
        if data[name].ndim!=1 or data[name].dtype.kind!='U':raise ValueError('string directory dtype mismatch')
    genes=data['genes'].tolist();pg=data['pair_gene'].tolist();pc=data['pair_category'].tolist()
    if len(set(genes))!=len(genes) or any(not g for g in genes):raise ValueError('invalid gene directory')
    if len(pg)!=len(pc) or len(set(zip(pg,pc)))!=len(pg):raise ValueError('duplicate/category pair directory')
    categories=data['prepared_categories'].tolist()
    if categories!=expected['categories']:raise ValueError('prepared category coverage mismatch')
    if any(g not in genes or c not in categories for g,c in zip(pg,pc)):raise ValueError('pair outside directory coverage')
    offsets=data['offsets'];flat=data['variant_rows'];sig=data['promoter_signature'];present=data['promoter_present']
    if offsets.dtype!=np.int64 or flat.dtype!=np.int64 or offsets.shape!=(len(pg)+1,) or flat.ndim!=1:raise ValueError('invalid row/offset dtype/shape')
    if offsets[0]!=0 or offsets[-1]!=len(flat) or np.any(offsets[1:]<offsets[:-1]) or np.any(flat<0) or np.any(flat>=expected['n_variants']):raise ValueError('invalid physical row/offset bounds')
    if sig.dtype!=np.int64 or sig.ndim!=2 or sig.shape[1]!=2 or np.any(sig[:,0]>sig[:,1]) or present.dtype!=np.uint8 or present.shape!=() or int(present) not in (0,1):raise ValueError('invalid promoter signature')
    if not int(present) and len(sig):raise ValueError('absent promoter reference has intervals')
    promoter=expected['promoter_manifest']
    if bool(int(present))!=(promoter is not None) or (promoter is not None and sig.tolist()!=promoter['normalized_signature']):raise ValueError('cached promoter signature differs from manifest')
    from ..annotation_index import CandidateAnnotationIndex
    index=CandidateAnnotationIndex(expected['chromosome'],expected['variant_type'])
    index._groups={gene:{} for gene in genes}
    for i,(gene,category) in enumerate(zip(pg,pc)):
        rows=flat[offsets[i]:offsets[i+1]]
        if np.any(rows[1:]<=rows[:-1]):raise ValueError('cache rows must remain unique original order')
        rows.setflags(write=False);index._groups[gene][category]=rows
    index.prepared_categories=set(categories)
    index.promoter_signature=tuple(map(tuple,sig.tolist())) if int(present) else None
    key=(expected['chromosome'],expected['variant_type'])
    if key in pipe._annotation_indexes:raise ValueError('refuse to replace a live prepared index')
    pipe._annotation_indexes[key]=index
    return index
