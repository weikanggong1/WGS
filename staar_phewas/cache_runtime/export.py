"""完整物理变异轴六状态 CSR 转存；仅显式 --execute 执行 CUDA。"""
import argparse
import json
from pathlib import Path
import time
from .binding import make_source_binding
from . import store

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('gds', 'packed-build', 'output'):
        result.add_argument('--'+name, type=Path, required=True)
    axis = result.add_mutually_exclusive_group(required=True)
    axis.add_argument('--samples-npy', type=Path, help='物理样本行号的一维 int64 .npy，顺序保留')
    axis.add_argument('--model', type=Path, help='由原 CLI sample-ID 规则绑定的既有 null model')
    result.add_argument('--sample-id-rule', default='auto')
    result.add_argument('--config', type=Path, help='可选分析配置，只加入输入 hash 绑定')
    result.add_argument('--source-manifest', type=Path, help='包相对源码路径到 SHA256 的可选 JSON')
    result.add_argument('--read-batch', type=int, default=4096)
    result.add_argument('--max-chunks', type=int, help='限制本会话帧数，保留可恢复未完成缓存')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--execute', action='store_true')
    return result

def convert(args):
    """输入 parser 参数，输出匿名汇总 dict；不运行关联分析或改写源 GDS。"""
    if args.read_batch not in (1024, 2048, 4096):
        raise ValueError('read_batch must be 1024, 2048 or 4096')
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError('max_chunks must be positive')
    plan = dict(execute=args.execute, format=store.FORMAT, physical_chunk=store.CHUNK,
                zstd_level=3, storage_fraction_cap=.10, association_computation=False)
    if not args.execute:return plan
    import numpy as np
    import torch
    from ..gds import SeqArrayGDS
    from ..io import load_null_model
    from ..cli import _bind_gds_samples
    from ..tf32 import configure_tf32
    from .state_reader import build_reader
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != 'cuda':raise ValueError('Conversion requires an explicit CUDA device')
    if device.index is None:device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        min(1., 20*2**30/torch.cuda.get_device_properties(device).total_memory), device)
    configure_tf32(memory_limit_gib=20, split_k=0)
    read_states = build_reader()
    inputs = {key: value for key, value in dict(samples=args.samples_npy, model=args.model,
                                               config=args.config).items() if value is not None}
    def proof():
        value = make_source_binding(args.gds, packed_directory=args.packed_build,
                                    input_files=inputs, source_manifest=args.source_manifest)
        value['decoder_ast_sha256'] = read_states.decoder_ast_sha256
        value['sample_id_rule'] = args.sample_id_rule
        return value
    before = proof()
    reader = SeqArrayGDS(args.gds, packed_reader_directory=args.packed_build)
    try:
        if args.samples_npy is not None:
            samples = np.load(args.samples_npy, allow_pickle=False)
            if samples.dtype != np.int64 or samples.ndim != 1:
                raise ValueError('Physical sample vector must be one-dimensional int64')
        else:
            model = load_null_model(args.model, device=device, matmul_mode='tf32')
            samples = _bind_gds_samples(reader, model, None, args.sample_id_rule)
            del model
        if (not len(samples) or len(samples)>=65536 or np.any(samples<0) or
                np.any(samples>=reader.n_samples) or len(np.unique(samples))!=len(samples)):
            raise ValueError('Sample axis must contain 1..65535 unique bounded physical rows')
        writer = store.Writer(args.output, before, samples, int(reader.n_variants),
                              source_bytes=args.gds.stat().st_size)
        reader._prepare_genotype_index()
        setup_seconds = time.perf_counter()-started
        torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        resumed = writer.next_start
        session_frames = 0
        while writer.next_start<reader.n_variants and (args.max_chunks is None or session_frames<args.max_chunks):
            first = writer.next_start
            batch = args.read_batch if args.max_chunks is None else min(
                args.read_batch, store.CHUNK*(args.max_chunks-session_frames))
            end = min(first+batch, reader.n_variants)
            result = read_states(reader, np.arange(first,end,dtype=np.int64), samples,
                                 device=device, memory_limit_gib=20)
            torch.cuda.synchronize(device)
            states = result['states'].cpu().numpy()
            half = np.count_nonzero(states>=4,axis=1).astype(np.int64)
            if int(half.sum())!=int(result['half_missing']):
                raise ValueError('Original half-missing total mismatch')
            counts = dict(reference_alleles=result['reference_ac'].astype(np.int64,copy=False),
                          called_alleles=result['called_alleles'].astype(np.int64,copy=False),
                          half_missing_samples=half)
            for offset in range(0,end-first,store.CHUNK):
                length = min(store.CHUNK,end-first-offset)
                writer.append(states[offset:offset+length],
                              {key:value[offset:offset+length] for key,value in counts.items()})
                session_frames += 1
            del result,states,counts
        if proof()!=before:raise RuntimeError('Source/input binding changed during conversion')
        if torch.cuda.max_memory_allocated(device)>20*2**30:
            raise MemoryError('Measured conversion GPU peak exceeds 20 GiB')
        complete = writer.next_start==reader.n_variants
        if complete:writer.finish()
        # No side report is added inside the storage-cap-controlled container.
        return dict(completed=complete,pilot_only=not complete,frames=len(writer.rows),
                    setup_seconds=setup_seconds,conversion_this_session_seconds=time.perf_counter()-began,
                    resumed_variants=resumed,storage=store.usage(args.output),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    source_prepost_unchanged=True,association_precision_acceptance=False,
                    full_association_benchmark_complete=False)
    finally:
        reader.close()

def main():
    print(json.dumps(convert(parser().parse_args()), sort_keys=True))

if __name__=='__main__':main()
