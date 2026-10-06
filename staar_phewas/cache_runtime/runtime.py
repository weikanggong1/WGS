"""显式缓存 reader factory 与串行 CLI 包装，不改变统计计算。"""
from dataclasses import dataclass
from pathlib import Path
import json
import threading
from typing import Callable, Mapping
from .adapter_fast import CachedGDSAdapter
from .fast_container import Container
from . import index_cache

@dataclass(frozen=True)
class CacheSpec:
    directory: Path
    expected_binding: dict
    source_proof: Callable[[], dict]
    expected_samples: object = None
    compact_cache_bytes: int = 64 * 2**20

@dataclass(frozen=True)
class IndexCacheSpec:
    path: Path
    expected_binding: dict
    max_uncompressed_bytes: int = 512 * 2**20

def make_reader_factory(original_factory, cache_specs: Mapping, *, device='cuda:0'):
    """返回与 SeqArrayGDS(path, **options) 相同调用形式的独立 factory。

    cache_specs 的键为源 GDS 路径；未配置的路径拒绝，不静默回退读取
    genotype。source_proof 必须重新检查全部当前输入/source/stat，返回与
    expected_binding 一致的 dict；每次 open/close 均执行。
    """
    specs = {str(Path(path).resolve()): spec for path, spec in cache_specs.items()}
    def factory(path, **options):
        key = str(Path(path).resolve())
        if key not in specs:
            raise ValueError('No explicit cache binding for requested GDS')
        spec = specs[key]
        expected = json.loads(json.dumps(spec.expected_binding, sort_keys=True, allow_nan=False))
        if not callable(spec.source_proof) or spec.source_proof() != expected:
            raise ValueError('Current source/input proof differs from cache binding')
        reader = original_factory(path, **options)
        container = None
        try:
            container = Container(spec.directory, expected, spec.expected_samples)
            class BoundAdapter(CachedGDSAdapter):
                def close(self):
                    if self._closed:return
                    try:
                        if spec.source_proof() != expected:
                            raise ValueError('Source/input binding changed during analysis')
                    finally:
                        super().close()
            adapter = BoundAdapter(reader, container, device=device,
                                   compact_cache_bytes=spec.compact_cache_bytes)
            adapter._cache_source_path = key
            return adapter
        except BaseException:
            try:
                if container is not None:container.close()
            finally:
                reader.close()
            raise
    return factory

def restore_indexes(pipeline, specs):
    """加载每个显式 IndexCacheSpec；拒绝 live index、绑定差异或超预算。"""
    return [index_cache.restore(pipeline, item.path, item.expected_binding,
                               max_uncompressed_bytes=item.max_uncompressed_bytes)
            for item in specs]

_CLI_LOCK = threading.Lock()

def run_cached_configuration(configuration, *, cache_specs, device='cuda:0',
                             index_caches=None, **run_options):
    """串行包装原 cli.run_configuration；统计、模型与输出仍由原 CLI 完成。

    index_caches 键为源 GDS 路径，值为 IndexCacheSpec 序列。
    由于原 CLI 不接收 factory，本入口仅在调用期间绑定该模块的 reader
    和 pipeline 构造器，finally 恢复。不可与普通 CLI 或另一个 CLI 在
    同一进程并行执行；并行使用者应使用独立 factory/进程。
    """
    from .. import cli
    if not _CLI_LOCK.acquire(blocking=False):
        raise RuntimeError('Cached CLI context is already running')
    original_reader = cli.SeqArrayGDS
    original_pipeline = cli.PheWASPipeline
    indexes = {str(Path(path).resolve()): value for path, value in (index_caches or {}).items()}
    try:
        factory = make_reader_factory(original_reader, cache_specs, device=device)
        class CachedPipeline(original_pipeline):
            def __init__(self, gds, *args, **kwargs):
                super().__init__(gds, *args, **kwargs)
                source = Path(gds._cache_source_path).resolve()
                restore_indexes(self, indexes.get(str(source), ()))
        cli.SeqArrayGDS = factory
        cli.PheWASPipeline = CachedPipeline
        return cli.run_configuration(configuration, device=device, **run_options)
    finally:
        cli.SeqArrayGDS = original_reader
        cli.PheWASPipeline = original_pipeline
        _CLI_LOCK.release()
