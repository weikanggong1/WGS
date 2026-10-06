"""六状态 CSR 缓存：保留物理样本轴、原始 REF 与半缺失信息。"""
from .fast_container import Container
from .adapter_fast import CachedGDSAdapter
from .runtime import CacheSpec, IndexCacheSpec, make_reader_factory, run_cached_configuration

__all__ = ['Container', 'CachedGDSAdapter', 'CacheSpec', 'IndexCacheSpec',
           'make_reader_factory', 'run_cached_configuration']
