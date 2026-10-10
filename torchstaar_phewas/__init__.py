"""Independent phenotype association runs sharing verified genotype IO."""
from staar_phewas.phewas_runtime.runtime import run_configuration
from staar_phewas.cache_runtime import CacheSpec
from staar_phewas import __version__
__all__ = ['run_configuration', 'CacheSpec']
