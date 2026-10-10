"""Verified six-state chromosome caches."""
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np
from .fast_container import Container
from .adapter_fast import CachedGenotypeAdapter


@dataclass(frozen=True)
class CacheSpec:
    """Bind one immutable prepared chromosome to its live source proof.

    ``directory`` is the compressed genotype container. ``expected_samples``
    contains physical source row numbers, separately from FID/IID identities.
    ``source_proof`` returns the current binding at both runtime boundaries.
    """
    directory: str
    expected_binding: Mapping
    source_proof: Callable[[], Mapping]
    expected_samples: np.ndarray | None = None


__all__ = ["Container", "CachedGenotypeAdapter", "CacheSpec"]
