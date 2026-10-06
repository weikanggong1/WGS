"""Torchstaar public API backed by the validated scientific implementation."""
from staar_phewas import *
from staar_phewas import __all__ as _scientific_exports, __version__
from .chromosome import chromosome_configuration, run_chromosome
from .cli import run_configuration
from .prepare import prepare_input, read_relationship_matrix

__all__ = [*_scientific_exports, "chromosome_configuration", "run_chromosome",
           "run_configuration", "prepare_input", "read_relationship_matrix"]
