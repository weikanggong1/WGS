"""Whole-chromosome configuration and execution public entry points."""
from staar_phewas.chromosome import (
    CODING_MASKS, NONCODING_MASKS, chromosome_configuration, run_chromosome, main,
)
__all__ = ["CODING_MASKS", "NONCODING_MASKS", "chromosome_configuration", "run_chromosome", "main"]
if __name__ == "__main__":
    main()
