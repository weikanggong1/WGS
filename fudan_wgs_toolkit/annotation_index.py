"""Reusable variant-to-gene annotation indices, without genotype payloads.

SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CandidateAnnotationIndex:
    """Candidate identity, before model-specific MAF and missing imputation.

    Every array contains unique zero-based genotype rows in file order. Distal
    enhancer assignment uses the original annotation, not a gene span.
    """
    chromosome: str
    variant_type: str
    prepared_categories: set = field(default_factory=set)
    promoter_signature: tuple | None = None
    _groups: dict = field(default_factory=dict, repr=False)

    @property
    def genes(self):
        return tuple(sorted(self._groups))

    def categories_for(self, gene_name):
        return tuple(self._groups.get(gene_name, {}))

    def indices(self, gene_name, category):
        if category not in self.prepared_categories:
            raise ValueError(f"annotation category {category!r} has not been prepared")
        return self._groups.get(gene_name, {}).get(category, np.empty(0, dtype=np.int64))

    def manifest(self):
        """Return public gene/category candidate counts; counts are not RV counts."""
        return [{"gene_name": gene, "category": category,
                 "candidate_variants": len(indices)}
                for gene in self.genes for category, indices in self._groups[gene].items()]

    def add_assignments(self, chunks, category, indices, assignments):
        local = defaultdict(list)
        for index, genes in zip(indices, assignments):
            for gene in dict.fromkeys(genes):
                if gene:
                    local[gene].append(int(index))
        for gene, rows in local.items():
            chunks.setdefault((gene, category), []).append(np.asarray(rows, dtype=np.int64))

    def finish(self, chunks, categories):
        for (gene, category), parts in chunks.items():
            # Input chunks are ordered; unique also removes repeated gene tokens.
            rows = np.unique(np.concatenate(parts))
            rows.setflags(write=False)
            self._groups.setdefault(gene, {})[category] = rows
        self.prepared_categories.update(categories)
