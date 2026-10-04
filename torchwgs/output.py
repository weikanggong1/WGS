"""REGENIE native split-trait file layout and C++ default numeric precision."""
from __future__ import annotations
from pathlib import Path
import gzip
import math
import json

RESULT_COLUMNS = ('CHROM','GENPOS','ID','ALLELE0','ALLELE1','A1FREQ','N',
                  'TEST','BETA','SE','CHISQ','LOG10P','EXTRA')


def _native(value):
    if value is None: return 'NA'
    if isinstance(value, str):
        if any(c.isspace() for c in value): raise ValueError('Result field contains whitespace')
        return value
    if isinstance(value, int): return str(value)
    number = float(value)
    return format(number, '.6g') if math.isfinite(number) else 'NA'


class RegenieWriter:
    """Write prefix_phenotype.regenie, optional .gz, optional sample IDs.

    Additional audit metadata is kept in a separate JSON; it never adds columns
    to the original REGENIE result table.
    """
    def __init__(self, prefix, phenotype, *, masks=None, gzip_output=False,
                 sample_ids=None, write_samples=True, print_pheno_name=False,
                 split_by_pheno=True):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.phenotype = phenotype
        self.split = split_by_pheno
        self.path = Path(str(prefix)+(f'_{phenotype}' if self.split else '')+'.regenie'+('.gz' if gzip_output else ''))
        self.partial_path=Path(str(self.path)+'.partial')
        self.stream = gzip.open(self.partial_path,'wt') if gzip_output else self.partial_path.open('w')
        if masks is not None:
            definitions = []
            for mask in masks:
                order=getattr(mask,'category_order',None)
                definitions.append(f'{mask.name}="'+','.join(order if order is not None else sorted(mask.categories))+'"')
            self.stream.write('##MASKS=<'+ ';'.join(definitions)+'>\n')
        columns = list(RESULT_COLUMNS)
        self.dictionary_path=None
        if not self.split:
            for name in ('BETA','SE','CHISQ','LOG10P'):
                columns[columns.index(name)] = name+'.Y1'
            self.dictionary_path=Path(str(prefix)+'.regenie.Ydict')
            self.dictionary_partial=Path(str(self.dictionary_path)+'.partial')
            self.dictionary_partial.write_text('Y1 '+phenotype+'\n')
        self.stream.write(' '.join(columns)+'\n')
        self.rows = 0
        self.closed = False
        self.ids_path=None
        if write_samples and sample_ids is not None:
            self.ids_path = Path(str(prefix)+f'_{phenotype}.regenie.ids')
            self.ids_partial=Path(str(self.ids_path)+'.partial')
            with self.ids_partial.open('w') as ids:
                if print_pheno_name: ids.write(phenotype+'\tNA\n')
                for fid,iid in sorted(sample_ids, key=lambda s:s[0]+'_'+s[1]):
                    ids.write(fid+'\t'+iid+'\n')

    def write(self, row):
        self.stream.write(' '.join(_native(row.get(k,'NA')) for k in RESULT_COLUMNS)+'\n')
        self.rows += 1

    def close(self,commit=True):
        if self.closed:return
        self.closed=True
        self.stream.close()
        if commit:
            self.partial_path.replace(self.path)
            if self.ids_path:self.ids_partial.replace(self.ids_path)
            if self.dictionary_path:self.dictionary_partial.replace(self.dictionary_path)
        else:
            self.partial_path.unlink(missing_ok=True)
            if self.ids_path:self.ids_partial.unlink(missing_ok=True)
            if self.dictionary_path:self.dictionary_partial.unlink(missing_ok=True)

    def __enter__(self): return self
    def __exit__(self,*exc): self.close(commit=exc[0] is None)


def write_run_manifest(path, metadata):
    target=Path(path)
    partial=Path(str(target)+'.partial')
    partial.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str)+'\n')
    partial.replace(target)
