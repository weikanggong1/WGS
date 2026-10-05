"""REGENIE native split-trait file layout and C++ default numeric precision."""
from __future__ import annotations
from pathlib import Path
import gzip
import math
import json
from datetime import datetime, timedelta

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
        self.test_counts = {}
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
        test=row.get('TEST','NA')
        self.test_counts[test]=self.test_counts.get(test,0)+1

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


def write_log_header(stream, *, phenotype, analysis, options=None, settings=None,
                     started_at=None):
    """Readable REGENIE-style layout with the actual PyTorch engine identity."""
    import torch
    start=started_at or datetime.now().astimezone()
    stream.write('REGENIE-compatible analysis log\n')
    stream.write(f'Engine: torchwgs (PyTorch {torch.__version__})\n')
    stream.write(f'Start time: {start.isoformat(sep=" ",timespec="seconds")}\n\n')
    stream.write('Options in effect (REGENIE-equivalent):\n')
    for name,value in (options or {}).items():
        if value is None or value is False:continue
        stream.write(f'  --{name}'+('' if value is True else f' {value}')+'\n')
    stream.write(f'\nAnalysis: {analysis}\nPhenotype: {phenotype}\n')
    for name,value in (settings or {}).items():
        stream.write(f' * {name}: {value}\n')
    stream.flush()


def write_log_event(stream, event):
    """Keep runtime events readable; their exact structured form lives in JSONL."""
    stage=event.get('stage',event.get('event','progress'))
    details='; '.join(f'{name}={value}' for name,value in event.items()
                     if name not in ('stage','event'))
    stream.write(f' * {stage}'+(f': {details}' if details else '')+'\n')
    stream.flush()


def write_log_summary(stream, report):
    stream.write('\nAnalysis summary:\n')
    labels={'n_input':'# genotype samples selected','n':'# samples analyzed',
            'n_variants':'# input variants','source_variants':'# source variants',
            'variants_scanned':'# variants scanned to EOF','candidate_variants':'# candidate variants',
            'mask_definitions':'# mask definitions','header_mask_definitions':'# active mask definitions',
            'mask_artifact_variants':'# masks written','rows':'# association rows'}
    for name,label in labels.items():
        if name in report and report[name] is not None:
            stream.write(f' * {label}: {report[name]}\n')
    for name,count in report.get('tests',{}).items():
        stream.write(f' * test {name}: {count} rows\n')
    for name,stage in report.get('stages',{}).items():
        if isinstance(stage,dict):
            details='; '.join(f'{field}={stage[field]}' for field in
                ('n','rows','source_variants','variants_scanned','seconds','cached','imported') if field in stage)
            if details:stream.write(f' * stage {name}: {details}\n')
    if report.get('cached'):stream.write(' * cached result reused\n')
    if report.get('failed_attempt_seconds') is not None:
        stream.write(f' * earlier memory-budget attempt: {report["failed_attempt_seconds"]:.6f}s\n')
    for name,count in report.get('numerics',{}).items():
        if count:stream.write(f' * numerical diagnostic {name}: {count}\n')
    if report.get('error_type'):
        stream.write(f'ERROR: {report["error_type"]}: {report.get("error_message","")}\n')
    elapsed=float(report.get('seconds',report.get('total_seconds',0.)))
    stream.write(f'\nElapsed time : {elapsed:.6f}s\n')
    stream.write(f'End time: {datetime.now().astimezone().isoformat(sep=" ",timespec="seconds")}\n')
    stream.flush()


def write_regenie_log(path, *, phenotype, analysis, options=None, settings=None,
                      report):
    """Write plain text .log; manifests/progress remain separate JSON files."""
    target=Path(path);partial=Path(str(target)+'.partial')
    seconds=float(report.get('seconds',report.get('total_seconds',0.)))
    with partial.open('w') as stream:
        write_log_header(stream,phenotype=phenotype,analysis=analysis,
                         options=options,settings=settings,
                         started_at=datetime.now().astimezone()-timedelta(seconds=seconds))
        write_log_summary(stream,report)
    partial.replace(target)
