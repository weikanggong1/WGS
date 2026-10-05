"""Discovery pipeline: all association statistics are computed in PyTorch."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import hashlib
import json
import time
import torch
from .config import WGSConfig
from .io import BedReader, load_phenotype, resolve_variant_include
from .step1 import fit_null, NullModel
from .output import (write_run_manifest, write_log_header, write_log_event,
                     write_log_summary, write_regenie_log)
from .summary import summarize_results
from .statistics import numerical_diagnostics
from .phenotype import prepare_phenotype
from .chromosome import run_chromosome, _add_counters
from .execution import GpuExecutor


@dataclass
class GeneAnalysis:
    name: str
    annotation_file: str
    setlist_file: str
    mask_definition_file: str
    variant_whitelist_file: str | None = None


@dataclass
class DiscoveryInputs:
    array_prefix: str
    phenotype_file: str
    phenotype_column: str
    discovery_samples: str
    wgs_prefixes: dict[str, str]
    array_variant_include: str | None = None
    sample_remove: str | None = None
    gene_analyses: dict[str, list[GeneAnalysis]] = field(default_factory=dict)
    covariates: torch.Tensor | None = None
    imported_loco: str | None = None

    def __post_init__(self):
        self.wgs_prefixes={str(k):str(v) for k,v in self.wgs_prefixes.items()}
        self.gene_analyses={str(k):list(v) for k,v in self.gene_analyses.items()}
        if self.covariates is not None:self.covariates=torch.as_tensor(self.covariates,dtype=torch.float64)
        if not self.discovery_samples:raise ValueError('Specify the discovery sample inclusion file')
        for chrom,analyses in self.gene_analyses.items():
            if chrom not in self.wgs_prefixes:raise ValueError(f'Gene chromosome {chrom} has no WGS prefix')
            names=[a.name for a in analyses]
            if len(set(names))!=len(names):raise ValueError(f'Duplicate gene output names for chromosome {chrom}')


def _file_identity(path):
    p = Path(path)
    result={'path':str(p.resolve()),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    if p.stat().st_size<=64*1024**2:
        result['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def _fingerprint(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True,default=str).encode()).hexdigest()


def _implementation_identity(names):
    package=Path(__file__).parent
    return {name:hashlib.sha256((package/(name+'.py')).read_bytes()).hexdigest() for name in names}


def _read_manifest(path):
    try:
        result=json.loads(Path(path).read_text())
        return result if isinstance(result,dict) else {}
    except (ValueError,OSError):return {}


def _identities_match(identities):
    if not isinstance(identities,(list,tuple)) or not identities:return False
    for identity in identities:
        if not isinstance(identity,dict) or not isinstance(identity.get('path'),str):return False
        try:
            artifact=Path(identity['path'])
            if not artifact.is_file() or _file_identity(artifact)!=identity:return False
        except (OSError,ValueError):return False
    return True


def _prediction_identity(path,phenotype):
    identities=[_file_identity(path)]
    source=Path(path)
    if source.name.endswith('.list'):
        matches=[line.split(maxsplit=1) for line in source.read_text().splitlines() if line.strip()]
        matches=[parts for parts in matches if len(parts)==2 and parts[0]==phenotype]
        if len(matches)!=1:raise ValueError('Prediction list requires one matching phenotype')
        referenced=Path(matches[0][1])
        if not referenced.is_absolute():referenced=source.parent/referenced
        identities.append(_file_identity(referenced))
    return identities


def _import_aligned_loco(path,phenotype,sample_ids):
    """Resolve native FID_IID tokens using the known array FAM identities.

    FID and IID can both contain underscores, so their separator cannot be
    inferred from a LOCO token. The source may omit excluded samples; rows
    with NA predictions are filtered by NullModel.from_regenie itself.
    """
    import gzip
    identities=_prediction_identity(path,phenotype)
    source=Path(identities[-1]['path'])
    opener=gzip.open if source.suffix=='.gz' else open
    with opener(source,'rt',encoding='utf-8') as handle:
        header=handle.readline().split()
    if not header or header[0]!='FID_IID':
        raise ValueError('Expected REGENIE FID_IID header')
    known={}
    for fid,iid in sample_ids:
        token=f'{fid}_{iid}'
        if token in known:
            raise ValueError(f'FID_IID token collision in array FAM: {token}')
        known[token]=(fid,iid)
    present=set(header[1:])
    selected=[sid for token,sid in known.items() if token in present]
    if not selected:
        raise ValueError('No array FAM sample matches the imported LOCO header')
    return NullModel.from_regenie(path,phenotype_name=phenotype,sample_ids=selected)


def _stage_cache(prefix,key):
    path=Path(str(prefix)+'.manifest.json')
    if not path.exists():return None
    cached=_read_manifest(path)
    result=cached.get('result')
    if cached.get('key')!=key or not isinstance(result,str) or not result:return None
    try:
        if not Path(result).is_file():return None
    except (OSError,ValueError):return None
    if not isinstance(cached.get('report'),dict):return None
    if not _identities_match(cached.get('artifacts',[])):return None
    return cached


def _write_stage_cache(prefix,key,writer,report,*,write_masks=False):
    paths=[writer.path,Path(str(prefix)+'.log')]
    if writer.ids_path:paths.append(writer.ids_path)
    if not writer.split:paths.append(Path(str(prefix)+'.regenie.Ydict'))
    if write_masks:
        paths.extend(Path(str(prefix)+'_masks'+suffix) for suffix in ('.bed','.bim','.fam','.snplist'))
    write_run_manifest(str(prefix)+'.manifest.json',{'key':key,'result':str(writer.path),'report':report,
                                                   'artifacts':[_file_identity(p) for p in paths]})


def _write_step1_log(path, *, inputs, config, null, mode, seconds, n_input):
    """Describe this run's fit, cache load or external prediction import."""
    imported = mode == 'imported'
    fitting = null.metadata.get('config', asdict(config.step1))
    options = {'step':1,'bed':inputs.array_prefix,'keep':inputs.discovery_samples,
               'remove':inputs.sample_remove,'phenoFile':inputs.phenotype_file,
               'phenoColList':inputs.phenotype_column,'out':str(Path(path).with_suffix(''))}
    settings = {'mode':mode,'prediction_units':'standardized_step1_phenotype'}
    if imported:
        options['pred'] = inputs.imported_loco
        settings['operation'] = 'Load and align predictions fitted by the source run; no fitting performed'
    else:
        options.update({'extract':inputs.array_variant_include,'qt':True,
                        'apply-rint':fitting['apply_rint'],'bsize':fitting['block_size'],
                        'cv':fitting['folds'],'lowmem':fitting['l0_storage']=='memmap',
                        'l0':','.join(str(x) for x in fitting['ridge_l0']),
                        'l1':','.join(str(x) for x in fitting['ridge_l1'])})
        settings.update(device=fitting['device'],dtype=fitting['dtype'],
                        TF32=bool(fitting['tf32'] and fitting['dtype']=='float32'
                                  and torch.device(fitting['device']).type=='cuda'),
                        operation='Fit ridge/LOCO' if mode=='fitted' else 'Load cached ridge/LOCO; no fitting performed')
        # The fit duration stays labelled as a previous fit when loading a cache;
        # Elapsed time below always describes the current operation.
        if 'total_seconds' in null.metadata:
            label = 'fit_seconds' if mode=='fitted' else 'original_fit_seconds'
            settings[label] = null.metadata['total_seconds']
        for name in ('n_blocks','n_level0_predictors','fold_sizes','fold_active_sizes',
                     'selected_ridge_l1_index','selected_ridge_l1_h','cv_mse','timings'):
            if name in null.metadata:settings[name] = null.metadata[name]
    report = {'n_input':n_input,'n':len(null.sample_ids),'seconds':seconds,
              'cached':mode=='cached'}
    if not imported and 'n_variants' in null.metadata:
        report['n_variants'] = null.metadata['n_variants']
    write_regenie_log(path,phenotype=inputs.phenotype_column,analysis='ridge/LOCO Step1',
                      options=options,settings=settings,report=report)


def run_discovery(inputs: DiscoveryInputs, *, config=None, output_dir,
                  run_single=True, run_gene=True, resume=True):
    """Run one phenotype over caller-specified discovery chromosomes/masks.

    Pass all 22 WGS chromosomes for a genome-wide run.  Cached Step1 is accepted
    only when genotype/phenotype/cohort/QC/fitting configuration agrees. Changes
    to Step2 thresholds/masks leave the compatible Step1 cache usable.
    """
    config = WGSConfig.paper() if config is None else config
    config.execution.__post_init__()
    if any(torch.device(value).type != 'cuda' for value in
           (config.step1.device, config.single_variant.device)):
        raise ValueError('Discovery association requires CUDA for Step1 and Step2')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; CPU association fallback is disabled')
    if config.phenotype_mode not in ('raw','residual'):
        raise ValueError('phenotype_mode must be raw or residual')
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    numerical_start=numerical_diagnostics()
    gpu_devices=[]
    for requested in (config.step1.device, config.single_variant.device):
        device=torch.device(requested)
        if device.type!='cuda':continue
        if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        if device not in gpu_devices:gpu_devices.append(device)
    # Set the current request's cap before any phenotype or ridge allocation.
    # A preceding run's smaller cap must not leak into this run's Step1.
    for device in gpu_devices:
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.,config.execution.max_gpu_gb*1024**3/total),device)
        torch.cuda.reset_peak_memory_stats(device)
    metadata = {'engine':'torchwgs', 'cohort':'discovery', 'phenotype':inputs.phenotype_column,
                'configuration':config.to_dict(), 'inputs':{}, 'stages':{}}
    for name,path in [('phenotype',inputs.phenotype_file),('discovery_samples',inputs.discovery_samples),
                      ('array_bed',inputs.array_prefix+'.bed'),('array_bim',inputs.array_prefix+'.bim'),
                      ('array_fam',inputs.array_prefix+'.fam'),('array_variant_include',inputs.array_variant_include),
                      ('sample_remove',inputs.sample_remove)]:
        if path: metadata['inputs'][name] = _file_identity(path)
    null_key = _fingerprint({'inputs':metadata['inputs'],'phenotype':inputs.phenotype_column,'phenotype_mode':config.phenotype_mode,
                             'phenotype_quantile_normalize':config.phenotype_quantile_normalize,
                             'phenotype_outlier_sd':config.phenotype_outlier_sd,
                             'implementation':_implementation_identity(['step1','phenotype','io','_packed_gpu','pipeline']),
                             'imported_loco':None if not inputs.imported_loco else _prediction_identity(inputs.imported_loco,inputs.phenotype_column),
                             'step1':asdict(config.step1),'covariates':None if inputs.covariates is None else
                             hashlib.sha256(inputs.covariates.cpu().numpy().tobytes()).hexdigest()})
    array = BedReader(inputs.array_prefix,keep=inputs.discovery_samples,remove=inputs.sample_remove)
    y_array = load_phenotype(inputs.phenotype_file,inputs.phenotype_column,array.sample_ids)
    analysis_covariates=inputs.covariates
    if config.phenotype_mode=='raw':
        prepared=prepare_phenotype(y_array,mode='raw',covariates=inputs.covariates,device=config.step1.device,
                                  outlier_sd=config.phenotype_outlier_sd,quantile_normalize=config.phenotype_quantile_normalize)
        y_array=torch.full_like(y_array,float('nan'))
        y_array[prepared.sample_indices]=prepared.values
        analysis_covariates=None
        metadata['stages']['phenotype']=prepared.metadata
    y_lookup={sid:float(y) for sid,y in zip(array.sample_ids,y_array)}
    x_lookup=None if analysis_covariates is None else {sid:analysis_covariates[i] for i,sid in enumerate(array.sample_ids)}
    null_dir = destination/'Step1'
    null_dir.mkdir(exist_ok=True)
    cache_path = null_dir/'null_model.pt'
    manifest_path = null_dir/'cache.json'
    step1_log_path = null_dir/'discovery.log'
    local_null_files = [cache_path,cache_path.with_suffix('.json'),
                        null_dir/'discovery_1.loco',null_dir/'discovery_pred.list']
    null_cached=_read_manifest(manifest_path) if resume else {}
    log_path = destination/'discovery.log'
    events_path = destination/'discovery.events.jsonl'
    legacy_events=None
    if resume and log_path.exists():
        with log_path.open() as previous_log:
            if previous_log.readline().lstrip().startswith('{'):
                legacy_events=log_path.read_text()
    with log_path.open('a' if resume and legacy_events is None else 'w') as log, events_path.open('a' if resume else 'w') as events:
        if legacy_events is not None:events.write(legacy_events)
        write_log_header(log,phenotype=inputs.phenotype_column,analysis='discovery pipeline',
                         options={'phenoFile':inputs.phenotype_file,'phenoColList':inputs.phenotype_column,
                                  'keep':inputs.discovery_samples,'remove':inputs.sample_remove},
                         settings={'cohort':'discovery','chromosomes':','.join(inputs.wgs_prefixes),
                                   'device':config.single_variant.device,'dtype':config.single_variant.dtype,
                                   'TF32 single-precision operations':bool(config.single_variant.tf32
                                      and config.single_variant.dtype=='float32'
                                      and torch.device(config.single_variant.device).type=='cuda'),
                                   'write_masks':config.write_masks})
        def progress(event):
            events.write(json.dumps(event,default=str)+'\n');events.flush()
            write_log_event(log,event)
        t = time.perf_counter()
        before_step1_peaks={str(device):torch.cuda.max_memory_allocated(device) for device in gpu_devices}
        if inputs.imported_loco:
            null = _import_aligned_loco(inputs.imported_loco,inputs.phenotype_column,array.sample_ids)
            source = _prediction_identity(inputs.imported_loco,inputs.phenotype_column)
            step1_files = [Path(identity['path']) for identity in source]
            step1_mode = 'imported'
            metadata['stages']['step1'] = {'imported':True,'source':source}
        elif (resume and cache_path.exists() and null_cached.get('key')==null_key
              and _identities_match(null_cached.get('artifacts',[]))
              and {str(p.resolve()) for p in [*local_null_files,step1_log_path]}.issubset(
                  identity['path'] for identity in null_cached['artifacts'])):
            null = NullModel.load(cache_path)
            step1_files = local_null_files
            step1_mode = 'cached'
            metadata['stages']['step1'] = {'cached':True,**null.metadata}
        else:
            selected = None if inputs.array_variant_include is None else resolve_variant_include(array,inputs.array_variant_include)
            fitting_config=replace(config.step1,max_gpu_gb=min(
                                   config.step1.max_gpu_gb,config.execution.max_gpu_gb))
            null = fit_null(array,y_array,config=fitting_config,covariates=analysis_covariates,
                            output_dir=null_dir,phenotype_name=inputs.phenotype_column,
                            variant_indices=selected,progress_callback=progress)
            native_files=null.export_regenie(null_dir/'discovery',phenotype_name=inputs.phenotype_column)
            step1_files = [cache_path,cache_path.with_suffix('.json'),*native_files]
            step1_mode = 'fitted'
            metadata['stages']['step1'] = dict(null.metadata)
        metadata['stages']['step1_wall_seconds'] = time.perf_counter()-t
        _write_step1_log(step1_log_path,inputs=inputs,config=config,null=null,mode=step1_mode,
                         seconds=metadata['stages']['step1_wall_seconds'],n_input=array.n_samples)
        step1_files.append(step1_log_path)
        # Cache hits rewrite the log for the current load. Record its new identity
        # only after the atomic log commit, leaving model and prediction bytes intact.
        write_run_manifest(manifest_path,{'key':null_key,'mode':step1_mode,
                            'artifacts':[_file_identity(p) for p in step1_files]})
        metadata['stages']['step1'].update(mode=step1_mode,files=[str(p) for p in step1_files])
        single_files, gene_files = [],[]
        worker_numerics = {}
        def chromosome_job(item, job_config=config, job_resume=resume):
            return run_chromosome(item, inputs=inputs, config=job_config,
                                  destination=destination, null=null, null_key=null_key,
                                  y_lookup=y_lookup, x_lookup=x_lookup, resume=job_resume,
                                  run_single=run_single, run_gene=run_gene)
        chromosomes = list(inputs.wgs_prefixes.items())
        with GpuExecutor(config.execution, config.single_variant.device) as executor:
            chromosome_reports = executor.map(chromosome_job, chromosomes)
        for report in chromosome_reports:
            metadata['stages'].update(report['stages'])
            single_files.extend(report['single_files'])
            gene_files.extend(report['gene_files'])
            _add_counters(worker_numerics, report['numerics'])
            progress({'stage': 'chromosome_cached' if report['cached'] else 'chromosome_completed',
                      'chromosome': report['chromosome']})
        metadata['summary'] = summarize_results(single_files,gene_files,output_dir=destination/'Summary',config=config.significance)
        metadata['output_files']={'step1':[str(p) for p in step1_files],
                                  'single':single_files,'gene':gene_files}
        metadata['numerics']={key:value-numerical_start.get(key,0) for key,value in numerical_diagnostics().items()}
        _add_counters(metadata['numerics'], worker_numerics)
        metadata['total_seconds']=time.perf_counter()-start
        if gpu_devices:
            metadata['peak_gpu_by_device_bytes']={str(device):max(before_step1_peaks[str(device)],
                    torch.cuda.max_memory_allocated(device)) for device in gpu_devices}
            metadata['peak_gpu_bytes']=max(metadata['peak_gpu_by_device_bytes'].values())
        write_run_manifest(destination/'run_manifest.json',metadata)
        write_log_summary(log,metadata)
    return metadata
