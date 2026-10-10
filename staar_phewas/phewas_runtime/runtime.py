"""Ordered independent association analyses with shared verified cache IO.

Each analysis retains the single-trait pipeline, output groups and model state.
Single scans are frame-major; gene jobs are family-major across all models.
"""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import inspect
import math
import threading
import time
import numpy as np
import torch

from .. import cli
from ..gds import SeqArrayGDS
from ..gds_device import DeviceMinorBlock
from ..io import load_null_model, fit_prepared_input, save_null_model
from ..compat import write_gaussian_null
from ..pipeline import AnalysisOptions, PheWASPipeline
from ..profiling import StageProfiler
from ..precision_audit import DenseProductAudit
from ..tf32 import configure_tf32, validate_split_k, execution_metadata as tf32_metadata
from ..statistics import statistics_execution_metadata
from ..cache_runtime.fast_container import Container
from ..r_output import write_association_batch
from .metadata import SharedMetadataReader
from .mask_limit import LimitedMaskPipeline
from .buffers import EffectiveBuffer
from .shared_state import SharedStateBroker
from .single import process_single_batches, execution_metadata as single_metadata

_LOCK = threading.Lock()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def validate_analyses(analyses):
    """Require aligned scientific schedules and separate original output files."""
    if not isinstance(analyses, (list, tuple)) or not analyses:
        raise ValueError('analyses must contain at least one standalone configuration')
    configs = [deepcopy(a) for a in analyses]
    signatures, outputs = [], {}
    input_paths = {str(Path(p[key]).resolve()) for c in configs for p in c.get('phenotypes', [])
                   for key in ('model', 'input', 'sample_indices_file') if key in p}
    shared_keys = ('qc_path', 'annotation_catalog', 'annotation_names', 'analysis_options',
        'packed_reader_directory', 'statistics_tail_optimization', 'local_mask_reuse',
        'weight_batch_optimization', 'weighted_eigensolver', 'maximum_mask_variants')
    for trait, config in enumerate(configs):
        if len(config.get('phenotypes', [])) != 1:
            raise ValueError('every analysis must contain exactly one independent phenotype')
        if config.get('matmul_mode', 'tf32') != 'tf32' or config.get('precision_control', False):
            raise ValueError('PheWAS production uses native TF32 with FP32 storage')
        if config.get('statistics_execution', 'serial') != 'serial':
            raise ValueError('PheWAS schedules use serial independent cores')
        if config.get('resident_genotypes', True) is not True or config.get('single_batch_optimization', True) is not True:
            raise ValueError('shared PheWAS requires resident genotypes and effective Single batches')
        validate_split_k(config.get('tf32_split_k', 0))
        obsolete = {'tf32_binned_tile_shape', 'tf32_binned_fused_small'} & config.keys()
        if obsolete:raise ValueError('TF32 reconstruction parameters are obsolete')
        for flag in ('precision_control', 'local_mask_reuse', 'weight_batch_optimization',
                     'statistics_tail_optimization', 'stage_profile', 'validation_reference', 'require_single_continuous'):
            if flag in config and type(config[flag]) is not bool:
                raise ValueError(flag + ' must be a JSON boolean')
        cli._weighted_eigensolver_settings(config, 'tf32', 'cuda:0')
        options = config.setdefault('analysis_options', {})
        if options.get('wrapper_semantics', 'base') != 'base':
            raise ValueError('single-trait equivalence requires base wrapper semantics')
        options['wrapper_semantics'] = 'base'
        budget = options.setdefault('memory_limit_gib', 20)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 0 < budget <= 20:
            raise ValueError('PheWAS memory_limit_gib must be finite and in (0, 20]')
        AnalysisOptions(**options)
        for field, default in [('individual_effective_block_size', 1024), ('individual_genotype_block_size', 1024)]:
            value = config.setdefault(field, default)
            if type(value) is not int or value < 1:
                raise ValueError(field + ' must be a positive integer')
        maximum = config.get('maximum_mask_variants')
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError('maximum_mask_variants must be a positive integer or omitted')
        for phenotype in config['phenotypes']:
            if ('model' in phenotype) == ('input' in phenotype):
                raise ValueError('each phenotype requires exactly one model or prepared input')
            fit = dict(phenotype.get('fit_options', {}))
            for key in ('family', 'joint_mode', 'binary_mode'):
                if key in phenotype and key in fit and phenotype[key] != fit[key]:
                    raise ValueError('conflicting phenotype fitting modes')
            if fit.get('matmul_mode', 'tf32') != 'tf32':
                raise ValueError('phenotype matmul_mode conflicts with native TF32')
            validate_split_k(fit.get('tf32_split_k', 0))
            if phenotype.get('joint_mode', fit.get('joint_mode')) is not None:
                raise ValueError('independent PheWAS requires single-trait models')
            for key in ('output_null', 'save_model'):
                if key not in phenotype:continue
                path = str(Path(phenotype[key]).resolve())
                if path in outputs or path in input_paths:
                    raise ValueError('null/model outputs must be distinct from every input and output')
                outputs[path] = (trait, key, None)
        schedule = []
        for chromosome in config.get('chromosomes', []):
            jobs = []
            for job in chromosome.get('jobs', []):
                if job.get('kind') not in ('individual', 'coding', 'noncoding', 'ncrna') or job.get('layout', 'base') != 'base':
                    raise ValueError('PheWAS accepts base Individual, coding, noncoding and ncRNA jobs')
                arguments = job.get('arguments', {})
                if not isinstance(arguments, dict):raise ValueError('job arguments must be an object')
                signature_args = inspect.signature(getattr(PheWASPipeline, job['kind'])).parameters
                permitted = set(signature_args) - {'self', 'chromosome'}
                if job['kind'] == 'noncoding':permitted.add('promoter_intervals_file')
                if set(arguments) - permitted:raise ValueError('unknown association job arguments')
                if job['kind'] == 'individual':
                    for key, default in (('mac_cutoff',20), ('subset_variants_num',5000)):
                        value = arguments.get(key, default)
                        if type(value) is not int or value < 1:raise ValueError(key + ' must be a positive integer')
                    if arguments.get('variant_type', 'variant') not in ('variant', 'SNV', 'Indel'):
                        raise ValueError('invalid Single variant_type')
                    if (arguments.get('start') is None) != (arguments.get('end') is None):
                        raise ValueError('provide both Single endpoints or neither')
                if not isinstance(job.get('output'), str) or Path(job['output']).suffix.lower() not in ('.rdata', '.rda', '.rds'):
                    raise ValueError('formal PheWAS outputs must be native R files')
                path = str(Path(job['output']).resolve())
                signature = (job['kind'], job.get('object_name'), job.get('layout', 'base'))
                if path in input_paths:
                    raise ValueError('association output cannot overwrite an input')
                if path in outputs:
                    owner, role, previous = outputs[path]
                    if owner != trait or role != 'association' or previous != signature or job['kind'] == 'individual':
                        raise ValueError('output collisions are allowed only for one trait and compatible gene batches')
                outputs[path] = (trait, 'association', signature)
                jobs.append({k: job.get(k) for k in ('name', 'kind', 'arguments', 'object_name', 'layout')})
            schedule.append({'name': chromosome['name'], 'gds': str(Path(chromosome['gds']).resolve()),
                             'annotation_index': chromosome.get('annotation_index'), 'jobs': jobs})
        if not schedule or not any(c['jobs'] for c in schedule):
            raise ValueError('provide a nonempty chromosome job schedule')
        signatures.append(_canonical({'settings': {k: config.get(k) for k in shared_keys}, 'schedule': schedule}))
    if len(set(signatures)) != 1:
        raise ValueError('independent analyses must share GDS, annotations, settings and ordered job arguments')
    return configs


class _NativeOutputs:
    def __init__(self, config):
        self.remaining = Counter(str(Path(j['output']).resolve()) for c in config['chromosomes'] for j in c['jobs'])
        self.groups, self.seconds, self.files = {}, 0., 0

    def append(self, job, result):
        path = Path(job['output']);key = str(path.resolve())
        signature = (job['kind'], job.get('object_name'), job.get('layout', 'base'))
        group = self.groups.setdefault(key, {'path': path, 'signature': signature, 'results': []})
        if group['signature'] != signature:
            raise ValueError('one output file cannot combine different native layouts or kinds')
        if job['kind'] == 'individual' and group['results']:
            raise ValueError('Individual jobs require separate native files')
        group['results'].append(result);self.remaining[key] -= 1
        if self.remaining[key] == 0:
            start = time.perf_counter();path.parent.mkdir(parents=True, exist_ok=True)
            write_association_batch(path, group['results'], kind=signature[0], object_name=signature[1], layout=signature[2])
            self.seconds += time.perf_counter()-start;self.files += 1
            group['results'].clear()

    def check(self):
        if any(self.remaining.values()) or any(g['results'] for g in self.groups.values()):
            raise RuntimeError('incomplete native output schedule')


def _load_models(configs, device):
    models, rows = [], []
    for config in configs:
        p = config['phenotypes'][0]
        if 'model' in p:
            model = load_null_model(p['model'], device=device, matmul_mode='tf32')
            row = np.load(p['sample_indices_file'], allow_pickle=False) if 'sample_indices_file' in p else None
        else:
            fit = dict(p.get('fit_options', {}))
            for key in ('family', 'binary_mode'):
                if key in p:fit[key] = p[key]
            fit.pop('tf32_split_k', None)
            family=fit.get('family', 'gaussian')
            if family != 'gaussian':raise ValueError('binary PheWAS requires a complete prefitted non-SPA state')
            fit['matmul_mode'] = 'tf32'
            model, row = fit_prepared_input(p['input'], device=device, transform=p.get('transform', 'none'), **fit)
        if model.n_pheno != 1 or model.use_spa or model.family not in ('gaussian', 'binomial'):
            raise ValueError('PheWAS requires independent single-trait Gaussian or non-SPA binary models')
        if model.family == 'binomial' and 'reference' in model.fit_method.lower() and not config.get('validation_reference', False):
            raise ValueError('an original R binary reference cache requires explicit validation_reference')
        if config.get('require_single_continuous', False) and model.family != 'gaussian':
            raise ValueError('require_single_continuous conflicts with a binary model')
        if model.x.dtype != torch.float32 or model.matmul_mode != 'tf32':
            raise RuntimeError('model storage must remain native FP32')
        models.append(model);rows.append(row)
    return models, rows


def _share_annotations(pipelines):
    first = pipelines[0]
    for pipeline in pipelines[1:]:
        pipeline._base_masks = first._base_masks
        pipeline._category_codes = first._category_codes
        pipeline._annotation_indexes = first._annotation_indexes
        pipeline._coding_masks_cache = first._coding_masks_cache


def _single(pipelines, broker, arguments, configs):
    chromosome=arguments['chromosome'];mac=arguments.get('mac_cutoff',20)
    subset=arguments.get('subset_variants_num',5000);variant_type=arguments.get('variant_type','variant')
    start, end = arguments.get('start'), arguments.get('end')
    if (start is None) != (end is None):raise ValueError('provide both Single endpoints or neither')
    first = pipelines[0]
    if start is None:
        selected = np.flatnonzero(first._base_mask(chromosome, variant_type))
    else:
        selected=first.region_indices(start,end)
        if variant_type == 'variant':selected=selected[first.qc[selected]=='PASS']
        else:
            from ..masks import variant_filter
            a=first.annotations(selected,include_weights=False,metadata='mask')
            selected=selected[variant_filter(a,variant_type)]
    buffers=[EffectiveBuffer(p.union_rows,c.get('individual_effective_block_size',1024),allocation_guard=broker._guard) for p,c in zip(pipelines,configs)]
    ordinals=[0]*len(pipelines);records=[[] for _ in pipelines]
    scan_step=configs[0]['individual_genotype_block_size']
    offset=0
    def consume(tail=False):
        nonlocal ordinals
        ready=[buffer.take(tail=tail) for buffer in buffers]
        if all(x is None for x in ready):return False
        result,ordinals=process_single_batches(pipelines,ready,ordinals,chromosome,mac,subset)
        for existing, block_rows in zip(records,result):existing.extend(block_rows)
        return True
    while offset < len(selected):
        frame=int(np.searchsorted(broker._starts,selected[offset],side='right')-1)
        stop=int(np.searchsorted(selected,broker._starts[frame]+broker._sizes[frame],side='left'))
        for begin in range(offset,stop,scan_step):
            raw=broker.read_states(selected[begin:min(begin+scan_step,stop)],minimum_mac_bound=mac)
            blocks=broker.trait_blocks(raw,[p.union_rows for p in pipelines],minimum_mac=mac)
            for block,buffer in zip(blocks,buffers):buffer.append(block)
            del blocks,block,raw
            while consume():pass
        offset=stop
    while consume(tail=True):pass
    return [p.individual_tables([rows]) for p,rows in zip(pipelines,records)]


def run_configuration(analyses, *, cache_specs, device='cuda:0', device_cache_bytes=512*2**20,
                      compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20,
                      cpu_threads=2):
    """Run standalone-compatible configs using existing verified genotype caches.

    ``analyses`` is a sequence of individual Torchstaar configurations with
    identical ordered scientific jobs and separate original output paths.
    ``cache_specs`` maps each source GDS path to the existing CacheSpec with a
    current source proof. Missingness and fitted model axes remain independent.
    ``cpu_threads`` is a positive integer controlling PyTorch CPU intra-op
    threads during this call, default 2. The previous process setting is restored
    on success or failure. This does not configure cache worker processes,
    inter-op threads, environment variables, or GPU scheduling.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    if not _LOCK.acquire(blocking=False):raise RuntimeError('PheWAS context is already running')
    previous_threads = None
    try:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(cpu_threads)
        effective_threads = torch.get_num_threads()
        if effective_threads != cpu_threads:
            raise RuntimeError('requested PyTorch CPU intra-op thread count was not applied')
        report = _run(analyses,cache_specs=cache_specs,device=device,device_cache_bytes=device_cache_bytes,
                      compact_cache_bytes=compact_cache_bytes,metadata_cache_bytes=metadata_cache_bytes)
        report['cpu_thread_control'] = dict(requested=cpu_threads, effective=effective_threads,
            previous=previous_threads, restored=True, scope='pytorch_intraop')
    finally:
        try:
            if previous_threads is not None:
                torch.set_num_threads(previous_threads)
        finally:
            _LOCK.release()
    return report


def _run(analyses, *, cache_specs, device, device_cache_bytes, compact_cache_bytes, metadata_cache_bytes):
    configs=validate_analyses(analyses)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('native PheWAS production requires CUDA')
    started=time.perf_counter();torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    budget=float(configs[0]['analysis_options'].get('memory_limit_gib',20))
    tf32_config=configure_tf32(split_k=0,memory_limit_gib=budget)
    tf32_metadata(reset=True);single_metadata(reset=True);statistics_execution_metadata(reset=True)
    before=time.perf_counter();models,rows=_load_models(configs,device)
    model_seconds=time.perf_counter()-before
    outputs=[_NativeOutputs(c) for c in configs]
    report={'schema_version':1,'matmul_mode':'tf32','models':[{'index':i,'n':m.n,'covariates':m.x.shape[1],
        'family':m.family,'use_spa':m.use_spa} for i,m in enumerate(models)],'jobs':[], 'job_groups':[], 'shared_readers':[]}
    specs={str(Path(p).resolve()):s for p,s in cache_specs.items()}
    interval_cache={};saved=False;setup_seconds=0.;annotation_seconds=0.
    from .. import _weighted_spectra
    solver=cli._weighted_eigensolver_settings(configs[0],'tf32',device)
    with _weighted_spectra.eigensolver_context(solver['effective'],memory_limit=int(budget*2**30)) as solver_state:
        with DenseProductAudit(forced=True) as audit:
            for ci,reference in enumerate(configs[0]['chromosomes']):
                source=str(Path(reference['gds']).resolve())
                if source not in specs:raise ValueError('every source GDS requires an explicit cache spec')
                spec=specs[source];expected=json.loads(json.dumps(spec.expected_binding,sort_keys=True,allow_nan=False))
                if not callable(spec.source_proof) or spec.source_proof()!=expected:
                    raise ValueError('current input/source proof differs from cache binding')
                begin=time.perf_counter()
                with ExitStack() as stack:
                    reader_options={} if configs[0].get('packed_reader_directory') is None else {'packed_reader_directory':configs[0]['packed_reader_directory']}
                    native=stack.enter_context(SeqArrayGDS(reference['gds'],**reader_options))
                    metadata=SharedMetadataReader(native,capacity_bytes=metadata_cache_bytes)
                    container=stack.enter_context(Container(spec.directory,expected,spec.expected_samples))
                    broker=stack.enter_context(SharedStateBroker(metadata,container,device=device,memory_limit_gib=budget,
                        device_cache_bytes=device_cache_bytes,compact_cache_bytes=compact_cache_bytes,own_reader=False,own_container=False))
                    def verify_after():
                        if spec.source_proof() != expected:raise ValueError('input/source binding changed during PheWAS')
                    stack.callback(verify_after)
                    metadata.pin_field('position')
                    metadata.pin_field(configs[0].get('qc_path','annotation/filter'))
                    pipelines=[]
                    catalog=configs[0].get('annotation_catalog',{})
                    if isinstance(catalog,str):catalog=json.loads(Path(catalog).read_text())
                    for i,(model,row,config) in enumerate(zip(models,rows,configs)):
                        physical=cli._bind_gds_samples(metadata,model,row,config['phenotypes'][0].get('sample_id_rule','auto'))
                        view=broker.reader_view(physical)
                        p=LimitedMaskPipeline(view,[model],qc_path=config.get('qc_path','annotation/filter'),
                            annotation_catalog=catalog,annotation_names=config.get('annotation_names',[]),
                            gds_sample_indices=[physical],options=AnalysisOptions(**config['analysis_options']))
                        p.resident_genotypes=True;p.single_batch_optimization=True
                        p.individual_effective_block_size=config['individual_effective_block_size']
                        p.statistics_execution='serial';p.statistics_tail_optimization=config.get('statistics_tail_optimization',True)
                        p.local_mask_reuse=config.get('local_mask_reuse',True);p.weight_batch_optimization=config.get('weight_batch_optimization',True)
                        p.maximum_mask_variants=config.get('maximum_mask_variants')
                        p.profiler=StageProfiler(device,enabled=config.get('stage_profile',False))
                        pipelines.append(p)
                    _share_annotations(pipelines)
                    setup_seconds+=time.perf_counter()-begin
                    categories=cli._scheduled_index_categories(reference['jobs'])
                    if categories:
                        index=reference.get('annotation_index',{})
                        filename=index.get('promoter_intervals_file')
                        if any(x.startswith('promoter_') for x in categories) and filename is None:
                            raise ValueError('promoter jobs require exact intervals')
                        if filename and filename not in interval_cache:interval_cache[filename]=cli._read_promoter_intervals(filename)
                        begin=time.perf_counter();pipelines[0].prepare_annotation_index(reference['name'],
                            promoter_intervals=interval_cache.get(filename),include_ncrna=False,categories=categories)
                        _share_annotations(pipelines);annotation_seconds+=time.perf_counter()-begin
                    if not saved:
                        for model,config in zip(models,configs):
                            p=config['phenotypes'][0]
                            if 'save_model' in p:
                                Path(p['save_model']).parent.mkdir(parents=True,exist_ok=True);save_null_model(model,p['save_model'])
                            if 'output_null' in p:
                                if model.family!='gaussian':raise ValueError('binary native null export remains unsupported')
                                write_gaussian_null(p['output_null'],model,original_sample_ids=model.gds_sample_ids,covariate_names=p.get('covariate_names'))
                        saved=True
                    for ji,job in enumerate(reference['jobs']):
                        kind=job['kind'];arguments=dict(job.get('arguments',{}));arguments['chromosome']=reference['name']
                        filename=arguments.pop('promoter_intervals_file',None)
                        if filename:
                            if filename not in interval_cache:interval_cache[filename]=cli._read_promoter_intervals(filename)
                            arguments['promoter_intervals']=interval_cache[filename]
                        print(json.dumps({'event':'started','chromosome_index':ci,'job_index':ji,'kind':kind}),flush=True)
                        begin=time.perf_counter()
                        if kind=='individual':
                            results=_single(pipelines,broker,arguments,configs)
                            computed=time.perf_counter()-begin
                            for ti,result in enumerate(results):
                                target=configs[ti]['chromosomes'][ci]['jobs'][ji]
                                outputs[ti].append(target,result)
                                report['jobs'].append({'trait_index':ti,'chromosome_index':ci,'job_index':ji,'kind':kind,
                                    'eligible_association_tests':cli._association_result_row_count(result,kind=kind)})
                            del results
                        else:
                            for ti,p in enumerate(pipelines):
                                result=getattr(p,kind)(**arguments)
                                outputs[ti].append(configs[ti]['chromosomes'][ci]['jobs'][ji],result)
                                report['jobs'].append({'trait_index':ti,'chromosome_index':ci,'job_index':ji,'kind':kind,
                                    'eligible_association_tests':cli._association_result_row_count(result,kind=kind)})
                                del result
                            computed=time.perf_counter()-begin
                        torch.cuda.synchronize(device)
                        report['job_groups'].append({'chromosome_index':ci,'job_index':ji,'kind':kind,
                            'compute_and_output_seconds':time.perf_counter()-begin})
                        print(json.dumps({'event':'finished','chromosome_index':ci,'job_index':ji,'kind':kind,'seconds':computed}),flush=True)
                    for p in pipelines:p.profiler.flush()
                    report['shared_readers'].append({'broker':broker.metrics,'metadata':dict(metadata.metrics),
                        'stage_profiles':[p.profiler.report() for p in pipelines],
                        'skipped_masks':[list(p.skipped_sets) for p in pipelines],
                        'local_mask_reuse':[dict(p.local_mask_reuse_counters) for p in pipelines],
                        'covariance_diagnostics':[list(getattr(p, 'covariance_diagnostics', [])) for p in pipelines]})
                del pipelines, p, metadata, native, container, broker
        report['dense_product_audit']=audit.report()
    for output in outputs:output.check()
    report.update(total_seconds=time.perf_counter()-started,model_seconds=model_seconds,setup_seconds=setup_seconds,
        annotation_seconds=annotation_seconds,native_output_seconds=sum(o.seconds for o in outputs),
        native_files=sum(o.files for o in outputs),single_execution=single_metadata(),tf32_execution=tf32_metadata(),
        statistics_execution_metadata=statistics_execution_metadata(),
        peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
        memory_limit_gib=budget,tf32_configuration=tf32_config,weighted_eigensolver_execution=dict(solver,**solver_state))
    report['eligible_association_tests']=sum(j['eligible_association_tests'] for j in report['jobs'])
    report['native_execution_status']=cli._native_execution_status(report['tf32_execution'],report['jobs'],planned_jobs=sum(len(c['jobs']) for config in configs for c in config['chromosomes']))
    if report['peak_gpu_mib']>budget*1024:raise MemoryError('observed PheWAS CUDA allocation exceeded configured budget')
    return report
