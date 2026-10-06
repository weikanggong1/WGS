"""Bounded, serial-GPU runtime sampling of the existing discovery functions.

This diagnostic writes anonymous timing JSON, never an association table. It
uses full sample sets and all masks for each sampled gene. A sampled gene is
never truncated to make it finish. The CLI supervises one worker with a hard
wall-clock budget; Python calls may also be interrupted at gene boundaries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from .runtime_estimate import (StageTimers, plan_windows, estimate_linear,
                               estimate_gene_strata, optimization_summary)


@dataclass
class ProbeConfig:
    sample_blocks: int = 8
    io_repeats: int = 2
    single_repeats: int = 2
    genes_per_stratum: int = 1
    gene_seconds: float = 15.0
    budget_seconds: float = 180.0
    seed: int = 0
    max_gpu_gb: float = 20.0
    run_gene: bool = True

    def __post_init__(self):
        for name in ('sample_blocks', 'io_repeats', 'single_repeats', 'genes_per_stratum'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        for name in ('gene_seconds', 'budget_seconds', 'max_gpu_gb'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        if not isinstance(self.run_gene, bool):
            raise ValueError('run_gene must be boolean')


class ProbeTimeout(TimeoutError):
    pass


def _atomic_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


@contextmanager
def _unit_deadline(seconds):
    """Cooperative interruption; the CLI parent supplies the hard deadline."""
    if not hasattr(signal, 'setitimer'):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def expired(signum, frame):
        raise ProbeTimeout('sample budget exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started))
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


class _BedReadProxy:
    """Time the production mmap gather without changing its indexing/copy."""
    def __init__(self, bed, timers):
        self.bed, self.timers = bed, timers
    def __getitem__(self, key):
        with self.timers.span('bed_page_access_and_cpu_copy'):
            return self.bed[key]
    def __getattr__(self, name):
        return getattr(self.bed, name)


class RuntimeInstrumentation:
    """Temporary measurement hooks; all originals are restored on exit.

    Inclusive parent costs are diagnostic. Only exclusive costs can be summed.
    ``packed_prepare_and_upload`` exclusive time includes sample-map setup and
    H2D; it is deliberately not labelled pure PCIe time. Synchronizing nested
    stages perturbs execution, so a second Single pass has no timing hooks.
    """
    def __init__(self, timers, reader=None):
        self.timers, self.reader = timers, reader
        self.geometry = {'masks': 0, 'vc_masks': 0, 'max_vc_columns': 0,
                         'sum_vc_columns': 0, 'max_sbat_columns': 0,
                         'decoded_columns': 0, 'packed_bytes': 0}
        self.stack = ExitStack()

    def _wrap(self, owner, name, stage, cuda=False):
        original = getattr(owner, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.timers.span(stage, cuda=cuda):
                result = original(*args, **kwargs)
            if stage == 'packed_prepare_and_upload':
                self.geometry['packed_bytes'] += result.packed.numel()
            elif stage == 'packed_decode':
                self.geometry['decoded_columns'] += result.shape[1]
            elif stage == 'sbat':
                self.geometry['max_sbat_columns'] = max(
                    self.geometry['max_sbat_columns'], len(args[0] if args else kwargs['score_vec']))
            return result
        self.stack.enter_context(patch.object(owner, name, wrapped))

    def __enter__(self):
        from . import io, single, gene, masks, statistics, _quadrature, mask_output, output
        try:
            if self.reader is not None:
                self.stack.enter_context(patch.object(self.reader, '_bed',
                    _BedReadProxy(self.reader._bed, self.timers)))
            hooks = [
                (io.BedReader, 'read_packed_block', 'packed_prepare_and_upload', True),
                (io.PackedBedBlock, 'allele_counts', 'packed_integer_counts', True),
                (io.PackedBedBlock, 'decode', 'packed_decode', True),
                (single, '_counts_from_observed', 'frequency_mac_arithmetic', True),
                (single.TestContext, 'residualize', 'covariate_projection', True),
                (single, '_score_counted_genotypes', 'single_score', True),
                (single, '_copy_result_columns', 'single_result_transfer', True),
                (masks.GeneMaskBuilder, 'update', 'mask_membership_and_reduction', True),
                (gene, '_score_covariance', 'burden_products', True),
                (gene, '_vc_score_covariance', 'vc_products', True),
                (gene, '_independent_columns', 'sbat_qr', True),
                (gene, '_materialize_gene_rows', 'gene_result_transfer', True),
                (statistics, 'skato_logp', 'skato', True),
                (statistics, '_association_tail', 'mixture_tail', True),
                (statistics, '_prepare_davies_spectrum', 'davies_spectrum_setup', True),
                (statistics, 'davies_logsf', 'davies', True),
                (statistics, 'kuonen_logsf', 'kuonen', True),
                (statistics, 'sbat_logp', 'sbat', True),
                (statistics, 'nnls_coefficients', 'nnls', True),
                (statistics, 'chi_bar_weights', 'sbat_weights', True),
                (statistics, 'normal_orthant_probability', 'normal_orthant', True),
                (statistics, 'acat_logp', 'acat', True),
                (_quadrature, 'integrate_log_qags', 'skato_quadrature', True),
                (_quadrature, 'integrate_log_gk21', 'skato_quadrature', True),
                (torch.linalg, 'eigh', 'eigensolver', True),
                (torch.linalg, 'eigvalsh', 'eigensolver', True),
                (mask_output.MaskWriter, '__call__', 'mask_pack_transfer_write', True),
                (output.RegenieWriter, '__init__', 'association_writer_setup', False),
                (output.RegenieWriter, 'write', 'association_text_write', False),
                (output.RegenieWriter, 'close', 'association_file_commit', False),
            ]
            for owner, name, stage, cuda in hooks:
                self._wrap(owner, name, stage, cuda)
            original = masks.GeneMaskBuilder.finish_iter
            @functools.wraps(original)
            def finished(builder):
                iterator = original(builder)
                try:
                    while True:
                        # Catch EOF *inside* the span: EOF is successful work.
                        with self.timers.span('mask_materialization', cuda=True):
                            item = next(iterator, None)
                        if item is None:
                            return
                        self.geometry['masks'] += 1
                        if item.vc_genotypes is not None:
                            columns = item.vc_genotypes.shape[1]
                            self.geometry['vc_masks'] += 1
                            self.geometry['sum_vc_columns'] += columns
                            self.geometry['max_vc_columns'] = max(
                                self.geometry['max_vc_columns'], columns)
                        yield item
                finally:
                    iterator.close()
            self.stack.enter_context(patch.object(masks.GeneMaskBuilder, 'finish_iter', finished))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)


def _candidate_stratum(size):
    return ('0001_0100' if size <= 100 else '0101_1000' if size <= 1000
            else '1001_5000' if size <= 5000 else '5001_plus')


def _gene_census(analyses, configuration, probe):
    """Stream setlists; keep only reservoirs of whole genes, not all site IDs.

    Counts are setlist-member upper bounds, not post-MAC VC dimensions. Actual
    VC dimensions are recorded when sampled genes run. Strata include the
    analysis ordinal so incompatible mask families are never pooled.
    """
    from .masks import GeneSet
    rng = random.Random(probe.seed)
    census, selected = {}, {}
    for ordinal, analysis in enumerate(analyses):
        with open(analysis.setlist_file) as stream:
            for line in stream:
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    continue
                if len(parts) != 4:
                    raise ValueError('Malformed setlist')
                name, chromosome, position, identifiers = parts
                if configuration.extract_genes is not None and name not in configuration.extract_genes:
                    continue
                ids = tuple(dict.fromkeys(identifiers.split(',')))
                label = 'a' + str(ordinal) + '_' + _candidate_stratum(len(ids))
                count = census[label] = census.get(label, 0) + 1
                reservoir = selected.setdefault(label, [])
                record = (ordinal, GeneSet(name, chromosome.removeprefix('chr'), int(position), ids))
                if len(reservoir) < probe.genes_per_stratum:
                    reservoir.append(record)
                else:
                    slot = rng.randrange(count)
                    if slot < probe.genes_per_stratum:
                        reservoir[slot] = record
    # Cover a large and a small stratum early; then round-robin analyses.
    labels = sorted(selected, key=lambda label: (label.split('_',1)[1], int(label.split('_',1)[0][1:])))
    jobs = [(label, record) for label in labels for record in selected[label]]
    if jobs:
        largest = max(range(len(jobs)), key=lambda j: len(jobs[j][1][1].variant_ids))
        jobs.insert(0, jobs.pop(largest))
    return [{'stratum': label, 'count': count} for label, count in census.items()], jobs


def _metadata_windows(reader, windows, timers):
    """One complete production BIM scan; retain only sampled metadata."""
    result = [[] for _ in windows]
    cursor = 0
    with timers.span('bim_full_scan_and_sample_metadata'):
        for variant in reader.iter_variants():
            while cursor < len(windows) and variant.index >= windows[cursor][1]:
                cursor += 1
            if cursor < len(windows) and windows[cursor][0] <= variant.index:
                result[cursor].append(variant)
    if any(len(records) != stop - start for records, (start, stop) in zip(result, windows)):
        raise ValueError('Incomplete sampled BIM metadata')
    return result


def _sample_metadata(reader, variants):
    """Bound iteration without changing count/decode/statistic implementations."""
    def blocks(instance, block_size, indices):
        if instance is not reader or list(indices) != [v.index for v in variants]:
            raise ValueError('Unexpected sampled metadata request')
        for start in range(0, len(variants), block_size):
            yield variants[start:start+block_size]
    return patch.object(type(reader), '_iter_variant_metadata_blocks', blocks)


def _load_context(inputs, chromosome, configuration, timers):
    from .io import BedReader, load_phenotype
    from .pipeline import _import_aligned_loco
    from .phenotype import prepare_phenotype
    from .single import create_test_context
    if not inputs.imported_loco:
        raise ValueError('Runtime sampling requires existing LOCO; fresh Step1 is outside this probe')
    with timers.span('array_reader_setup'):
        array = BedReader(inputs.array_prefix, keep=inputs.discovery_samples, remove=inputs.sample_remove)
    with timers.span('phenotype_load'):
        values = load_phenotype(inputs.phenotype_file, inputs.phenotype_column, array.sample_ids)
    covariates = inputs.covariates
    if configuration.phenotype_mode == 'raw':
        with timers.span('phenotype_preprocessing', cuda=True):
            prepared = prepare_phenotype(values, mode='raw', covariates=covariates,
                device=configuration.single_variant.device,
                outlier_sd=configuration.phenotype_outlier_sd,
                quantile_normalize=configuration.phenotype_quantile_normalize)
            values = torch.full_like(values, float('nan'))
            values[prepared.sample_indices] = prepared.values
        covariates = None
    with timers.span('loco_import'):
        null = _import_aligned_loco(inputs.imported_loco, inputs.phenotype_column, array.sample_ids)
    with timers.span('wgs_reader_setup'):
        reader = BedReader(inputs.wgs_prefixes[str(chromosome)], keep=inputs.discovery_samples,
                           remove=inputs.sample_remove)
    with timers.span('sample_alignment'):
        lookup = {sid: value for sid, value in zip(array.sample_ids, values.tolist())}
        phenotype = torch.tensor([lookup.get(sid, float('nan')) for sid in reader.sample_ids], dtype=torch.float64)
        x = None
        if covariates is not None:
            x_lookup = dict(zip(array.sample_ids, covariates))
            example = next(iter(x_lookup.values()))
            x = torch.stack([x_lookup.get(sid, torch.full_like(example, float('nan'))) for sid in reader.sample_ids])
        predictions = torch.full((reader.n_samples,), float('nan'), dtype=torch.float64)
        available = set(null.sample_ids)
        positions = [i for i, sid in enumerate(reader.sample_ids) if sid in available]
        predictions[positions] = null.align([reader.sample_ids[i] for i in positions])[
            :, null.chromosomes.index(int(chromosome))].to(torch.float64)
    with timers.span('context_setup', cuda=True):
        context = create_test_context(phenotype, predictions, covariates=x, sample_ids=reader.sample_ids,
            apply_rint=configuration.single_variant.apply_rint, device=configuration.single_variant.device,
            dtype=configuration.single_variant.dtype, tf32=configuration.single_variant.tf32)
    return reader, context


def _software_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob('*.py')}


def _gpu_snapshot(device):
    free, total = torch.cuda.mem_get_info(device)
    return {'free_bytes': int(free), 'total_bytes': int(total),
            'shared_used_bytes': int(total-free)}


def _process_io_snapshot():
    """Optional Linux accounting, not a physical storage bandwidth meter."""
    counters = {}
    try:
        for line in Path('/proc/self/io').read_text().splitlines():
            name, value = line.split(':',1)
            if name in ('read_bytes', 'rchar', 'syscr'):
                counters[name] = int(value.strip())
    except (OSError, ValueError):
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    counters.update(minor_page_faults=usage.ru_minflt, major_page_faults=usage.ru_majflt,
                    user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime)
    return counters


def _gene_estimate(report):
    estimate = estimate_gene_strata(report['gene_census'], report['gene_samples'])
    estimate['census_complete'] = report['gene_census_complete']
    if not report['gene_census_complete']:
        estimate.update(total_seconds=None, range_seconds=None)
    return estimate


def probe_discovery(inputs, chromosome, *, configuration=None, probe=None,
                    scratch_dir=None, progress=None):
    """Sample real inputs using paper/default override parameters on one GPU.

    ``progress(report)`` receives only anonymous JSON-compatible measurements.
    Python API deadlines are cooperative; use the CLI for a hard subprocess
    deadline. This never fits Step1, decompresses a chromosome or clears any
    OS/Triton cache. The report distinguishes unmeasured phases explicitly.
    """
    from .config import WGSConfig
    from .single import iter_single_variant_results
    from .gene import test_gene_based
    from .masks import load_annotations, load_mask_definitions, load_variant_whitelist
    from .mask_output import MaskWriter
    from .output import RegenieWriter
    from .statistics import diagnostics_scope
    configuration, probe = configuration or WGSConfig.paper(), probe or ProbeConfig()
    if str(chromosome) not in inputs.wgs_prefixes:
        raise ValueError('Requested chromosome absent from inputs')
    if not inputs.imported_loco:
        raise ValueError('Existing LOCO is required')
    device = torch.device(configuration.single_variant.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Runtime association probe requires CUDA')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if configuration.single_variant.genotype_reader != 'cuda_packed':
        raise ValueError('Runtime probe requires the production packed GPU reader')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(min(1., probe.max_gpu_gb*1024**3 /
        torch.cuda.get_device_properties(device).total_memory), device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_initialization_seconds = time.monotonic()-started
    timers = StageTimers(device=device)
    analyses = inputs.gene_analyses.get(str(chromosome), []) if probe.run_gene else []
    report = {'schema_version': 1, 'status': 'running', 'complete': False,
        'full_chromosome_measured': False, 'full_pipeline_measured': False,
        'scope': 'bounded_imported_loco_step2_samples',
        'cache': {'os_cache_state': 'uncontrolled', 'triton_cache_state': 'existing_not_cleared',
                  'single_file_pages': 'warmed_by_io_probe', 'storage_cache_evicted': False},
        'privacy': {'input_paths_exported': False, 'input_identifiers_exported': False,
                    'input_hashes_exported': False, 'association_tables_exported': False},
        'execution': {'workers': 1, 'serial_gpu': True, 'tf32': configuration.single_variant.tf32,
                      'single_dtype': configuration.single_variant.dtype, 'gene_dtype': 'float64',
                      'memory_cap_bytes': int(probe.max_gpu_gb*1024**3)},
        'software_sha256': _software_hashes(),
        'device': {'name': torch.cuda.get_device_name(device), 'torch': torch.__version__,
                   'before': _gpu_snapshot(device)},
        'io_samples': [], 'single_samples': [], 'gene_samples': [], 'gene_census': [],
        'gene_census_complete': not bool(analyses),
        'unmeasured': ['fresh_step1', 'full_annotation_preparation', 'summary', 'whole_pipeline'],
        'whole_pipeline_estimate_seconds': None}
    report['gpu_initialization_seconds'] = gpu_initialization_seconds
    def checkpoint():
        report['measured_wall_seconds'] = time.monotonic()-started
        report['setup_stages'] = timers.summary()
        report['estimates'] = {
            'bed_page_access_first_observed': estimate_linear(
                [dict(units=r['bytes'], seconds=r['seconds']) for r in report['io_samples'] if r['repeat']==0],
                report.get('source', {}).get('packed_payload_bytes', 0)),
            'bed_page_access_repeat': estimate_linear(
                [dict(units=r['bytes'], seconds=r['seconds']) for r in report['io_samples'] if r['repeat']>0],
                report.get('source', {}).get('packed_payload_bytes', 0)),
            'single_steady_without_bim_scan': estimate_linear(
                [dict(units=r['raw_sites'], seconds=r['seconds']) for r in report['single_samples']
                 if r['repeat']>0 and r['complete']], report.get('source', {}).get('variants', 0)),
            'gene_stratified': _gene_estimate(report)}
        if progress is not None:
            progress(report)
    def time_left():
        return probe.budget_seconds-(time.monotonic()-started)
    checkpoint()
    with tempfile.TemporaryDirectory(prefix='runtime_probe_', dir=scratch_dir) as temporary:
        scratch = Path(temporary)
        scratch.chmod(0o700)
        reader, context = _load_context(inputs, chromosome, configuration, timers)
        report['fixed_setup_seconds'] = sum(entry['seconds_exclusive'] for entry in timers.summary().values())
        report['source'] = {'source_samples': len(reader._all_sample_ids), 'selected_samples': reader.n_samples,
            'analysis_samples': len(context.y), 'variants': reader.n_variants,
            'source_stride_bytes': reader._stride, 'packed_payload_bytes': reader._stride*reader.n_variants}
        windows = plan_windows(reader.n_variants, configuration.single_variant.block_size,
                               probe.sample_blocks, probe.seed)
        metadata = _metadata_windows(reader, windows, timers)
        checkpoint()
        for repeat in range(probe.io_repeats):
            for ordinal, (start, stop) in enumerate(windows):
                if time_left() <= 0:
                    break
                io_before = _process_io_snapshot()
                t = time.perf_counter()
                raw = np.ascontiguousarray(reader._bed[np.arange(start, stop), :])
                seconds = time.perf_counter()-t
                io_after = _process_io_snapshot()
                report['io_samples'].append({'window': ordinal, 'repeat': repeat,
                    'bytes': raw.nbytes, 'seconds': seconds, 'raw_sites': stop-start,
                    'process_accounting_delta': {name: io_after[name]-io_before[name]
                        for name in io_before.keys() & io_after.keys()}})
                del raw
            checkpoint()
        for repeat in range(probe.single_repeats):
            for ordinal, ((start, stop), variants) in enumerate(zip(windows, metadata)):
                if time_left() <= 0:
                    break
                rows = 0
                local = StageTimers(device=device)
                instrumentation = RuntimeInstrumentation(local, reader)
                with ExitStack() as stack:
                    stack.enter_context(_sample_metadata(reader, variants))
                    if repeat == 0:
                        stack.enter_context(instrumentation)
                    with RegenieWriter(scratch/'single', inputs.phenotype_column,
                        sample_ids=context.sample_ids, write_samples=configuration.write_samples,
                        print_pheno_name=configuration.print_pheno_name,
                        split_by_pheno=configuration.split_by_pheno,
                        gzip_output=configuration.gzip_output) as writer:
                        torch.cuda.synchronize(device)
                        t = time.perf_counter()
                        for row in iter_single_variant_results(reader, context,
                            config=configuration.single_variant, variant_indices=range(start,stop)):
                            writer.write(row)
                            rows += 1
                        torch.cuda.synchronize(device)
                        seconds = time.perf_counter()-t
                report['single_samples'].append({'window': ordinal, 'repeat': repeat,
                    'raw_sites': stop-start, 'result_rows': rows, 'seconds': seconds, 'complete': True,
                    'instrumented': repeat == 0, 'stages': local.summary(),
                    'geometry': instrumentation.geometry if repeat == 0 else {}})
                checkpoint()
        if analyses and time_left() > 0:
            with timers.span('gene_setlist_census'):
                census, jobs = _gene_census(analyses, configuration.gene_based, probe)
            report['gene_census'] = census
            report['gene_census_complete'] = True
            report['planned_gene_samples'] = len(jobs)
            report['gene_stratification'] = 'analysis_ordinal_and_setlist_member_upper_bound'
            if configuration.bim_index_enabled:
                reader.bim_index_path = scratch/'bim.sqlite'
                with timers.span('bim_disk_index_prepare'):
                    reader.prepare_bim_index()
            checkpoint()
            for ordinal, (stratum, (analysis_index, gene_set)) in enumerate(jobs):
                if time_left() <= 0:
                    break
                analysis = analyses[analysis_index]
                local = StageTimers(device=device)
                measurement = {'sample': ordinal, 'stratum': stratum, 'complete': False,
                    'setlist_members': len(gene_set.variant_ids), 'seconds': 0., 'result_rows': 0}
                # Checkpoint before entering a potentially expensive gene.
                report['gene_samples'].append(measurement)
                checkpoint()
                t = time.perf_counter()
                instrument = RuntimeInstrumentation(local, reader)
                mask_writer = None
                diagnostics = None
                try:
                    with _unit_deadline(min(probe.gene_seconds, max(.001,time_left()))), instrument:
                        with local.span('sample_annotation_and_whitelist_load'):
                            annotations = load_annotations(analysis.annotation_file, {gene_set.gene})
                            definitions = load_mask_definitions(analysis.mask_definition_file)
                            candidates = {a.variant_id for a in annotations}
                            gene_config = replace(configuration.gene_based,
                                max_matrix_bytes=min(configuration.gene_based.max_matrix_bytes,
                                    int(max(.25, probe.max_gpu_gb-1)*1024**3)))
                            if gene_config.extract_variants is not None:
                                candidates.intersection_update(gene_config.extract_variants)
                            if analysis.variant_whitelist_file:
                                gene_config.extract_variants = load_variant_whitelist(
                                    analysis.variant_whitelist_file, candidates)
                            if gene_config.extract_variants is not None:
                                candidates.intersection_update(gene_config.extract_variants)
                        with local.span('sample_bim_lookup'):
                            lookup = reader.find_variants(candidates)
                        if configuration.write_masks:
                            with local.span('mask_writer_setup'):
                                active = context.sample_indices.tolist()
                                mask_writer = MaskWriter(scratch/'gene', context.sample_ids,
                                                         [reader.sample_sex[i] for i in active])
                        with RegenieWriter(scratch/'gene', inputs.phenotype_column, masks=definitions,
                            sample_ids=context.sample_ids, write_samples=configuration.write_samples,
                            print_pheno_name=configuration.print_pheno_name,
                            split_by_pheno=configuration.split_by_pheno,
                            gzip_output=configuration.gzip_output) as writer:
                            with local.span('whole_sampled_gene', cuda=True), diagnostics_scope() as diagnostics:
                                for row in test_gene_based(reader, context, annotations, [gene_set],
                                    definitions, gene_config, variant_lookup=lookup,
                                    artifact_callback=mask_writer):
                                    writer.write(row)
                                    measurement['result_rows'] += 1
                                measurement['diagnostics'] = dict(diagnostics)
                        if mask_writer is not None:
                            mask_writer.close()
                        measurement['complete'] = True
                        measurement['mask_definitions'] = len(definitions)
                        measurement['geometry'] = instrument.geometry
                        measurement['sample_lookup_sites'] = len(lookup)
                except Exception as error:
                    # Never export errors which could contain paths/IDs.
                    measurement['error_type'] = type(error).__name__
                    measurement['geometry'] = instrument.geometry
                    if mask_writer is not None and not mask_writer.closed:
                        mask_writer.close(commit=False)
                finally:
                    if diagnostics is not None:
                        measurement['diagnostics'] = dict(diagnostics)
                    measurement['stages'] = local.summary()
                    measurement['sample_elapsed_seconds'] = time.perf_counter()-t
                    measurement['seconds'] = measurement['stages'].get(
                        'whole_sampled_gene', {}).get('seconds_inclusive', 0.)
                    measurement['preparation_and_file_lifecycle_seconds'] = max(0.,
                        measurement['sample_elapsed_seconds']-measurement['seconds'])
                    checkpoint()
        io_complete = len(report['io_samples']) == len(windows)*probe.io_repeats
        single_complete = len(report['single_samples']) == len(windows)*probe.single_repeats
        gene_complete = (report['gene_census_complete']
                         and len(report['gene_samples']) == report.get('planned_gene_samples',0)
                         and all(sample['complete'] for sample in report['gene_samples']))
        report['sampling_coverage'] = {'io_plan_complete': io_complete,
            'single_plan_complete': single_complete, 'gene_plan_complete': gene_complete}
        report['status'] = ('budget_exhausted' if time_left() <= 0 else
                            'completed' if io_complete and single_complete and gene_complete
                            else 'samples_incomplete')
        report['complete'] = report['status'] == 'completed'
        report['device']['after'] = _gpu_snapshot(device)
        report['device']['process_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        report['device']['process_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        all_stages = {}
        for sample in report['single_samples'] + report['gene_samples']:
            for name, entry in sample['stages'].items():
                total = all_stages.setdefault(name, {'seconds_exclusive': 0., 'calls': 0})
                total['seconds_exclusive'] += entry['seconds_exclusive']
                total['calls'] += entry['calls']
        report['sample_stage_ranking'] = optimization_summary(all_stages)
        report['profile_changes_synchronization'] = True
        report['gene_times_include_sample_input_preparation'] = False
        report['storage_access_note'] = 'mmap page access plus CPU copy; not physical disk bandwidth'
        report['single_estimate_note'] = 'steady sampled windows; excludes BIM scan and setup; no full-chromosome claim'
        checkpoint()
    return report


def _load_inputs(path):
    from .pipeline import DiscoveryInputs, GeneAnalysis
    values = json.loads(Path(path).read_text())
    values['gene_analyses'] = {str(c): [GeneAnalysis(**a) for a in analyses]
        for c, analyses in values.get('gene_analyses', {}).items()}
    return DiscoveryInputs(**values)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded REGENIE serial-GPU runtime estimate')
    parser.add_argument('--inputs', required=True, help='Private DiscoveryInputs JSON used by the normal pipeline')
    parser.add_argument('--config', help='Optional WGSConfig JSON overrides')
    parser.add_argument('--chromosome', required=True)
    parser.add_argument('--out', required=True, help='Anonymous timing/coverage JSON')
    parser.add_argument('--scratch-dir', help='Private scratch directory for temporary native-format outputs/index')
    parser.add_argument('--sample-blocks', type=int, default=8)
    parser.add_argument('--single-repeats', type=int, default=2)
    parser.add_argument('--io-repeats', type=int, default=2)
    parser.add_argument('--genes-per-stratum', type=int, default=1)
    parser.add_argument('--gene-seconds', type=float, default=15.)
    parser.add_argument('--budget-seconds', type=float, default=180.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-gpu-gb', type=float, default=20.)
    parser.add_argument('--single-only', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    probe = ProbeConfig(args.sample_blocks, args.io_repeats, args.single_repeats,
        args.genes_per_stratum, args.gene_seconds, args.budget_seconds,
        args.seed, args.max_gpu_gb, not args.single_only)
    if args._worker:
        try:
            from .config import WGSConfig
            configuration = (WGSConfig.from_dict(json.loads(Path(args.config).read_text()))
                             if args.config else WGSConfig.paper())
            report = probe_discovery(_load_inputs(args.inputs), args.chromosome,
                configuration=configuration, probe=probe, scratch_dir=args.scratch_dir,
                progress=lambda value: _atomic_report(args.out,value))
            _atomic_report(args.out,report)
        except Exception as error:
            report = json.loads(Path(args.out).read_text()) if Path(args.out).is_file() else {}
            report.update(status='failed', complete=False, error_type=type(error).__name__,
                          full_pipeline_measured=False, whole_pipeline_estimate_seconds=None)
            _atomic_report(args.out,report)
            return 1
        return 0
    # One supervised worker. No shell interpolation, no background GPU queue.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if Path(args.out).exists():
        parser.error('Output already exists; choose a fresh report path')
    with tempfile.TemporaryDirectory(prefix='runtime_probe_supervisor_', dir=args.scratch_dir) as directory:
        with open(Path(directory)/'worker.log', 'wb') as log:
            child = subprocess.Popen([sys.executable, '-m', 'torchwgs.runtime_probe', *arguments, '--_worker'],
                                     stdout=log, stderr=log, start_new_session=True)
            try:
                returncode = child.wait(timeout=probe.budget_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                report = json.loads(Path(args.out).read_text()) if Path(args.out).is_file() else {}
                report.update(status='hard_budget_exhausted', complete=False,
                    full_pipeline_measured=False, whole_pipeline_estimate_seconds=None)
                _atomic_report(args.out,report)
                returncode = 2
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
    if not Path(args.out).is_file():
        _atomic_report(args.out, {'status': 'failed', 'complete': False,
            'error_type': 'WorkerExit', 'full_pipeline_measured': False,
            'whole_pipeline_estimate_seconds': None})
        if returncode == 0:
            returncode = 1
    print(json.dumps({'status': json.loads(Path(args.out).read_text()).get('status','failed'),
                      'returncode': returncode, 'private_fields_exported': False}))
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
