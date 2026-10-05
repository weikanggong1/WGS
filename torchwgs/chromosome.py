"""GPU chromosome and complete mask-group stages in deterministic input order."""
from dataclasses import asdict, replace
from pathlib import Path
import time
import torch
from .io import BedReader, materialize_bed
from .single import create_test_context, iter_single_variant_results
from .gene import test_gene_based
from .masks import (load_annotations, load_mask_definitions, load_variant_whitelist,
                    effective_mask_definitions)
from .output import RegenieWriter, write_run_manifest, write_regenie_log
from .mask_output import MaskWriter
from .statistics import diagnostics_scope


def _add_counters(total, values):
    for key, value in values.items():
        total[key] = total.get(key, 0) + value


def _association_log(output, *, chromosome, prefix, inputs, config, report, analysis=None):
    gene=analysis is not None
    params=config.gene_based if gene else config.single_variant
    options={'step':2,'chr':chromosome,'bed':prefix,'keep':inputs.discovery_samples,
             'remove':inputs.sample_remove,'phenoFile':inputs.phenotype_file,
             'phenoColList':inputs.phenotype_column,
             'pred':inputs.imported_loco or str(Path(output).parent.parent/'Step1/discovery_pred.list'),
             'qt':True,'apply-rint':params.apply_rint,'minMAC':params.min_mac,
             'bsize':params.variant_block_size if gene else params.block_size,
             'write-samples':config.write_samples,'print-pheno':config.print_pheno_name,
             'gz':config.gzip_output,'out':str(output)}
    settings={'device':config.single_variant.device,'context_dtype':config.single_variant.dtype,
              'association_dtype':'float64' if gene else config.single_variant.dtype,
              'TF32':bool(not gene and config.single_variant.tf32
                          and config.single_variant.dtype=='float32'
                          and torch.device(config.single_variant.device).type=='cuda'),
              'genotype_reader':params.genotype_reader}
    if gene:
        options.update({'anno-file':analysis.annotation_file,'set-list':analysis.setlist_file,
                        'mask-def':analysis.mask_definition_file,'extract':analysis.variant_whitelist_file,
                        'aaf-bins':','.join(str(x) for x in params.aaf_bins),
                        'vc-maxAAF':params.vc_max_aaf,'vc-tests':'skato,acato-full' if params.acato_full else 'skato,acato',
                        'rgc-gene-p':params.gene_p,'skip-sbat':not params.run_sbat,
                        'write-mask':config.write_masks,'write-mask-snplist':config.write_masks})
        settings.update(vc_storage=params.vc_storage,vc_score_method=params.vc_score_method,
                        davies_controller=params.davies_controller,
                        skato_integral_backend=params.skato_integral_backend)
    write_regenie_log(str(output)+'.log',phenotype=inputs.phenotype_column,
                      analysis='gene-based '+analysis.name if gene else 'single-variant association',
                      options=options,settings=settings,report=report)


def _gene_job(analysis, *, prefix, context, inputs, config, destination, key):
    from .pipeline import _write_stage_cache
    start = time.perf_counter()
    reader = BedReader(prefix, keep=inputs.discovery_samples, remove=inputs.sample_remove)
    gene_config = replace(config.gene_based)
    # Dense working sets and masks share one process-wide 20 GiB allocator cap.
    per_job = config.execution.max_gpu_gb / config.execution.concurrency
    gene_config.max_matrix_bytes = min(gene_config.max_matrix_bytes,
                                      int(max(.25, per_job - 1.) * 1024**3))
    # Native annotation categories are registered after variant extraction,
    # before set-list/gene filtering and before sample-dependent MAC/AAF tests.
    annotations=load_annotations(analysis.annotation_file)
    candidates={record.variant_id for record in annotations}
    if gene_config.extract_variants is not None:
        candidates.intersection_update(gene_config.extract_variants)
    if analysis.variant_whitelist_file:
        gene_config.extract_variants = load_variant_whitelist(analysis.variant_whitelist_file, candidates)
        candidates.intersection_update(gene_config.extract_variants)
    variant_lookup=reader.find_variants(candidates)
    definitions = load_mask_definitions(analysis.mask_definition_file)
    header_definitions=effective_mask_definitions(definitions,annotations,variant_lookup)
    output = destination / 'Gene' / f'discovery_c{key[0]}_{analysis.name}'
    active_sex = [reader.sample_sex[i] for i in context.sample_indices.tolist()]
    mask_writer = MaskWriter(output, context.sample_ids, active_sex) if config.write_masks else None
    status_path=Path(str(output)+'.progress.json')
    write_run_manifest(status_path, {'status':'running','rows':0,'mask_definitions':len(definitions)})
    last_progress=start
    writer=None
    with diagnostics_scope() as ledger:
        try:
            with RegenieWriter(output, inputs.phenotype_column, masks=header_definitions,
                               gzip_output=config.gzip_output, sample_ids=context.sample_ids,
                               write_samples=config.write_samples, print_pheno_name=config.print_pheno_name,
                               split_by_pheno=config.split_by_pheno) as writer:
                for row in test_gene_based(reader, context, annotations,
                                          analysis.setlist_file, definitions, gene_config,
                                          artifact_callback=mask_writer,variant_lookup=variant_lookup):
                    writer.write(row)
                    now=time.perf_counter()
                    if now-last_progress>=15.:
                        write_run_manifest(status_path, {'status':'running','rows':writer.rows,
                                                         'seconds':now-start})
                        last_progress=now
        except BaseException as error:
            if mask_writer:
                mask_writer.close(commit=False)
            write_run_manifest(status_path, {'status':'failed','rows':0 if writer is None else writer.rows,
                                             'seconds':time.perf_counter()-start,
                                             'error_type':type(error).__name__, 'error_message':str(error),
                                             'numerics':dict(ledger)})
            raise
        else:
            if mask_writer:
                mask_writer.close()
    report = {'rows': writer.rows, 'seconds': time.perf_counter()-start,
              'n': len(context.y),'n_input':reader.n_samples,
              'source_variants':reader.n_variants,'candidate_variants':len(variant_lookup),
              'tests':dict(writer.test_counts),'mask_definitions': len(definitions),
              'header_mask_definitions':len(header_definitions),
              'mask_artifact_variants':None if mask_writer is None else mask_writer.n_masks,
              'numerics': dict(ledger)}
    write_run_manifest(status_path, {'status':'completed',**report})
    _association_log(output,chromosome=key[0],prefix=prefix,inputs=inputs,config=config,
                     report=report,analysis=analysis)
    _write_stage_cache(output, key[1], writer, report, write_masks=config.write_masks)
    return analysis.name, str(writer.path), report


def run_chromosome(item, *, inputs, config, destination, null, null_key,
                   y_lookup, x_lookup, resume, run_single, run_gene):
    from .pipeline import (_file_identity, _fingerprint, _implementation_identity,
                           _stage_cache, _write_stage_cache)
    chromosome, original_prefix = item
    original_prefix = str(original_prefix)
    stages, single_files, gene_files, numerics = {}, [], [], {}
    source_identity = {suffix: _file_identity(original_prefix+suffix) for suffix in ('.bim','.fam')}
    bed_suffix = '.bed' if Path(original_prefix+'.bed').exists() else '.bed.gz'
    source_identity[bed_suffix] = _file_identity(original_prefix+bed_suffix)
    settings = {'gzip':config.gzip_output, 'split':config.split_by_pheno,
                'write_samples':config.write_samples, 'print_pheno':config.print_pheno_name}
    single_prefix = destination/'Single'/f'discovery_c{chromosome}'
    single_key = _fingerprint({'null':null_key, 'source':source_identity,
                              'parameters':asdict(config.single_variant), 'output':settings,
                              'implementation':_implementation_identity(['single','io','statistics','output','chromosome'])})
    single_cached = _stage_cache(single_prefix, single_key) if resume and run_single else None
    analyses = inputs.gene_analyses.get(str(chromosome), []) if run_gene else []
    gene_keys, gene_caches = {}, {}
    for analysis in analyses:
        source = {name:_file_identity(getattr(analysis,name)) for name in
                  ('annotation_file','setlist_file','mask_definition_file')}
        if analysis.variant_whitelist_file:
            source['variant_whitelist_file'] = _file_identity(analysis.variant_whitelist_file)
        gene_keys[analysis.name] = _fingerprint({
            'null':null_key, 'source':source_identity, 'gene_files':source,
            'parameters':asdict(config.gene_based), 'runtime':{
                'device':config.single_variant.device,'dtype':config.single_variant.dtype,
                'tf32':config.single_variant.tf32}, 'output':{**settings,'write_masks':config.write_masks},
            'implementation':_implementation_identity(['gene','masks','single','io','statistics',
                                                       '_norm_gpu','_rank_one','_davies_bounds','_quadrature',
                                                       'output','mask_output','chromosome'])})
        gene_caches[analysis.name] = _stage_cache(
            destination/'Gene'/f'discovery_c{chromosome}_{analysis.name}',
            gene_keys[analysis.name]) if resume else None
    if single_cached:
        single_files.append(single_cached['result'])
        stages[f'single_c{chromosome}'] = {'cached':True, **single_cached['report']}
    pending, results = [], []
    for analysis in analyses:
        cached = gene_caches[analysis.name]
        if cached:
            gene_files.append(cached['result'])
            stages[f'gene_c{chromosome}_{analysis.name}'] = {'cached':True, **cached['report']}
        else:
            pending.append(analysis)
    if (not run_single or single_cached) and not pending:
        return dict(chromosome=chromosome,cached=True,stages=stages,
                    single_files=single_files,gene_files=gene_files,numerics=numerics)
    prefix = materialize_bed(original_prefix, destination/'InputCache')
    reader = BedReader(prefix, keep=inputs.discovery_samples, remove=inputs.sample_remove)
    phenotype = torch.tensor([y_lookup.get(sid,float('nan')) for sid in reader.sample_ids],dtype=torch.float64)
    covariates = None
    if x_lookup is not None:
        example = next(iter(x_lookup.values()))
        covariates = torch.stack([x_lookup.get(sid,torch.full_like(example,float('nan'))) for sid in reader.sample_ids])
    prediction = torch.full((reader.n_samples,),float('nan'),dtype=torch.float64)
    null_ids = set(null.sample_ids)
    predicted_rows = [i for i,sid in enumerate(reader.sample_ids) if sid in null_ids]
    prediction[predicted_rows] = null.align([reader.sample_ids[i] for i in predicted_rows])[
        :,null.chromosomes.index(int(chromosome))].to(torch.float64)
    context = create_test_context(phenotype,prediction,sample_ids=reader.sample_ids,covariates=covariates,
                                  apply_rint=config.single_variant.apply_rint,device=config.single_variant.device,
                                  dtype=config.single_variant.dtype,tf32=config.single_variant.tf32)
    try:
        if run_single and not single_cached:
            start = time.perf_counter()
            with RegenieWriter(single_prefix, inputs.phenotype_column,
                               gzip_output=config.gzip_output,sample_ids=context.sample_ids,
                               write_samples=config.write_samples,print_pheno_name=config.print_pheno_name,
                               split_by_pheno=config.split_by_pheno) as writer:
                for row in iter_single_variant_results(reader,context,config=config.single_variant):
                    writer.write(row)
            # This stage iterates the entire reader without variant_indices.
            # Reaching this point certifies EOF; MAC-rejected sites were scanned.
            report = {'rows':writer.rows,'seconds':time.perf_counter()-start,'n':len(context.y),
                      'n_input':reader.n_samples,'source_variants':reader.n_variants,
                      'variants_scanned':reader.n_variants,'tests':dict(writer.test_counts)}
            stages[f'single_c{chromosome}'] = report
            single_files.append(str(writer.path))
            _association_log(single_prefix,chromosome=chromosome,prefix=prefix,
                             inputs=inputs,config=config,report=report)
            _write_stage_cache(single_prefix,single_key,writer,report)
        if pending:
            gene_context = context if config.gene_based.apply_rint == config.single_variant.apply_rint else create_test_context(
                phenotype,prediction,sample_ids=reader.sample_ids,covariates=covariates,
                apply_rint=config.gene_based.apply_rint,device=config.single_variant.device,
                dtype=config.single_variant.dtype,tf32=config.single_variant.tf32)
            function = lambda a: _gene_job(a,prefix=prefix,context=gene_context,inputs=inputs,
                                            config=config,destination=destination,
                                            key=(chromosome,gene_keys[a.name]))
            results = [function(analysis) for analysis in pending]
            for name,path,report in results:
                gene_files.append(path)
                stages[f'gene_c{chromosome}_{name}'] = report
                _add_counters(numerics,report['numerics'])
    finally:
        if Path(prefix).resolve()!=Path(original_prefix).resolve() and not config.keep_uncompressed_inputs:
            del reader
            Path(prefix+'.bed').unlink(missing_ok=True)
    # Preserve requested group order even when only some groups resumed.
    if analyses:
        gene_files = [gene_caches[a.name]['result'] if gene_caches[a.name] else
                      next(path for name,path,report in results if name==a.name) for a in analyses]
    return dict(chromosome=chromosome,cached=False,stages=stages,
                single_files=single_files,gene_files=gene_files,numerics=numerics)
