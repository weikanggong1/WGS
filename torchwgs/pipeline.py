"""Discovery pipeline: all association statistics are computed in PyTorch."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import hashlib
import json
import time
import torch
from .config import WGSConfig
from .io import BedReader, load_phenotype, resolve_variant_include, materialize_bed
from .step1 import fit_null, NullModel
from .single import create_test_context, iter_single_variant_results
from .gene import test_gene_based
from .masks import load_mask_definitions, load_variant_whitelist
from .output import RegenieWriter, write_run_manifest
from .summary import summarize_results
from .statistics import numerical_diagnostics
from .phenotype import prepare_phenotype
from .mask_output import MaskWriter


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
    try:return json.loads(Path(path).read_text())
    except (ValueError,OSError):return {}


def _identities_match(identities):
    if not identities:return False
    for identity in identities:
        artifact=Path(identity['path'])
        if not artifact.is_file() or _file_identity(artifact)!=identity:return False
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


def _stage_cache(prefix,key):
    path=Path(str(prefix)+'.manifest.json')
    if not path.exists():return None
    cached=_read_manifest(path)
    if cached.get('key')!=key or not Path(cached.get('result','')).is_file():return None
    if not _identities_match(cached.get('artifacts',[])):return None
    return cached


def _write_stage_cache(prefix,key,writer,report,*,write_masks=False):
    paths=[writer.path]
    if writer.ids_path:paths.append(writer.ids_path)
    if not writer.split:paths.append(Path(str(prefix)+'.regenie.Ydict'))
    if write_masks:
        paths.extend(Path(str(prefix)+'_masks'+suffix) for suffix in ('.bed','.bim','.fam','.snplist'))
    write_run_manifest(str(prefix)+'.manifest.json',{'key':key,'result':str(writer.path),'report':report,
                                                   'artifacts':[_file_identity(p) for p in paths]})


def run_discovery(inputs: DiscoveryInputs, *, config=None, output_dir,
                  run_single=True, run_gene=True, resume=True):
    """Run one phenotype over caller-specified discovery chromosomes/masks.

    Pass all 22 WGS chromosomes for a genome-wide run.  Cached Step1 is accepted
    only when genotype/phenotype/cohort/QC/fitting configuration agrees. Changes
    to Step2 thresholds/masks leave the compatible Step1 cache usable.
    """
    config = WGSConfig.paper() if config is None else config
    if config.phenotype_mode not in ('raw','residual'):
        raise ValueError('phenotype_mode must be raw or residual')
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    numerical_start=numerical_diagnostics()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
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
                             'implementation':_implementation_identity(['step1','phenotype','io']),
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
    null_cached=_read_manifest(manifest_path) if resume else {}
    log_path = destination/'discovery.log'
    with log_path.open('a' if resume else 'w') as log:
        def progress(event):
            log.write(json.dumps(event,default=str)+'\n'); log.flush()
        t = time.perf_counter()
        if inputs.imported_loco:
            null = NullModel.from_regenie(inputs.imported_loco,phenotype_name=inputs.phenotype_column)
            metadata['stages']['step1'] = {'imported':True,'source':_prediction_identity(inputs.imported_loco,inputs.phenotype_column)}
        elif resume and cache_path.exists() and null_cached.get('key')==null_key and _identities_match(null_cached.get('artifacts',[])):
            null = NullModel.load(cache_path)
            metadata['stages']['step1'] = {'cached':True,**null.metadata}
        else:
            selected = None if inputs.array_variant_include is None else resolve_variant_include(array,inputs.array_variant_include)
            null = fit_null(array,y_array,config=config.step1,covariates=analysis_covariates,
                            output_dir=null_dir,phenotype_name=inputs.phenotype_column,
                            variant_indices=selected,progress_callback=progress)
            native_files=null.export_regenie(null_dir/'discovery',phenotype_name=inputs.phenotype_column)
            write_run_manifest(manifest_path,{'key':null_key,'artifacts':[_file_identity(p) for p in
                                [cache_path,cache_path.with_suffix('.json'),*native_files]]})
            metadata['stages']['step1'] = null.metadata
        metadata['stages']['step1_wall_seconds'] = time.perf_counter()-t
        single_files, gene_files = [],[]
        null_ids = set(null.sample_ids)
        for chromosome,prefix in inputs.wgs_prefixes.items():
            original_prefix=str(prefix)
            source_identity={suffix:_file_identity(original_prefix+suffix) for suffix in ('.bim','.fam')}
            bed_suffix='.bed' if Path(original_prefix+'.bed').exists() else '.bed.gz'
            source_identity[bed_suffix]=_file_identity(original_prefix+bed_suffix)
            output_settings={'gzip':config.gzip_output,'split':config.split_by_pheno,'write_samples':config.write_samples}
            single_prefix=destination/'Single'/f'discovery_c{chromosome}'
            single_key=_fingerprint({'null':null_key,'source':source_identity,'parameters':asdict(config.single_variant),
                                     'output':output_settings,'implementation':_implementation_identity(['single','statistics','output'])})
            single_cached=_stage_cache(single_prefix,single_key) if resume and run_single else None
            gene_caches={};gene_keys={}
            for analysis in inputs.gene_analyses.get(str(chromosome),[]) if run_gene else []:
                source={name:_file_identity(getattr(analysis,name)) for name in ['annotation_file','setlist_file','mask_definition_file']}
                if analysis.variant_whitelist_file:source['variant_whitelist_file']=_file_identity(analysis.variant_whitelist_file)
                key=_fingerprint({'null':null_key,'source':source_identity,'gene_files':source,'parameters':asdict(config.gene_based),
                                  'runtime':{'device':config.single_variant.device,'dtype':config.single_variant.dtype,'tf32':config.single_variant.tf32},
                                  'output':{**output_settings,'write_masks':config.write_masks},
                                  'implementation':_implementation_identity(['gene','masks','single','statistics','_norm_gpu','output','mask_output'])})
                gene_keys[analysis.name]=key
                gene_caches[analysis.name]=_stage_cache(destination/'Gene'/f'discovery_c{chromosome}_{analysis.name}',key) if resume else None
            if (not run_single or single_cached) and all(gene_caches.values()):
                if single_cached:
                    single_files.append(single_cached['result']);metadata['stages'][f'single_c{chromosome}']={'cached':True,**single_cached['report']}
                for name,cached in gene_caches.items():
                    gene_files.append(cached['result']);metadata['stages'][f'gene_c{chromosome}_{name}']={'cached':True,**cached['report']}
                progress({'stage':'chromosome_cached','chromosome':chromosome})
                continue
            prefix=materialize_bed(prefix,destination/'InputCache')
            reader = BedReader(prefix,keep=inputs.discovery_samples,remove=inputs.sample_remove)
            phenotype=torch.tensor([y_lookup.get(sid,float('nan')) for sid in reader.sample_ids],dtype=torch.float64)
            covariates=None if x_lookup is None else torch.stack([x_lookup.get(sid,torch.full_like(analysis_covariates[0],float('nan')))
                                                                for sid in reader.sample_ids])
            prediction=torch.full((reader.n_samples,),float('nan'),dtype=torch.float64)
            predicted_rows=[i for i,sid in enumerate(reader.sample_ids) if sid in null_ids]
            predicted_ids=[reader.sample_ids[i] for i in predicted_rows]
            prediction[predicted_rows]=null.align(predicted_ids)[:,null.chromosomes.index(int(chromosome))].to(torch.float64)
            context = create_test_context(phenotype,prediction,sample_ids=reader.sample_ids,covariates=covariates,
                          apply_rint=config.single_variant.apply_rint,device=config.single_variant.device,
                          dtype=config.single_variant.dtype,tf32=config.single_variant.tf32)
            if run_single:
                if single_cached:
                    single_files.append(single_cached['result']);metadata['stages'][f'single_c{chromosome}']={'cached':True,**single_cached['report']}
                else:
                    t = time.perf_counter()
                    out_prefix = single_prefix
                    with RegenieWriter(out_prefix,inputs.phenotype_column,gzip_output=config.gzip_output,
                                       sample_ids=context.sample_ids,write_samples=config.write_samples,
                                       split_by_pheno=config.split_by_pheno) as writer:
                        for row in iter_single_variant_results(reader,context,config=config.single_variant): writer.write(row)
                        single_files.append(str(writer.path))
                        metadata['stages'][f'single_c{chromosome}'] = {'rows':writer.rows,'seconds':time.perf_counter()-t,'n':len(context.y)}
                    Path(str(out_prefix)+'.log').write_text(json.dumps(metadata['stages'][f'single_c{chromosome}'],indent=2)+'\n')
                    _write_stage_cache(out_prefix,single_key,writer,metadata['stages'][f'single_c{chromosome}'])
            if run_gene:
                gene_context=context if config.gene_based.apply_rint==config.single_variant.apply_rint else create_test_context(
                    phenotype,prediction,sample_ids=reader.sample_ids,covariates=covariates,
                    apply_rint=config.gene_based.apply_rint,device=config.single_variant.device,
                    dtype=config.single_variant.dtype,tf32=config.single_variant.tf32)
                for analysis in inputs.gene_analyses.get(str(chromosome),[]):
                    cached=gene_caches[analysis.name]
                    if cached:
                        gene_files.append(cached['result']);metadata['stages'][f'gene_c{chromosome}_{analysis.name}']={'cached':True,**cached['report']}
                        continue
                    t = time.perf_counter()
                    gene_config = replace(config.gene_based)
                    if analysis.variant_whitelist_file:
                        gene_config.extract_variants = load_variant_whitelist(analysis.variant_whitelist_file)
                    definitions = load_mask_definitions(analysis.mask_definition_file)
                    out_prefix = destination/'Gene'/f'discovery_c{chromosome}_{analysis.name}'
                    active_sex=[reader.sample_sex[i] for i in gene_context.sample_indices.tolist()]
                    mask_writer=MaskWriter(out_prefix,gene_context.sample_ids,active_sex) if config.write_masks else None
                    try:
                        with RegenieWriter(out_prefix,inputs.phenotype_column,masks=definitions,
                                           gzip_output=config.gzip_output,sample_ids=gene_context.sample_ids,
                                           write_samples=config.write_samples,split_by_pheno=config.split_by_pheno) as writer:
                            for row in test_gene_based(reader,gene_context,analysis.annotation_file,analysis.setlist_file,
                                                       definitions,gene_config,artifact_callback=mask_writer): writer.write(row)
                            gene_files.append(str(writer.path))
                            stage_name=f'gene_c{chromosome}_{analysis.name}'
                            metadata['stages'][stage_name]={'rows':writer.rows,'seconds':time.perf_counter()-t,'n':len(context.y)}
                    except BaseException:
                        if mask_writer:mask_writer.close(commit=False)
                        raise
                    else:
                        if mask_writer:mask_writer.close()
                    Path(str(out_prefix)+'.log').write_text(json.dumps(metadata['stages'][stage_name],indent=2)+'\n')
                    _write_stage_cache(out_prefix,gene_keys[analysis.name],writer,metadata['stages'][stage_name],write_masks=config.write_masks)
            progress({'stage':'chromosome_completed','chromosome':chromosome})
            if Path(prefix).resolve()!=Path(original_prefix).resolve() and not config.keep_uncompressed_inputs:
                del reader
                Path(prefix+'.bed').unlink()
        metadata['summary'] = summarize_results(single_files,gene_files,output_dir=destination/'Summary',config=config.significance)
        metadata['output_files']={'single':single_files,'gene':gene_files}
        metadata['numerics']={key:value-numerical_start.get(key,0) for key,value in numerical_diagnostics().items()}
        metadata['total_seconds']=time.perf_counter()-start
        if torch.cuda.is_available(): metadata['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        write_run_manifest(destination/'run_manifest.json',metadata)
    return metadata
