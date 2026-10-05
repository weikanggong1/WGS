"""Private JSON configuration for chromosome-sharded association analyses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
from .gds import SeqArrayGDS
from .sparse_numerics import sparse_execution_metadata
from .numerics import reference_dot_execution_metadata
from ._precision_eigen import precision_eigen_execution_metadata
from ._reference_weights import reference_weights_execution_metadata
from .pipeline import PheWASPipeline, AnalysisOptions
from .io import fit_prepared_input, save_null_model, load_null_model
from .r_output import write_association_output, write_association_batch
from .compat import write_gaussian_null
from .compat_joint import write_joint_gaussian_null


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _read_promoter_intervals(filename):
    intervals = []
    with open(filename) as stream:
        for line in stream:
            if line.strip() and not line.startswith('#'):
                chromosome, start, end = line.split()[:3]
                if start == 'start' and end == 'end':
                    continue
                intervals.append((chromosome, int(start), int(end)))
    return intervals


def _bind_gds_samples(gds,model,prepared_indices=None,rule="auto"):
    """Bind once to actual GDS IDs; remap each subsequent chromosome by ID."""
    if rule not in ("auto","exact","last_underscore_token"):
        raise ValueError("sample_id_rule must be auto, exact, or last_underscore_token")
    canonical=getattr(model,"gds_sample_ids",None)
    if canonical is not None:
        canonical=np.asarray(canonical,dtype=str)
        if canonical.shape!=(model.n,) or len(set(canonical))!=model.n:
            raise ValueError("cached GDS identifiers must be unique and aligned")
        indices=gds.sample_indices(canonical)
        if not np.array_equal(gds.sample_ids()[indices],canonical):
            raise ValueError("chromosome GDS sample alignment is inconsistent")
        return indices
    identifiers=np.asarray(model.sample_ids,dtype=str)
    available=gds.sample_ids()
    if prepared_indices is None:
        if rule in ("exact","auto"):
            try:indices=gds.sample_indices(identifiers)
            except ValueError:
                if rule=="exact":raise
                indices=None
        else:indices=None
        if indices is None:
            normalized=np.asarray([item.rsplit("_",1)[-1] for item in available],dtype=str)
            if len(set(normalized))!=len(normalized):
                raise ValueError("GDS sample normalization is ambiguous")
            lookup={identifier:j for j,identifier in enumerate(normalized)}
            try:indices=np.asarray([lookup[item] for item in identifiers],dtype=np.int64)
            except KeyError:raise ValueError("a model identifier is absent after GDS normalization") from None
    else:
        raw=np.asarray(prepared_indices)
        if raw.shape!=(model.n,) or not np.issubdtype(raw.dtype,np.integer):
            raise ValueError("prepared sample indices must be an aligned integer vector")
        indices=raw.astype(np.int64)
        if len(set(indices))!=model.n or np.any(indices<0) or np.any(indices>=len(available)):
            raise ValueError("prepared GDS indices are nonunique or out of range")
    selected=available[indices]
    exact=np.array_equal(selected,identifiers)
    normalized=np.array_equal(np.asarray([item.rsplit("_",1)[-1] for item in selected]),identifiers)
    if not (exact if rule=="exact" else normalized if rule=="last_underscore_token" else exact or normalized):
        raise ValueError("prepared sample indices do not match model identifiers")
    model.gds_sample_ids=selected.copy()
    return indices


def run_configuration(config, *, device="cuda"):
    """Fit/load one model per phenotype, then run ordered chromosome jobs."""
    precision_eigen_execution_metadata(reset=True)
    reference_weights_execution_metadata(reset=True)
    full_started = time.perf_counter()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.startswith('cuda'):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    cuda_initialization_seconds = time.perf_counter() - full_started if device.startswith('cuda') else 0.0
    models_started = time.perf_counter()
    models, rows = [], []
    for phenotype in config["phenotypes"]:
        if "model" in phenotype:
            model = load_null_model(phenotype["model"], device=device)
            index = None
            if "sample_indices_file" in phenotype:
                index = np.load(phenotype["sample_indices_file"], allow_pickle=False)
        else:
            fit_options=dict(phenotype.get("fit_options", {}))
            for key in ("joint_mode","family","binary_mode"):
                if key in phenotype:
                    if key in fit_options and fit_options[key]!=phenotype[key]:
                        raise ValueError("conflicting phenotype fitting modes")
                    fit_options[key]=phenotype[key]
            model, index = fit_prepared_input(phenotype["input"], device=device,
                transform=phenotype.get("transform", "none"), **fit_options)
        if model.n_pheno>1:
            requested=phenotype.get("joint_mode",phenotype.get("fit_options",{}).get("joint_mode"))
            expected={"ordinary_REML":"ordinary","AI_REML":"strict","factor_REML":"robust"}.get(model.fit_method)
            if requested is None or requested!=expected:
                raise ValueError("a joint cache requires its explicit matching joint_mode")
            names=phenotype.get("phenotype_names")
            if names is not None:
                if len(names)!=model.n_pheno:raise ValueError("phenotype_names must match joint columns")
                model.phenotype_names=tuple(names)
        if model.family=="binomial" and "reference" in model.fit_method.lower() and not config.get("validation_reference",False):
            raise ValueError("an original R binary reference cache is restricted to explicit validation_reference runs")
        if config.get('require_single_continuous', False) and (model.family != 'gaussian' or model.n_pheno != 1):
            raise ValueError('the complete chromosome workflow requires a single Gaussian phenotype')
        models.append(model); rows.append(index)
    null_fit_seconds = time.perf_counter() - models_started
    catalog = config.get("annotation_catalog", {})
    if isinstance(catalog, str):
        catalog = json.loads(Path(catalog).read_text())
    report = {"schema_version": 1, "phenotypes": [p["name"] for p in config["phenotypes"]], "jobs": []}
    # Report excludes the input configuration and any subject identifiers.
    output_groups={}
    remaining_outputs = Counter(str(Path(job['output']).resolve())
        for chromosome in config['chromosomes'] for job in chromosome['jobs'])
    interval_cache = {}
    statistics_execution = config.get('statistics_execution', 'serial')
    if statistics_execution not in ('serial', 'batched'):
        raise ValueError('statistics_execution must be serial or batched')
    index_seconds = 0.0
    setup_seconds = 0.0
    native_output_seconds = 0.0
    null_output_seconds = 0.0
    for chromosome in config["chromosomes"]:
        before_setup = time.perf_counter()
        with SeqArrayGDS(chromosome["gds"]) as gds:
            current_rows=[_bind_gds_samples(gds,model,index,phenotype.get("sample_id_rule","auto"))
                          for model,index,phenotype in zip(models,rows,config["phenotypes"])]
            pipeline = PheWASPipeline(gds, models, qc_path=config.get("qc_path", "annotation/filter"),
                annotation_catalog=catalog, annotation_names=config.get("annotation_names", []),
                gds_sample_indices=current_rows,
                options=AnalysisOptions(**config.get("analysis_options", {})))
            pipeline.statistics_execution = statistics_execution
            setup_seconds += time.perf_counter() - before_setup
            if 'annotation_index' in chromosome:
                index_config = chromosome['annotation_index']
                promoter_file = index_config.get('promoter_intervals_file')
                if promoter_file is not None and promoter_file not in interval_cache:
                    interval_cache[promoter_file] = _read_promoter_intervals(promoter_file)
                before_index = time.perf_counter()
                pipeline.prepare_annotation_index(chromosome['name'],
                    promoter_intervals=None if promoter_file is None else interval_cache[promoter_file],
                    include_ncrna=index_config.get('include_ncrna', True))
                index_seconds += time.perf_counter() - before_index
            if not report.get("models_saved",False):
                before_null_output = time.perf_counter()
                for model, phenotype in zip(models, config["phenotypes"]):
                    if "save_model" in phenotype:
                        Path(phenotype["save_model"]).parent.mkdir(parents=True, exist_ok=True)
                        save_null_model(model,phenotype["save_model"])
                    if "output_null" in phenotype:
                        if model.family!="gaussian":
                            raise NotImplementedError("binary native null R serialization is not implemented; use the complete NPZ state cache")
                        writer=write_joint_gaussian_null if model.n_pheno>1 else write_gaussian_null
                        extra={"phenotype_names":phenotype.get("phenotype_names"),"layout":phenotype.get("null_layout","phewas")} if model.n_pheno>1 else {}
                        writer(phenotype["output_null"],model,original_sample_ids=model.gds_sample_ids,
                               covariate_names=phenotype.get("covariate_names"),**extra)
                report["models_saved"]=True
                null_output_seconds += time.perf_counter() - before_null_output
            for job in chromosome["jobs"]:
                kind = job["kind"]
                arguments = dict(job.get("arguments", {}))
                arguments["chromosome"] = chromosome["name"]
                if "promoter_intervals_file" in arguments:
                    filename = arguments.pop("promoter_intervals_file")
                    if filename not in interval_cache:
                        interval_cache[filename] = _read_promoter_intervals(filename)
                    arguments["promoter_intervals"] = interval_cache[filename]
                print(json.dumps({"event": "started", "job": job.get("name", kind), "chromosome": chromosome["name"]}), flush=True)
                started = time.perf_counter()
                result = getattr(pipeline, kind)(**arguments)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                output = Path(job["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.suffix.lower() in (".rdata", ".rda", ".rds"):
                    key=str(output.resolve())
                    signature=(kind,job.get("object_name"),job.get("layout","phewas"))
                    if key in output_groups and output_groups[key]["signature"]!=signature:
                        raise ValueError("one association output cannot combine different kinds, names, or layouts")
                    group=output_groups.setdefault(key,{"path":output,"signature":signature,"results":[]})
                    if kind=="individual" and group["results"]:
                        raise ValueError("multiple individual jobs require separate native output files")
                    group["results"].append(result)
                    remaining_outputs[key] -= 1
                    if remaining_outputs[key] == 0:
                        # Final occurrence is known from the complete schedule.
                        # Flush whole tutorial batches as soon as ready, so a
                        # chromosome does not retain all individual data.frames.
                        before_write = time.perf_counter()
                        write_association_batch(group['path'], group['results'], kind=kind,
                            object_name=job.get('object_name'), layout=job.get('layout', 'phewas'))
                        native_output_seconds += time.perf_counter() - before_write
                        group['results'].clear()
                elif output.suffix.lower() == ".json" and config.get("debug_json", False):
                    output.write_text(json.dumps(result, ensure_ascii=False, default=_json_value, allow_nan=False))
                else:
                    raise ValueError("formal outputs must be native .Rdata/.rds; JSON requires explicit debug_json")
                if "debug_output" in job:
                    if not config.get("debug_json",False):
                        raise ValueError("job.debug_output requires explicit debug_json=true")
                    debug_path=Path(job["debug_output"])
                    if debug_path.suffix.lower()!=".json":
                        raise ValueError("debug_output must have .json suffix")
                    debug_path.parent.mkdir(parents=True,exist_ok=True)
                    debug_path.write_text(json.dumps(result,ensure_ascii=False,default=_json_value,allow_nan=False))
                report["jobs"].append({"name": job.get("name", kind), "chromosome": chromosome["name"],
                                       "kind": kind, "seconds": elapsed})
                print(json.dumps({"event": "finished", "job": job.get("name", kind),
                                  "seconds": elapsed}), flush=True)
                del result
            if getattr(pipeline, 'batch_diagnostics', None):
                report.setdefault('batch_diagnostics', []).extend(pipeline.batch_diagnostics)
    for group in output_groups.values():
        kind,object_name,layout=group["signature"]
        if group['results']:
            before_write = time.perf_counter()
            write_association_batch(group["path"],group["results"],kind=kind,object_name=object_name,layout=layout)
            native_output_seconds += time.perf_counter() - before_write
    report["null_fit_seconds"] = null_fit_seconds
    report["total_seconds"] = time.perf_counter() - full_started
    report["peak_gpu_mib"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None
    report["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20 if device.startswith("cuda") else None
    report.pop("models_saved",None)
    report["number_samples"] = [model.n for model in models]
    report["number_phenotypes"] = [model.n_pheno for model in models]
    report["null_fit_methods"] = [getattr(model,"fit_method","single_gaussian") for model in models]
    report["reference_validation"] = bool(config.get("validation_reference",False))
    report["device"] = device
    report['statistics_execution'] = statistics_execution
    report['index_preparation_seconds'] = index_seconds
    report['gds_setup_seconds'] = setup_seconds
    report['genotype_reader'] = gds.reader_metadata
    report['ordered_addition_execution'] = sparse_execution_metadata()
    report['reference_projection_execution'] = reference_dot_execution_metadata()
    report['precision_eigen_execution'] = precision_eigen_execution_metadata()
    report['annotation_weight_execution'] = reference_weights_execution_metadata()
    report['native_association_output_seconds'] = native_output_seconds
    report['native_null_and_cache_output_seconds'] = null_output_seconds
    report['torch_version'] = torch.__version__
    report['cuda_version'] = torch.version.cuda
    report['cuda_initialization_seconds'] = cuda_initialization_seconds
    report['gpu_name'] = torch.cuda.get_device_name(device) if device.startswith('cuda') else None
    return report


def main():
    parser = argparse.ArgumentParser(description="GPU STAAR PheWAS on native GDS")
    parser.add_argument("config", type=Path, help="private JSON analysis configuration")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--report", type=Path, help="aggregate execution summary")
    args = parser.parse_args()
    report = run_configuration(json.loads(args.config.read_text()), device=args.device)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
