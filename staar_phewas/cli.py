"""Private JSON configuration for chromosome-sharded association analyses."""
from dataclasses import replace
import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import time
import numpy as np
import torch
from .gds import SeqArrayGDS
from .tf32 import validate_mode, configure_tf32, execution_metadata as tf32_execution_metadata
from .profiling import StageProfiler
from .precision_audit import DenseProductAudit
from .sparse_numerics import sparse_execution_metadata
from .numerics import reference_dot_execution_metadata
from ._precision_eigen import precision_eigen_execution_metadata
from ._reference_weights import reference_weights_execution_metadata
from .pipeline import PheWASPipeline, AnalysisOptions
from .masks import NONCODING_CATEGORIES
from .io import fit_prepared_input, save_null_model, load_null_model
from .null_model import GaussianNullModel
from .r_output import write_association_output, write_association_batch
from .compat import write_gaussian_null
from .compat_joint import write_joint_gaussian_null


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _association_result_row_count(result, *, kind):
    """Count computed trait rows, independently of their P values.

    Gene categories contain trait row lists; direct gene/Single outputs contain
    trait row lists without a category layer. Empty lists serialize as R NULL.
    Unexpected shapes fail closed instead of being classified as empty.
    """
    if kind not in {"coding", "noncoding", "ncrna", "singlevariant", "individual"}:
        raise ValueError("unsupported association result kind")
    if isinstance(result, Mapping):
        if kind not in {"coding", "noncoding"}:
            raise ValueError("unexpected association result category layer")
        groups = result.values()
    else:
        groups = [result]
    count = 0
    for traits in groups:
        if not isinstance(traits, list) or not traits:
            raise ValueError("association results require a nonempty trait list")
        for rows in traits:
            if not isinstance(rows, list) or any(not isinstance(row, Mapping) or not row for row in rows):
                raise ValueError("association results require trait lists of nonempty records")
            count += len(rows)
    return count


def _native_execution_status(metadata, jobs, *, planned_jobs):
    """Allow zero products only for a completed, explicitly empty schedule."""
    if metadata["ptx_verified_tf32_gemm_count"] != metadata["tf32_gemm_call_count"] or metadata["fp64_gemm_fallback_count"] != 0:
        raise RuntimeError("Forced TF32 verification failed")
    if metadata["logical_product_count"] > 0:
        return "executed_products"
    complete_empty = metadata["logical_product_count"] == 0 and planned_jobs > 0 and len(jobs) == planned_jobs and all(
        type(job.get("eligible_association_tests")) is int and job["eligible_association_tests"] == 0
        for job in jobs)
    if not complete_empty:
        raise RuntimeError("Native run performed no matrix/vector products")
    return "no_eligible_analysis_products"


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


def _scheduled_index_categories(jobs):
    """Collect only categories used by coordinate-free gene jobs, in job order."""
    categories = []
    for job in jobs:
        arguments = job.get("arguments", {})
        if arguments.get("start") is not None or arguments.get("end") is not None:
            continue
        if job["kind"] == "ncrna":
            requested = ["ncRNA"]
        elif job["kind"] == "noncoding":
            category = arguments.get("category", "all_categories")
            requested = list(NONCODING_CATEGORIES) if category == "all_categories" else [category]
            if arguments.get("include_ncrna", False):
                requested.append("ncRNA")
        else:
            continue
        for category in requested:
            if category not in categories:
                categories.append(category)
    return categories


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
    """Run a forced TF32 analysis, or an explicit FP64 reference control."""
    mode = validate_mode(config.get("matmul_mode", "tf32"))
    # Every run resets the native backend. Reconstruction controls were
    # removed rather than silently mapped to a different arithmetic mode.
    obsolete = {"tf32_binned_tile_shape", "tf32_binned_fused_small"} & config.keys()
    if obsolete:
        raise ValueError("Removed TF32 reconstruction parameters: " + ", ".join(sorted(obsolete)))
    requested_split_k = config.get("tf32_split_k", 0)
    tf32_configuration = configure_tf32(split_k=requested_split_k,
        memory_limit_gib=config.get("analysis_options", {}).get("memory_limit_gib", 20))
    tf32_configuration.update(requested_split_k=requested_split_k,
        effective_split_k=0 if mode == "tf32" else None,
        split_k_applies=False)
    if mode == "fp64":
        report = _run_configuration(config, device=device)
        report["dense_product_audit"] = {"enabled": False, "scope": "reference control"}
        report["tf32_configuration"] = tf32_configuration
        return report
    with DenseProductAudit(forced=True) as audit:
        report = _run_configuration(config, device=device)
    report["dense_product_audit"] = audit.report()
    report["tf32_configuration"] = tf32_configuration
    return report


def _run_configuration(config, *, device="cuda"):
    """Fit/load one model per phenotype, then run ordered chromosome jobs."""
    matmul_mode = validate_mode(config.get("matmul_mode", "tf32"))
    tail_optimization = config.get("statistics_tail_optimization", True)
    if not isinstance(tail_optimization, bool):
        raise ValueError("statistics_tail_optimization must be a JSON boolean")
    precision_control = config.get("precision_control", False)
    if not isinstance(precision_control, bool):
        raise ValueError("precision_control must be a JSON boolean")
    if matmul_mode == "fp64" and not precision_control:
        raise ValueError("FP64 controls require explicit precision_control=true; production uses native tf32")
    if matmul_mode != "fp64" and not str(device).startswith("cuda"):
        raise ValueError("Forced TF32 requires CUDA; CPU controls require matmul_mode=fp64 and precision_control=true")
    if config.get("statistics_execution", "serial") != "serial":
        raise ValueError("TF32 validation runs require serial statistics execution")
    supported_kinds = {"coding", "noncoding", "ncrna", "singlevariant", "individual"}
    packed_directory = config.get("packed_reader_directory")
    if packed_directory is not None and (not isinstance(packed_directory, str) or not packed_directory.strip()):
        raise ValueError("packed_reader_directory must be a nonempty local build directory string")
    for chromosome in config.get("chromosomes", []):
        for job in chromosome.get("jobs", []):
            if job.get("kind") not in supported_kinds:
                raise ValueError("Pipeline jobs must be coding, noncoding, ncrna, singlevariant, or individual")
    for optimization in ("local_mask_reuse", "weight_batch_optimization"):
        if not isinstance(config.get(optimization, True), bool):
            raise ValueError(f"{optimization} must be a JSON boolean")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tf32_execution_metadata(reset=True)
    from .statistics import statistics_execution_metadata
    statistics_execution_metadata(reset=True)
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
    source_modes, model_sources = [], []
    for phenotype in config["phenotypes"]:
        if "model" in phenotype:
            model = load_null_model(phenotype["model"], device=device, matmul_mode=matmul_mode)
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
            run_split_k = config.get("tf32_split_k", 0)
            if fit_options.pop("tf32_split_k", run_split_k) != run_split_k:
                raise ValueError("phenotype fit_options.tf32_split_k conflicts with run tf32_split_k")
            if fit_options.get("matmul_mode", matmul_mode) != matmul_mode:
                raise ValueError("phenotype fit_options.matmul_mode conflicts with run matmul_mode")
            if fit_options.get("family", "gaussian") == "gaussian" and fit_options.get("joint_mode") is None:
                fit_options["matmul_mode"] = matmul_mode
            model, index = fit_prepared_input(phenotype["input"], device=device,
                transform=phenotype.get("transform", "none"), **fit_options)
        if matmul_mode != "fp64" and (not isinstance(model, GaussianNullModel) or model.n_pheno != 1):
            raise ValueError("Forced TF32 pipeline currently supports single Gaussian models")
        source_modes.append(getattr(model, "source_matmul_mode", getattr(model, "matmul_mode", "fp64")))
        model_sources.append("loaded_cache" if "model" in phenotype else "fitted_input")
        # A legacy cache defaults to FP64; explicit run mode always overrides
        # the loaded state before association and before save_model writes.
        if isinstance(model, GaussianNullModel):
            model.set_matmul_mode(matmul_mode)
        else:
            model.matmul_mode = matmul_mode
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
    stage_reports = []
    reader_reports = []
    null_output_seconds = 0.0
    for chromosome in config["chromosomes"]:
        before_setup = time.perf_counter()
        reader_options = {} if packed_directory is None else {"packed_reader_directory": packed_directory}
        with SeqArrayGDS(chromosome["gds"], **reader_options) as gds:
            current_rows=[_bind_gds_samples(gds,model,index,phenotype.get("sample_id_rule","auto"))
                          for model,index,phenotype in zip(models,rows,config["phenotypes"])]
            pipeline = PheWASPipeline(gds, models, qc_path=config.get("qc_path", "annotation/filter"),
                annotation_catalog=catalog, annotation_names=config.get("annotation_names", []),
                gds_sample_indices=current_rows,
                options=AnalysisOptions(**config.get("analysis_options", {})))
            pipeline.statistics_execution = statistics_execution
            pipeline.statistics_tail_optimization = tail_optimization
            pipeline.local_mask_reuse = config.get("local_mask_reuse", True)
            pipeline.weight_batch_optimization = config.get("weight_batch_optimization", True)
            pipeline.resident_genotypes = bool(config.get("resident_genotypes", device.startswith("cuda")))
            pipeline.profiler = StageProfiler(device, enabled=config.get("stage_profile", False))
            gds._stage_profiler = pipeline.profiler if pipeline.profiler.enabled else None
            setup_seconds += time.perf_counter() - before_setup
            index_categories = _scheduled_index_categories(chromosome['jobs'])
            if 'annotation_index' in chromosome and index_categories:
                index_config = chromosome['annotation_index']
                promoter_file = (index_config.get('promoter_intervals_file')
                    if any(category.startswith('promoter_') for category in index_categories) else None)
                if promoter_file is not None and promoter_file not in interval_cache:
                    interval_cache[promoter_file] = _read_promoter_intervals(promoter_file)
                before_index = time.perf_counter()
                pipeline.prepare_annotation_index(chromosome['name'],
                    promoter_intervals=None if promoter_file is None else interval_cache[promoter_file],
                    include_ncrna=False, categories=index_categories)
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
                individual_block = config.get("individual_genotype_block_size", 8192 if matmul_mode == "tf32" else None)
                if kind == "individual" and individual_block is not None:
                    if type(individual_block) is not int or individual_block < 1:
                        raise ValueError("individual_genotype_block_size must be a positive integer")
                    old_options = pipeline.options
                    pipeline.options = replace(old_options, genotype_block_size=individual_block)
                    try:
                        result = getattr(pipeline, kind)(**arguments)
                    finally:
                        pipeline.options = old_options
                else:
                    result = getattr(pipeline, kind)(**arguments)
                eligible_tests = _association_result_row_count(result, kind=kind)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                pipeline.profiler.flush()
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
                                       "kind": kind, "seconds": elapsed,
                                       "eligible_association_tests": eligible_tests})
                print(json.dumps({"event": "finished", "job": job.get("name", kind),
                                  "seconds": elapsed}), flush=True)
                del result
            report["statistics_tail_optimization_calls"] = report.get("statistics_tail_optimization_calls", 0) + getattr(pipeline, "statistics_tail_optimization_calls", 0)
            report["memory_guard_max_estimated_bytes"] = max(report.get("memory_guard_max_estimated_bytes", 0),
                getattr(pipeline, "memory_guard_max_estimated_bytes", 0))
            stage_reports.append(pipeline.profiler.report())
            reader_reports.append(gds.reader_metadata)
            reuse_report = report.setdefault("local_mask_reuse_execution", {})
            for key, value in getattr(pipeline, "local_mask_reuse_counters", {}).items():
                reuse_report[key] = reuse_report.get(key, 0) + int(value)
            if getattr(pipeline, 'batch_diagnostics', None):
                report.setdefault('batch_diagnostics', []).extend(pipeline.batch_diagnostics)
    for group in output_groups.values():
        kind,object_name,layout=group["signature"]
        if group['results']:
            before_write = time.perf_counter()
            write_association_batch(group["path"],group["results"],kind=kind,object_name=object_name,layout=layout)
            native_output_seconds += time.perf_counter() - before_write
    tf32_metadata = tf32_execution_metadata()
    report["eligible_association_tests"] = sum(job["eligible_association_tests"] for job in report["jobs"])
    report["association_test_count_scope"] = "computed gene-mask-trait or Single variant-trait output rows; independent of P values"
    if matmul_mode != "fp64":
        report["native_execution_status"] = _native_execution_status(tf32_metadata, report["jobs"],
            planned_jobs=sum(len(chromosome["jobs"]) for chromosome in config["chromosomes"]))
    else:
        report["native_execution_status"] = "reference_control"
    core_dtype = "float32" if matmul_mode == "tf32" else "float64"
    report["precision_boundary"] = {"dense_products": matmul_mode,
        "null_genotype_score_covariance": core_dtype,
        "eigh_cholesky_solve": core_dtype + " library precision",
        "matrix_accumulation": "float32" if matmul_mode == "tf32" else "float64",
        "vector_products": "CUDA float32 mv/dot (not TF32 MMA)" if matmul_mode == "tf32" else "float64",
        "native_R_real_storage": "double serialized from computed values",
        "reconstruction_components": 0}
    report["matmul_mode"] = matmul_mode
    report["precision_control"] = bool(config.get("precision_control", False))
    report["model_matmul_modes"] = [model.matmul_mode for model in models]
    report["source_model_matmul_modes"] = source_modes
    report["null_model_sources"] = model_sources
    report["null_cache_written"] = ["save_model" in phenotype for phenotype in config["phenotypes"]]
    report["tf32_execution"] = tf32_metadata
    report["stage_profile"] = stage_reports
    from .statistics import statistics_execution_metadata
    report["statistics_execution_metadata"] = statistics_execution_metadata()
    if matmul_mode == "tf32":
        from . import _burden, _weighted_spectra, _fused_saddle
        report["burden_execution"] = _burden.execution_metadata()
        report["weighted_spectrum_execution"] = _weighted_spectra.execution_metadata()
        report["probability_execution"] = _fused_saddle.execution_metadata()
    report["resident_genotypes"] = bool(config.get("resident_genotypes", device.startswith("cuda")))
    report["null_fit_seconds"] = null_fit_seconds
    report["total_seconds"] = time.perf_counter() - full_started
    report["peak_gpu_mib"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None
    report["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20 if device.startswith("cuda") else None
    memory_limit_gib = float(config.get("analysis_options", {}).get("memory_limit_gib", 20.0))
    report["memory_limit_gib"] = memory_limit_gib
    report["peak_gpu_budget_pass"] = report["peak_gpu_mib"] is None or report["peak_gpu_mib"] <= memory_limit_gib * 1024
    if not report["peak_gpu_budget_pass"]:
        raise MemoryError(f"Actual peak CUDA allocation {report['peak_gpu_mib']:.3f} MiB exceeded configured {memory_limit_gib:.3f} GiB budget; precision and chunks were not changed")
    report.pop("models_saved",None)
    report["number_samples"] = [model.n for model in models]
    report["number_phenotypes"] = [model.n_pheno for model in models]
    report["null_fit_methods"] = [getattr(model,"fit_method","single_gaussian") for model in models]
    report["reference_validation"] = bool(config.get("validation_reference",False))
    report["device"] = device
    report['statistics_execution'] = statistics_execution
    report['statistics_tail_optimization'] = tail_optimization
    report['local_mask_reuse'] = config.get("local_mask_reuse", True)
    report['weight_batch_optimization'] = config.get("weight_batch_optimization", True)
    report['index_preparation_seconds'] = index_seconds
    report['gds_setup_seconds'] = setup_seconds
    report['genotype_reader'] = gds.reader_metadata
    report['genotype_readers'] = reader_reports
    report['gds_sdk_read_seconds'] = sum(float(route.get('seconds', 0.0))
        for reader in reader_reports for route in reader.get('native_reads', {}).values())
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
