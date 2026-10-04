"""Private JSON configuration for chromosome-sharded association analyses."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .gds import SeqArrayGDS
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
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    full_started = time.perf_counter()
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
        models.append(model); rows.append(index)
    null_fit_seconds = time.perf_counter() - full_started
    catalog = config.get("annotation_catalog", {})
    if isinstance(catalog, str):
        catalog = json.loads(Path(catalog).read_text())
    report = {"schema_version": 1, "phenotypes": [p["name"] for p in config["phenotypes"]], "jobs": []}
    # Report excludes the input configuration and any subject identifiers.
    output_groups={}
    for chromosome in config["chromosomes"]:
        with SeqArrayGDS(chromosome["gds"]) as gds:
            current_rows=[_bind_gds_samples(gds,model,index,phenotype.get("sample_id_rule","auto"))
                          for model,index,phenotype in zip(models,rows,config["phenotypes"])]
            pipeline = PheWASPipeline(gds, models, qc_path=config.get("qc_path", "annotation/filter"),
                annotation_catalog=catalog, annotation_names=config.get("annotation_names", []),
                gds_sample_indices=current_rows,
                options=AnalysisOptions(**config.get("analysis_options", {})))
            if not report.get("models_saved",False):
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
            for job in chromosome["jobs"]:
                kind = job["kind"]
                arguments = dict(job.get("arguments", {}))
                arguments["chromosome"] = chromosome["name"]
                if "promoter_intervals_file" in arguments:
                    filename = arguments.pop("promoter_intervals_file")
                    intervals = []
                    with open(filename) as stream:
                        for line in stream:
                            if line.strip() and not line.startswith("#"):
                                c, start, end = line.split()[:3]
                                if start == "start" and end == "end":
                                    continue
                                intervals.append((c, int(start), int(end)))
                    arguments["promoter_intervals"] = intervals
                print(json.dumps({"event": "started", "job": job.get("name", kind), "chromosome": chromosome["name"]}), flush=True)
                started = time.perf_counter()
                result = getattr(pipeline, kind)(**arguments)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
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
    for group in output_groups.values():
        kind,object_name,layout=group["signature"]
        write_association_batch(group["path"],group["results"],kind=kind,object_name=object_name,layout=layout)
    report["null_fit_seconds"] = null_fit_seconds
    report["total_seconds"] = time.perf_counter() - full_started
    report["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
    report.pop("models_saved",None)
    report["number_samples"] = [model.n for model in models]
    report["number_phenotypes"] = [model.n_pheno for model in models]
    report["null_fit_methods"] = [getattr(model,"fit_method","single_gaussian") for model in models]
    report["reference_validation"] = bool(config.get("validation_reference",False))
    report["device"] = device
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
