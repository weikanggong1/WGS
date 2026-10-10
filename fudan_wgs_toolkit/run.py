"""Single and multiple phenotype association on prepared genetic data."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .identity import sample_keys, validate_sample_pairs, pairs_from_keys

_MISSING = {"", "na", "nan", "null", "none", "n/a"}
_KINDS = ("individual", "coding", "noncoding", "ncrna")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class NumericCSV:
    sample_pairs: np.ndarray
    names: tuple[str, ...]
    values: np.ndarray

    @property
    def sample_ids(self):
        return sample_keys(self.sample_pairs)


def _read_csv(path, *, require_columns):
    path = Path(path)
    pairs, rows = [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path.name}: missing CSV header")
        header = [name.strip() for name in header]
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError(f"{path.name}: column names must be nonempty and unique")
        if not {"FID", "IID"} <= set(header):
            raise ValueError(f"{path.name}: FID and IID columns are required")
        fid, iid = header.index("FID"), header.index("IID")
        columns = [i for i, name in enumerate(header) if name not in ("FID", "IID")]
        if require_columns and not columns:
            raise ValueError(f"{path.name}: provide at least one named phenotype column")
        for line, row in enumerate(reader, 2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name}:{line}: number of columns differs from the header")
            pairs.append([row[fid], row[iid]])
            values = []
            for column in columns:
                token = row[column].strip()
                if token.lower() in _MISSING:
                    value = float("nan")
                else:
                    try:
                        value = float(token)
                    except ValueError as error:
                        raise ValueError(f"{path.name}:{line}: {header[column]} must contain numeric or missing values") from error
                    if not np.isfinite(value):
                        raise ValueError(f"{path.name}:{line}: infinity is not a valid numeric value")
                values.append(value)
            rows.append(values)
    if not rows:
        raise ValueError(f"{path.name}: no participant rows")
    pairs = validate_sample_pairs(np.asarray(pairs, dtype=str), where=path.name)
    names = tuple(header[column] for column in columns)
    return NumericCSV(pairs, names, np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)))


def read_csv_inputs(phenotype_csv, covariate_csv):
    """Read numeric values using exact FID/IID strings without imputing values."""
    return _read_csv(phenotype_csv, require_columns=True), _read_csv(covariate_csv, require_columns=False)


def aligned_phenotype(phenotypes, covariates, cache_pairs, trait, *, covariate_names=None):
    """Select this phenotype's complete cases in prepared sample-axis order."""
    if trait not in phenotypes.names:
        raise ValueError("unknown phenotype column")
    if covariate_names is not None:
        if (not isinstance(covariate_names, (list, tuple)) or
                any(not isinstance(name, str) for name in covariate_names) or
                len(set(covariate_names)) != len(covariate_names)):
            raise ValueError("selected covariate names must be a sequence of distinct column names")
        if set(covariate_names) - set(covariates.names):
            raise ValueError("unknown selected covariate column")
        columns = [covariates.names.index(name) for name in covariate_names]
        covariates = NumericCSV(covariates.sample_pairs, tuple(covariate_names), covariates.values[:, columns])
    cached = validate_sample_pairs(cache_pairs, where="prepared sample index")
    keys = sample_keys(cached)
    pheno_lookup = {key: i for i, key in enumerate(phenotypes.sample_ids)}
    cov_lookup = {key: i for i, key in enumerate(covariates.sample_ids)}
    matched = [(row, pheno_lookup[key], cov_lookup[key]) for row, key in enumerate(keys)
               if key in pheno_lookup and key in cov_lookup]
    cache_rows = np.asarray([item[0] for item in matched], dtype=np.int64)
    pheno_rows = np.asarray([item[1] for item in matched], dtype=np.int64)
    cov_rows = np.asarray([item[2] for item in matched], dtype=np.int64)
    y = phenotypes.values[pheno_rows, phenotypes.names.index(trait)]
    x = covariates.values[cov_rows]
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    identifiers, y, x = keys[cache_rows][keep], y[keep], x[keep]
    has_intercept = bool(x.shape[1] and np.any(np.all(x == 1, axis=0)))
    names = covariates.names
    if not has_intercept:
        x = np.column_stack((np.ones(len(y)), x))
        names = ("Intercept", *names)
    if len(y) <= x.shape[1]:
        raise ValueError(f"phenotype {trait}: insufficient complete-case samples for the covariate design")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError(f"phenotype {trait}: covariate design is rank deficient")
    cache_set = set(keys)
    counts = dict(phenotype_rows=len(phenotypes.values), covariate_rows=len(covariates.values),
        prepared_samples=len(cached), matched_rows_before_missing=len(matched),
        excluded_missing_rows=int((~keep).sum()), analysis_samples=len(y),
        phenotype_rows_absent_from_prepared=sum(key not in cache_set for key in pheno_lookup),
        phenotype_rows_absent_from_covariates=sum(key not in cov_lookup for key in pheno_lookup),
        intercept_added=not has_intercept, design_columns=x.shape[1])
    return identifiers, y, x, names, counts


def _cache_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("prepared dataset paths must be nonempty relative paths")
    path = Path(os.path.abspath(root / value))
    if not path.is_relative_to(root):
        raise ValueError("prepared dataset paths must remain within the prepared directory")
    return path


def _dataset(prepared_directory, chromosomes=None):
    root = Path(prepared_directory).resolve()
    manifest_path = root / "dataset.json"
    raw = manifest_path.read_bytes()
    try:
        complete = (root / "COMPLETE").read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError("prepared dataset lacks its COMPLETE integrity proof") from error
    if complete != hashlib.sha256(raw).hexdigest():
        raise ValueError("prepared dataset COMPLETE hash differs from dataset.json")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported dataset.json schema_version; prepare the genetic data first")
    _analysis_settings(manifest)
    raw_entries = manifest.get("chromosomes")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("dataset.json must declare chromosome entries")
    if isinstance(chromosomes, (str, int)):
        chromosomes = [chromosomes]
    wanted = None if chromosomes is None else {str(value).removeprefix("chr") for value in chromosomes}
    entries, seen = [], set()
    for raw in raw_entries:
        entry = dict(raw)
        name = str(entry.get("name", "")).removeprefix("chr")
        if not name.isdecimal() or str(int(name)) != name or not 1 <= int(name) <= 22 or name in seen:
            raise ValueError("chromosome entries must have distinct names from 1 to 22")
        seen.add(name)
        if wanted is not None and name not in wanted:
            continue
        entry["name"] = name
        for key in ("container_directory", "metadata_directory", "gene_catalog", "promoter_intervals"):
            if key in entry:
                entry[key] = str(_cache_path(root, entry[key]))
        if not {"container_directory", "metadata_directory"} <= entry.keys():
            raise ValueError("each chromosome requires genotype and metadata directories")
        entries.append(entry)
    if wanted is not None and wanted - seen:
        raise ValueError("requested chromosomes are absent from the prepared dataset")
    if not entries:
        raise ValueError("no chromosomes selected")
    entries.sort(key=lambda entry: int(entry["name"]))
    pairs = validate_sample_pairs(np.load(_cache_path(root, manifest.get("sample_pairs", "sample_pairs.npy")),
                                        allow_pickle=False), where="prepared sample index")
    if "sample_ids" in manifest:
        stored = np.load(_cache_path(root, manifest["sample_ids"]), allow_pickle=False)
        if stored.ndim != 1 or not np.array_equal(stored, sample_keys(pairs)):
            raise ValueError("prepared sample keys disagree with the FID/IID sample axis")
    return root, manifest_path, manifest, entries, pairs


def _analysis_settings(manifest):
    """Require the prepared root's explicit scientific annotation settings."""
    required = ("annotation_catalog", "annotation_names", "qc_path")
    if any(name not in manifest for name in required):
        raise ValueError("prepared dataset requires annotation_catalog, annotation_names and qc_path")
    catalog, names, qc_path = (manifest[name] for name in required)
    if (not isinstance(catalog, dict) or
            any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in catalog.items())):
        raise ValueError("prepared annotation_catalog must map nonempty names to nonempty field paths")
    if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or set(names) - set(catalog)):
        raise ValueError("prepared annotation_names must be distinct catalog names in weight order")
    if not isinstance(qc_path, str) or not qc_path:
        raise ValueError("prepared qc_path must be a nonempty field path")
    return dict(annotation_catalog=dict(catalog), annotation_names=list(names), qc_path=qc_path)


def _safe_name(name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or name in ("models", "plan.private.json", "report.private.json")
            or any(character in name for character in ("/", "\\", "\0"))
            or any(character.isspace() and character not in (" ",) for character in name)
            or name != name.strip() or Path(name).name != name):
        raise ValueError("phenotype names must be safe single-directory names")
    return name


def _proof(directory, expected_sha256):
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256 or (directory / "COMPLETE").read_text().strip() != digest:
        raise ValueError("immutable prepared genotype manifest changed")
    return json.loads(raw)["binding"]


def _annotation_coverage(entry, *, gene_variant_type, analyses, expected_sample_keys=None,
                         expected_analysis=None):
    """Validate annotation availability on the selected, bound variant axis."""
    from .cache_runtime.portable import PortableMetadataReader
    with PortableMetadataReader(entry["metadata_directory"], entry["container_directory"]) as reader:
        if expected_analysis is not None:
            analysis = reader.manifest.get("analysis")
            if not isinstance(analysis, dict) or any(name not in analysis for name in expected_analysis):
                raise ValueError("chromosome metadata requires explicit annotation_catalog, annotation_names and qc_path")
            if any(analysis[name] != value for name, value in expected_analysis.items()):
                raise ValueError("prepared dataset annotation mapping, weight order or QC differs from chromosome metadata")
        if expected_sample_keys is not None and not np.array_equal(reader.sample_ids(), expected_sample_keys):
            raise ValueError("chromosome sample keys differ from the prepared dataset FID/IID axis")
        coverage = reader.manifest.get("annotation_coverage")
        if not isinstance(coverage, dict):
            raise ValueError("prepared metadata lacks selected-axis annotation coverage")
        for name in ("n_variants", "matched_variants", "missing_snv", "missing_non_snv"):
            if type(coverage.get(name)) is not int or coverage[name] < 0:
                raise ValueError("invalid selected-axis annotation coverage counts")
        if (coverage["n_variants"] != reader.n_variants or
                coverage["matched_variants"] + coverage["missing_snv"] + coverage["missing_non_snv"] != reader.n_variants):
            raise ValueError("annotation coverage does not match the selected variant axis")
        availability = np.asarray(reader.read_field(coverage.get("available_field", "annotation_available")))
        if availability.shape != (reader.n_variants,) or availability.dtype.kind != "b":
            raise ValueError("annotation availability must be a boolean selected-variant vector")
        if int(availability.sum()) != coverage["matched_variants"]:
            raise ValueError("annotation availability differs from the coverage proof")
        if coverage["missing_snv"]:
            raise ValueError("selected SNVs lack required functional annotations")
        if coverage["missing_non_snv"]:
            if coverage.get("allow_missing_non_snv") is not True:
                raise ValueError("missing non-SNV annotations require an explicit preparation policy")
            if gene_variant_type != "SNV" and any(kind != "individual" for kind in analyses):
                raise ValueError("Indel or all-variant gene analysis requires complete non-SNV annotations")
        if "individual" in analyses:
            entry["_maximum_position"] = int(np.asarray(reader.read_field("position"), dtype=np.int64).max(initial=0))
        return dict(coverage)


def _jobs(entry, analyses, output, *, single_mac_cutoff, single_group_variants, single_region_size):
    jobs, chromosome = [], entry["name"]
    if "individual" in analyses:
        maximum = entry.get("_maximum_position")
        if maximum is None:
            from .cache_runtime.portable import PortableMetadataReader
            with PortableMetadataReader(entry["metadata_directory"], entry["container_directory"]) as reader:
                maximum = int(np.asarray(reader.read_field("position"), dtype=np.int64).max(initial=0))
        for block, start in enumerate(range(1, maximum + 1, single_region_size), 1):
            jobs.append(dict(name=f"single_{block:04d}", kind="individual", layout="base",
                output=str(output / f"chr{chromosome}_single_{block:04d}.csv"),
                arguments=dict(start=start, end=min(start + single_region_size - 1, maximum),
                    mac_cutoff=single_mac_cutoff, subset_variants_num=single_group_variants, variant_type="variant")))
    gene_kinds = set(analyses) - {"individual"}
    if gene_kinds:
        if "gene_catalog" not in entry:
            raise ValueError("gene analyses require an annotated gene_catalog for each chromosome")
        genes = json.loads(Path(entry["gene_catalog"]).read_text(encoding="utf-8"))
        if isinstance(genes, dict):
            genes = genes.get("jobs", genes.get("genes"))
        if not isinstance(genes, list):
            raise ValueError("gene_catalog must contain an ordered list of gene jobs")
        for ordinal, gene in enumerate(genes):
            kind = gene.get("kind")
            if kind not in gene_kinds:
                continue
            arguments = dict(gene.get("arguments", {}))
            arguments.pop("chromosome", None)
            arguments.pop("promoter_intervals_file", None)
            for key in ("gene_name", "start", "end", "category", "include_ptv", "include_ncrna"):
                if key in gene:
                    arguments[key] = gene[key]
            if not isinstance(arguments.get("gene_name"), str) or not arguments["gene_name"]:
                raise ValueError("gene jobs require a nonempty gene_name")
            if kind == "coding" and not {"start", "end"} <= arguments.keys():
                raise ValueError("coding jobs require inclusive start and end coordinates")
            if kind == "noncoding":
                if "promoter_intervals" not in entry:
                    raise ValueError("noncoding analyses require exact promoter intervals")
                arguments["promoter_intervals_file"] = entry["promoter_intervals"]
            jobs.append(dict(name=f"{kind}_{ordinal:06d}", kind=kind, layout="base", arguments=arguments,
                             output=str(output / f"chr{chromosome}_{kind}.csv")))
    if not jobs:
        raise ValueError("no association jobs found for the requested chromosome and analyses")
    return jobs


def run_WGS_all(phenotype_csv, covariate_csv, prepared_directory, *, output_directory,
                chromosomes=None, analyses=_KINDS, device="cuda:0", cpu_threads=8,
                memory_limit_gib=20.0, phenotype_families=None, phenotype_covariates=None,
                transform="none", null_fit_mode="fp64",
                gene_variant_type="SNV",
                covariance_block_size=4096, long_mask_threshold=5000, long_mask_rank=512, seed=1729,
                single_mac_cutoff=20, single_group_variants=5000, single_region_size=10_000_000,
                individual_effective_block_size=1024, device_cache_bytes=512*2**20,
                compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20, resume=False):
    """Run four association classes with independent complete cases per phenotype.

    CSVs contain exact string FID/IID pairs and numeric measurements. Ordinary
    Gaussian fitting is the default; ``phenotype_families`` may explicitly map
    columns to gaussian or binomial. No relationship structure is inferred.
    Associations use native TF32/FP32 on one CUDA device. Every mask is tested;
    masks above long_mask_threshold use the configured approximate spectrum.
    Final CSVs are saved under output_directory/<phenotype>/, with private model
    NPZs and JSON provenance. resume reuses a completed identical verified run;
    an interrupted or changed run requires a fresh output directory.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    if any(type(value) is not int or value<0 for value in (device_cache_bytes,compact_cache_bytes)):
        raise ValueError("device and compact cache capacities must be nonnegative integers")
    if type(metadata_cache_bytes) is not int or metadata_cache_bytes<1:
        raise ValueError("metadata_cache_bytes must be a positive integer")
    if isinstance(memory_limit_gib,bool) or not isinstance(memory_limit_gib,(int,float)) or not math.isfinite(memory_limit_gib) or not 0<memory_limit_gib<=20:
        raise ValueError("memory_limit_gib must be finite and in (0,20]")
    positive = (covariance_block_size,long_mask_threshold,long_mask_rank,single_mac_cutoff,
                single_group_variants,single_region_size,individual_effective_block_size)
    if any(type(value) is not int or value<1 for value in positive):
        raise ValueError("block sizes, mask limits and MAC cutoff must be positive integers")
    if type(seed) is not int or seed<0:
        raise ValueError("seed must be a nonnegative integer")
    if transform not in ("none","rint") or null_fit_mode not in ("fp64","tf32"):
        raise ValueError("transform must be none or rint; null_fit_mode must be fp64 or tf32")
    if gene_variant_type not in ("SNV", "Indel", "variant"):
        raise ValueError("gene_variant_type must be SNV, Indel or variant")
    kinds = tuple(dict.fromkeys(analyses))
    if not kinds or any(kind not in _KINDS for kind in kinds):
        raise ValueError("analyses must select individual, coding, noncoding or ncrna")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("run_WGS_all requires an available CUDA device")
    started = time.perf_counter()
    root,manifest_path,manifest,entries,pairs = _dataset(prepared_directory,chromosomes)
    analysis = _analysis_settings(manifest)
    phenotypes,covariates = read_csv_inputs(phenotype_csv,covariate_csv)
    for name in phenotypes.names:_safe_name(name)
    families = {} if phenotype_families is None else dict(phenotype_families)
    if set(families)-set(phenotypes.names) or any(value not in ("gaussian","binomial") for value in families.values()):
        raise ValueError("phenotype_families must map existing columns to gaussian or binomial")
    covariate_selection = {} if phenotype_covariates is None else dict(phenotype_covariates)
    if set(covariate_selection) - set(phenotypes.names):
        raise ValueError("phenotype_covariates must use existing phenotype column names")
    for columns in covariate_selection.values():
        if (not isinstance(columns, (list, tuple)) or any(not isinstance(name, str) for name in columns)
                or len(set(columns)) != len(columns) or set(columns) - set(covariates.names)):
            raise ValueError("phenotype_covariates must select distinct existing numeric covariate columns")
    covariate_selection = {name:list(columns) for name,columns in covariate_selection.items()}
    options = dict(memory_limit_gib=float(memory_limit_gib),genotype_block_size=128,
        annotation_block_size=250_000,variant_tile_size=512,covariance_backend="cached",
        cached_variant_tile_size=covariance_block_size,long_mask_threshold=long_mask_threshold,
        long_mask_method="fastskat",long_mask_rank=long_mask_rank,long_mask_seed=seed,
        wrapper_semantics="base",variant_type=gene_variant_type,rv_num_cutoff_max=1_000_000_000,
        rv_num_cutoff_max_prefilter=1_000_000_000)
    from .cache_runtime import CacheSpec
    from .pipeline import AnalysisOptions
    AnalysisOptions(**options)
    specs,bindings = {},[]
    for entry in entries:
        coverage = _annotation_coverage(entry, gene_variant_type=gene_variant_type, analyses=kinds,
                                        expected_sample_keys=sample_keys(pairs),expected_analysis=analysis)
        directory = Path(entry["container_directory"])
        digest = _sha256(directory/"manifest.json")
        binding = _proof(directory,digest)
        samples = np.load(directory/"samples.npy",allow_pickle=False)
        specs[entry["metadata_directory"]] = CacheSpec(str(directory),binding,partial(_proof,directory,digest),samples)
        bindings.append(dict(name=entry["name"],annotation_coverage=coverage,genotype_manifest_sha256=digest,
            metadata_manifest_sha256=_sha256(Path(entry["metadata_directory"])/"manifest.json"),
            gene_catalog_sha256=None if "gene_catalog" not in entry else _sha256(entry["gene_catalog"]),
            promoter_intervals_sha256=None if "promoter_intervals" not in entry else _sha256(entry["promoter_intervals"])))
    identity = dict(phenotype_sha256=_sha256(phenotype_csv),covariate_sha256=_sha256(covariate_csv),
        implementation_sha256=_implementation_sha256(),dataset_manifest_sha256=_sha256(manifest_path),
        sample_pairs_sha256=_sha256(_cache_path(root,manifest.get("sample_pairs","sample_pairs.npy"))),
        prepared_bindings=bindings,chromosomes=[entry["name"] for entry in entries],analyses=list(kinds),
        device=str(device),cpu_threads=cpu_threads,analysis_options=options,phenotype_families=families,
        phenotype_covariates=covariate_selection,
        transform=transform,null_fit_mode=null_fit_mode,gene_variant_type=gene_variant_type,single_mac_cutoff=single_mac_cutoff,
        single_group_variants=single_group_variants,single_region_size=single_region_size,
        individual_effective_block_size=individual_effective_block_size,
        shared_cache_bytes=[device_cache_bytes,compact_cache_bytes,metadata_cache_bytes])
    output = Path(output_directory).resolve()
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    plan_path,report_path = output/"plan.private.json",output/"report.private.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if not resume or previous.get("input_identity")!=identity or not report_path.is_file():
            raise ValueError("output directory contains an existing or incomplete run; use a new directory")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("completed") is not True:
            raise ValueError("only a completed identical run can be reused")
        for saved in report.get("files",[]):
            path = output/saved["path"]
            if not path.is_file() or _sha256(path)!=saved["sha256"]:
                raise ValueError("completed output changed after verification")
        return report
    if any(output.iterdir()):raise ValueError("output directory must be empty for a new run")
    # Build structural schedules once; every phenotype reuses the same
    # coordinate scan and annotation catalog rather than reopening metadata.
    schedules = [_jobs(entry,kinds,output/"_schedule",single_mac_cutoff=single_mac_cutoff,
        single_group_variants=single_group_variants,single_region_size=single_region_size)
        for entry in entries]
    from .io import save_null_model
    from .null_model import fit_gaussian_null,rank_inverse_normal
    from .binary_null import fit_logistic_null
    configs,models = [],[]
    model_directory = output/"models"
    model_directory.mkdir(mode=0o700)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        for ordinal,name in enumerate(phenotypes.names):
            ids,y,x,cov_names,alignment = aligned_phenotype(phenotypes,covariates,pairs,name,
                covariate_names=covariate_selection.get(name))
            family,before = families.get(name,"gaussian"),time.perf_counter()
            if family=="binomial":
                if transform!="none":raise ValueError("rint transformation requires Gaussian outcomes")
                model = fit_logistic_null(y,covariates=x,sample_ids=ids,device="cpu",use_spa=False)
            else:
                if transform=="rint":y=rank_inverse_normal(y)
                model = fit_gaussian_null(y,sample_ids=ids,covariates=x,
                    device="cpu" if null_fit_mode=="fp64" else device,matmul_mode=null_fit_mode)
            model_path = model_directory/f"trait_{ordinal+1:04d}.npz"
            model.genotype_sample_ids,model.sample_pairs = ids.copy(),pairs_from_keys(ids)
            save_null_model(model,model_path)
            models.append(dict(index=ordinal,name=name,family=family,alignment=alignment,
                covariate_names=list(cov_names),model_sha256=_sha256(model_path),null_fit_seconds=time.perf_counter()-before))
            config = dict(matmul_mode="tf32",precision_control=False,statistics_execution="serial",
                resident_genotypes=True,single_batch_optimization=True,weight_batch_optimization=True,
                statistics_tail_optimization=True,local_mask_reuse=True,maximum_mask_variants=None,
                individual_effective_block_size=individual_effective_block_size,
                individual_genotype_block_size=individual_effective_block_size,analysis_options=dict(options),
                qc_path=analysis["qc_path"],
                annotation_catalog=analysis["annotation_catalog"],annotation_names=analysis["annotation_names"],
                phenotypes=[dict(name=name,model=str(model_path),sample_id_rule="exact")],chromosomes=[])
            for entry,schedule in zip(entries,schedules):
                jobs = [dict(job,arguments=dict(job["arguments"]),
                             output=str(output/name/Path(job["output"]).name)) for job in schedule]
                chromosome = dict(name=entry["name"],genotype=entry["metadata_directory"],
                    jobs=jobs)
                if "promoter_intervals" in entry:
                    chromosome["annotation_index"] = dict(promoter_intervals_file=entry["promoter_intervals"])
                config["chromosomes"].append(chromosome)
            configs.append(config)
            del model,y,x
    finally:torch.set_num_threads(previous_threads)
    _json_write(plan_path,dict(schema_version=2,input_identity=identity,phenotypes=models,analyses=configs,
        prepared_directory=str(root),output_directory=str(output)))
    preparation_seconds = time.perf_counter()-started
    from .phewas_runtime.runtime import run_configuration
    report = run_configuration(configs,cache_specs=specs,device=device,cpu_threads=cpu_threads,
        device_cache_bytes=device_cache_bytes,compact_cache_bytes=compact_cache_bytes,
        metadata_cache_bytes=metadata_cache_bytes)
    files = [dict(path=str(path.relative_to(output)),size=path.stat().st_size,sha256=_sha256(path))
             for path in sorted(output.rglob("*")) if path.suffix in (".csv",".npz")]
    report.update(completed=True,schema_version=2,phenotype_inputs=models,files=files,
        preparation_seconds=preparation_seconds,end_to_end_seconds=time.perf_counter()-started,
        output_directory=str(output),input_identity=identity)
    _json_write(report_path,report)
    return report


def main(argv=None):
    def positive_integer(value):
        try:
            result = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a positive integer") from error
        if result < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return result
    parser = argparse.ArgumentParser(description="Run WGS associations from FID/IID CSVs and prepared genetic data")
    for name in ("phenotype_csv","covariate_csv","prepared_directory"):parser.add_argument(name)
    parser.add_argument("--output-directory",required=True)
    parser.add_argument("--chromosomes",nargs="+")
    parser.add_argument("--analyses",nargs="+",choices=_KINDS,default=_KINDS)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--cpu-threads",type=positive_integer,default=8)
    parser.add_argument("--memory-limit-gib",type=float,default=20.)
    parser.add_argument("--phenotype-families",help="JSON file mapping phenotype names to gaussian or binomial")
    parser.add_argument("--phenotype-covariates",help="JSON file mapping phenotype names to selected covariate column names")
    parser.add_argument("--transform",choices=("none","rint"),default="none")
    parser.add_argument("--null-fit-mode",choices=("fp64","tf32"),default="fp64")
    parser.add_argument("--gene-variant-type", choices=("SNV", "Indel", "variant"), default="SNV")
    for name,default in (("covariance-block-size",4096),("long-mask-threshold",5000),("long-mask-rank",512),
        ("seed",1729),("single-mac-cutoff",20),("single-group-variants",5000),("single-region-size",10_000_000),
        ("individual-effective-block-size",1024),("device-cache-bytes",512*2**20),
        ("compact-cache-bytes",64*2**20),("metadata-cache-bytes",256*2**20)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--resume",action="store_true")
    arguments = vars(parser.parse_args(argv))
    if arguments["phenotype_families"] is not None:
        arguments["phenotype_families"] = json.loads(Path(arguments["phenotype_families"]).read_text(encoding="utf-8"))
    if arguments["phenotype_covariates"] is not None:
        arguments["phenotype_covariates"] = json.loads(Path(arguments["phenotype_covariates"]).read_text(encoding="utf-8"))
    report = run_WGS_all(**arguments)
    print(json.dumps({key:report.get(key) for key in ("completed","eligible_association_tests","end_to_end_seconds","output_directory")},indent=2))


if __name__=="__main__":main()
