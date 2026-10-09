"""Three-file, cache-only Gaussian association workflow.

The result directory contains private participant/model and association data.
Only counts, timing, checksums and numeric configuration belong in public reports.
SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import argparse
import csv
import gc
from functools import wraps
from dataclasses import dataclass, replace
from contextlib import ExitStack
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sqlite3
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
import torch

_MISSING = {"", "na", "nan", "null", "none", "n/a"}
_KINDS = ("individual", "coding", "noncoding", "ncrna")
_FILE_DIGEST_CACHE = {}


class _HostLeaseDeferred(Exception):
    def __init__(self, required):
        self.required = required


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted((*package.rglob("*.py"), *package.rglob("*.cpp"))):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _bound_digest(path):
    path = str(path)
    stat = Path(path).stat()
    identity = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    previous = _FILE_DIGEST_CACHE.get(path)
    if previous is None or previous[0] != identity:
        previous = (identity,_sha256(path))
        _FILE_DIGEST_CACHE[path] = previous
    return previous[1]


def _json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _eid(value, *, where):
    value = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", value) or int(value) > np.iinfo(np.int64).max:
        raise ValueError(f"{where}: eid must be a positive decimal integer without decimals or leading zeros")
    return value


@dataclass(frozen=True)
class NumericCSV:
    sample_ids: np.ndarray
    names: tuple[str, ...]
    values: np.ndarray


def _read_csv(path, *, require_columns):
    path = Path(path)
    identifiers, rows, seen = [], [], set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[0].strip() != "eid":
            raise ValueError(f"{path.name}: the first column must be named eid")
        header = [name.strip() for name in header]
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError(f"{path.name}: column names must be nonempty and unique")
        if require_columns and len(header) < 2:
            raise ValueError(f"{path.name}: provide at least one named phenotype column")
        for line, row in enumerate(reader, 2):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name}:{line}: number of columns differs from the header")
            identifier = _eid(row[0], where=f"{path.name}:{line}")
            if identifier in seen:
                raise ValueError(f"{path.name}:{line}: duplicate eid")
            seen.add(identifier)
            values = []
            for name, raw in zip(header[1:], row[1:]):
                token = raw.strip()
                if token.lower() in _MISSING:
                    value = float("nan")
                else:
                    try:
                        value = float(token)
                    except ValueError as error:
                        raise ValueError(f"{path.name}:{line}: {name} must contain numeric values or missing values") from error
                    if not np.isfinite(value):
                        raise ValueError(f"{path.name}:{line}: infinity is not a valid numeric value")
                values.append(value)
            identifiers.append(identifier)
            rows.append(values)
    if not rows:
        raise ValueError(f"{path.name}: no participant rows")
    return NumericCSV(np.asarray(identifiers, dtype=str), tuple(header[1:]),
                      np.asarray(rows, dtype=np.float64).reshape(len(rows), len(header) - 1))


def read_csv_inputs(phenotype_csv, covariate_csv):
    """Read numeric CSVs without rounding participant IDs through float64.

    Missing phenotype values are removed separately for each phenotype.
    A missing covariate excludes that participant from every phenotype.
    This function reads inputs; ``prepare_run`` additionally aligns the cache.
    """
    return _read_csv(phenotype_csv, require_columns=True), _read_csv(covariate_csv, require_columns=False)


def aligned_phenotype(phenotypes, covariates, cache_ids, trait):
    """Return ascending standard IDs, complete-case y, and covariates with an intercept."""
    if trait not in phenotypes.names:
        raise ValueError("unknown phenotype column")
    cache_ids = np.asarray([_eid(value, where="cache sample index") for value in cache_ids], dtype=str)
    if len(np.unique(cache_ids)) != len(cache_ids):
        raise ValueError("cache sample index has duplicate eid values")
    cache_set = set(cache_ids.tolist())
    cov_lookup = {identifier: index for index, identifier in enumerate(covariates.sample_ids)}
    candidate = [(index, cov_lookup[identifier]) for index, identifier in enumerate(phenotypes.sample_ids)
                 if identifier in cache_set and identifier in cov_lookup]
    candidate.sort(key=lambda pair: int(phenotypes.sample_ids[pair[0]]))
    pheno_rows = np.asarray([pair[0] for pair in candidate], dtype=np.int64)
    cov_rows = np.asarray([pair[1] for pair in candidate], dtype=np.int64)
    column = phenotypes.names.index(trait)
    y = phenotypes.values[pheno_rows, column]
    x = covariates.values[cov_rows]
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    identifiers, y, x = phenotypes.sample_ids[pheno_rows][keep], y[keep], x[keep]
    # An explicitly supplied all-one column already is the intercept.
    has_intercept = bool(x.shape[1] and np.any(np.all(x == 1, axis=0)))
    names = covariates.names
    if not has_intercept:
        x = np.column_stack((np.ones(len(y)), x))
        names = ("Intercept", *names)
    if len(y) <= x.shape[1]:
        raise ValueError(f"phenotype {trait}: insufficient complete-case samples for the covariate design")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError(f"phenotype {trait}: covariate design is rank deficient")
    return identifiers, y, x, names, {
        "phenotype_rows": len(phenotypes.sample_ids), "covariate_rows": len(covariates.sample_ids),
        "cache_rows": len(cache_ids), "matched_rows_before_missing": len(candidate),
        "excluded_missing_rows": int((~keep).sum()), "analysis_samples": len(y),
        "phenotype_rows_absent_from_cache": sum(identifier not in cache_set for identifier in phenotypes.sample_ids),
        "phenotype_rows_absent_from_covariates": sum(identifier not in cov_lookup for identifier in phenotypes.sample_ids),
        "intercept_added": not has_intercept, "design_columns": x.shape[1],
    }


def _cache_path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("cache manifest paths must be nonempty strings")
    if Path(value).is_absolute():
        raise ValueError("cache manifest paths must be relative to the cache directory")
    # Validate the logical cache tree while allowing deliberate dataset links
    # to reused transferred chromosome directories on another filesystem.
    path = Path(os.path.abspath(root / value))
    if not path.is_relative_to(root):
        raise ValueError("cache manifest paths must remain within the cache directory")
    return path


def _dataset(cache_directory, chromosomes=None):
    root = Path(cache_directory).resolve()
    manifest_path = root / "cache_dataset.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported cache_dataset.json schema_version")
    entries = manifest.get("chromosomes")
    if isinstance(entries, dict):
        entries = [dict(value, name=str(key)) for key, value in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError("cache_dataset.json must declare chromosome entries")
    wanted = None if chromosomes is None else {str(value).removeprefix("chr") for value in chromosomes}
    selected, seen = [], set()
    for raw in entries:
        entry = dict(raw)
        name = str(entry["name"]).removeprefix("chr")
        if not re.fullmatch(r"[1-9][0-9]*", name) or not 1 <= int(name) <= 22 or name in seen:
            raise ValueError("chromosome entries must have distinct names from 1 to 22")
        seen.add(name)
        if wanted is not None and name not in wanted:
            continue
        container = entry.get("container_directory", entry.get("cache_directory"))
        entry.update(name=name, container_directory=str(_cache_path(root, container)))
        entry["metadata_directory"] = str(_cache_path(root, entry.get("metadata_directory", container + "/metadata")))
        for key in ("gene_catalog", "promoter_intervals"):
            if key in entry:
                entry[key] = str(_cache_path(root, entry[key]))
        selected.append(entry)
    if wanted is not None and wanted - seen:
        raise ValueError("requested chromosomes are absent from the cache")
    selected.sort(key=lambda entry: int(entry["name"]))
    if not selected:
        raise ValueError("no chromosomes selected")
    sample_path = _cache_path(root, manifest.get("sample_ids", "sample_ids.npy"))
    samples = np.load(sample_path, allow_pickle=False)
    if samples.ndim != 1:
        raise ValueError("cache sample index must be one dimensional")
    return root, manifest_path, manifest, selected, samples


def _analysis_defaults(memory_limit_gib, covariance_block_size, long_mask_threshold, long_mask_rank, seed):
    return dict(memory_limit_gib=memory_limit_gib, genotype_block_size=128,
        annotation_block_size=250_000, variant_tile_size=512,
        covariance_backend="cached", cached_variant_tile_size=covariance_block_size,
        long_mask_threshold=long_mask_threshold, long_mask_method="fastskat",
        long_mask_rank=long_mask_rank, long_mask_seed=seed, wrapper_semantics="base", variant_type="variant")


def prepare_run(phenotype_csv, covariate_csv, cache_directory, *, output_directory=None,
                chromosomes=None, analyses=_KINDS, memory_limit_gib=40.0,
                covariance_block_size=4096, long_mask_threshold=5000, long_mask_rank=512,
                seed=1729, individual_effective_block_size=1024, single_mac_cutoff=20,
                single_group_variants=5000, single_output_groups=20, transform="none",
                matmul_mode="tf32", null_fit_mode="fp64", analysis_options=None,
                host_memory_limit_gib=200.0, host_memory_reserve_gib=20.0, cpu_threads_per_worker=2, resume=True):
    """Validate three inputs, fit each ordinary Gaussian model and persist a resumable plan.

    Null fitting explicitly uses FP64 by default; association uses TF32/FP32.
    No original GDS path or fitted-model input is needed or read.
    ``analyses`` defaults to individual, coding, noncoding, and ncRNA.
    """
    from .io import save_null_model
    from .null_model import fit_gaussian_null, rank_inverse_normal
    if not 0 < memory_limit_gib <= 40:
        raise ValueError("memory_limit_gib must be in (0, 40]")
    for name, value in (("covariance_block_size", covariance_block_size), ("long_mask_threshold", long_mask_threshold),
                        ("long_mask_rank", long_mask_rank), ("individual_effective_block_size", individual_effective_block_size),
                        ("single_group_variants", single_group_variants), ("single_output_groups", single_output_groups)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0 or single_mac_cutoff < 0 or host_memory_limit_gib <= 0:
        raise ValueError("invalid random seed, MAC cutoff or host memory limit")
    if not 0 <= host_memory_reserve_gib < host_memory_limit_gib or type(cpu_threads_per_worker) is not int or cpu_threads_per_worker < 1:
        raise ValueError("host reserve must be below the host limit; CPU threads must be a positive integer")
    if matmul_mode not in ("tf32", "fp64") or null_fit_mode not in ("fp64", "tf32"):
        raise ValueError("matmul_mode and null_fit_mode must be tf32 or fp64")
    if transform not in ("none", "rint"):
        raise ValueError("transform must be none or rint")
    kinds = tuple(dict.fromkeys(analyses))
    if not kinds or any(kind not in _KINDS for kind in kinds):
        raise ValueError("analyses must select individual, coding, noncoding or ncrna")
    root, manifest_path, manifest, entries, cache_ids = _dataset(cache_directory, chromosomes)
    options = _analysis_defaults(memory_limit_gib, covariance_block_size, long_mask_threshold, long_mask_rank, seed)
    if analysis_options:
        options.update(dict(analysis_options))
        if options.get("memory_limit_gib", memory_limit_gib) != memory_limit_gib:
            raise ValueError("analysis_options.memory_limit_gib conflicts with memory_limit_gib")
    identity = dict(phenotype_sha256=_sha256(phenotype_csv), covariate_sha256=_sha256(covariate_csv),
        implementation_sha256=_implementation_sha256(),
        cache_manifest_sha256=_sha256(manifest_path), cache_sample_ids_sha256=_sha256(root / manifest.get("sample_ids", "sample_ids.npy")),
        chromosomes=[entry["name"] for entry in entries], analyses=list(kinds), analysis_options=options,
        matmul_mode=matmul_mode, null_fit_mode=null_fit_mode, transform=transform,
        individual_effective_block_size=individual_effective_block_size, single_mac_cutoff=single_mac_cutoff,
        single_group_variants=single_group_variants, single_output_groups=single_output_groups,
        host_memory_limit_gib=host_memory_limit_gib,host_memory_reserve_gib=host_memory_reserve_gib,
        cpu_threads_per_worker=cpu_threads_per_worker)
    cache_bindings, initial_metadata = {}, {}
    for entry in entries:
        binding = {}
        for key,path in (("genotype_manifest",Path(entry["container_directory"])/"manifest.json"),
                         ("gene_catalog",Path(entry["gene_catalog"]) if "gene_catalog" in entry else None),
                         ("promoter_intervals",Path(entry["promoter_intervals"]) if "promoter_intervals" in entry else None)):
            if path is not None and path.is_file():
                binding[key] = dict(path=str(path),sha256=_sha256(path))
        metadata = Path(entry["metadata_directory"])/"manifest.json"
        if (metadata.parent/"COMPLETE").is_file():
            initial_metadata[entry["name"]] = _sha256(metadata)
        cache_bindings[entry["name"]] = binding
    identity["cache_bindings"] = cache_bindings
    if output_directory is None:
        output_directory = Path(phenotype_csv).resolve().parent / ("torchstaar_results_" + time.strftime("%Y%m%d_%H%M%S"))
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan_path = output / "plan.private.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if not resume or previous.get("input_identity") != identity:
            raise ValueError("result directory already contains a different plan; use a new output directory")
        for saved in previous.get("phenotypes",[]):
            if _sha256(saved["path"]) != saved["sha256"]:
                raise ValueError("persisted null model changed after input preparation")
        for entry in entries:
            expected = previous.get("initial_metadata_manifest_sha256",{}).get(entry["name"])
            if expected and _sha256(Path(entry["metadata_directory"])/"manifest.json") != expected:
                raise ValueError("chromosome metadata changed after input preparation")
        return plan_path
    phenotypes, covariates = read_csv_inputs(phenotype_csv, covariate_csv)
    models = []
    (output / "models").mkdir(exist_ok=True, mode=0o700)
    for ordinal, trait in enumerate(phenotypes.names):
        before = time.perf_counter()
        ids, y, x, names, alignment = aligned_phenotype(phenotypes, covariates, cache_ids, trait)
        if transform == "rint":
            y = rank_inverse_normal(y)
        fit_device = "cpu" if null_fit_mode == "fp64" else "cuda:0"
        model = fit_gaussian_null(y, sample_ids=ids, covariates=x,
                                  device=fit_device, matmul_mode=null_fit_mode)
        model_path = output / "models" / f"trait_{ordinal + 1:04d}.npz"
        save_null_model(model, model_path)
        models.append(dict(ordinal=ordinal, name=trait, path=str(model_path), sha256=_sha256(model_path),
            covariate_names=list(names), alignment=alignment, null_fit_seconds=time.perf_counter() - before,
            null_fit_mode=null_fit_mode, family="gaussian", kinship=False))
        del model, y, x
    tasks = []
    for trait in models:
        for entry in entries:
            if "individual" in kinds:
                tasks.append(dict(trait=trait["ordinal"], chromosome=entry["name"], kind="individual", arguments={}))
            if not any(kind in kinds for kind in ("coding", "noncoding", "ncrna")):
                continue
            if "gene_catalog" not in entry:
                raise ValueError("complete gene analysis requires a gene_catalog in each chromosome cache entry")
            genes = json.loads(Path(entry["gene_catalog"]).read_text())
            if isinstance(genes, dict):
                genes = genes.get("jobs", genes.get("genes"))
            if not isinstance(genes, list):
                raise ValueError("gene_catalog must contain a list of gene jobs")
            for gene in genes:
                kind = gene.get("kind")
                if kind not in kinds:
                    continue
                arguments = dict(gene.get("arguments", {}))
                arguments.pop("promoter_intervals_file",None)
                arguments.pop("chromosome",None)
                for key in ("gene_name", "start", "end", "category", "include_ptv", "include_ncrna"):
                    if key in gene:
                        arguments[key] = gene[key]
                if not isinstance(arguments.get("gene_name"), str) or not arguments["gene_name"]:
                    raise ValueError("gene jobs require gene_name")
                if kind == "coding" and ("start" not in arguments or "end" not in arguments):
                    raise ValueError("coding jobs require inclusive start and end coordinates")
                if "start" in arguments and "end" in arguments and int(arguments["start"]) > int(arguments["end"]):
                    raise ValueError("gene job start exceeds end")
                tasks.append(dict(trait=trait["ordinal"], chromosome=entry["name"], kind=kind, arguments=arguments,
                    max_variants=gene.get("max_variants"), host_memory_reserve_gib=gene.get("host_memory_reserve_gib")))
    plan = dict(schema_version=1, input_identity=identity, cache_directory=str(root), output_directory=str(output),
        initial_metadata_manifest_sha256=initial_metadata,
        phenotypes=models, chromosomes=entries, dataset_analysis={key:manifest.get(key) for key in
        ("annotation_catalog", "annotation_names", "qc_path") if key in manifest}, task_count=len(tasks))
    database = output / "jobs.sqlite.preparing"
    if database.exists():
        raise ValueError("an unfinished input-preparation database exists; use a new output directory")
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER, owner_identity TEXT, started REAL, finished REAL, report TEXT, error TEXT, deferrals INTEGER NOT NULL DEFAULT 0)")
        connection.execute("CREATE TABLE leases (job_id INTEGER PRIMARY KEY, gib REAL NOT NULL)")
        connection.execute("CREATE TABLE workers (pid INTEGER PRIMARY KEY, identity TEXT NOT NULL)")
        connection.execute("CREATE TABLE chromosome_bindings (chromosome TEXT PRIMARY KEY, metadata_sha256 TEXT NOT NULL)")
        connection.executemany("INSERT INTO jobs(id,payload,status) VALUES(?,?, 'pending')",
                               ((index + 1, json.dumps(task)) for index, task in enumerate(tasks)))
    database.replace(output / "jobs.sqlite")
    _json_write(plan_path, plan)
    return plan_path


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_identity(pid):
    """Linux start ticks and boot ID distinguish a worker from a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        return dict(pid=int(pid), start_ticks=int(fields[19]),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (OSError, ValueError, IndexError):
        return None


def _probability_gate(result):
    """Reject invalid completed scientific values before serializing a result."""
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (int, float, np.number)):
                probability = key == "pvalue" or key.startswith(("SKAT", "Burden", "ACAT", "STAAR"))
                if probability and (not np.isfinite(value) or not 0 <= value <= 1):
                    raise ArithmeticError("result contains an invalid probability")
                if key in ("pvalue_log", "pvalue_log10", "Score", "Score_se", "Est", "Est_se") and not np.isfinite(value):
                    raise ArithmeticError("result contains a nonfinite Single statistic")
            elif isinstance(value, (dict, list, tuple)):
                _probability_gate(value)
    elif isinstance(result, (list, tuple)):
        for value in result:
            _probability_gate(value)


def _rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError,ValueError):
        pass
    return 0


def _read_host_memory():
    """Read live cgroup RSS pressure, excluding reclaimable inactive file pages."""
    v1,v2 = Path("/sys/fs/cgroup/memory"),Path("/sys/fs/cgroup")
    try:
        if (v1 / "memory.limit_in_bytes").is_file():
            current = int((v1 / "memory.usage_in_bytes").read_text())
            maximum = int((v1 / "memory.limit_in_bytes").read_text())
            stats = dict(line.split() for line in (v1 / "memory.stat").read_text().splitlines())
            inactive = int(stats.get("total_inactive_file",stats.get("inactive_file",0)))
            if maximum < 2**60:
                return dict(current_bytes=current,limit_bytes=maximum,inactive_file_bytes=inactive,
                            effective_current_bytes=max(0,current-inactive),scope="cgroup_v1")
        if (v2 / "memory.max").is_file():
            raw = (v2 / "memory.max").read_text().strip()
            if raw != "max":
                current = int((v2 / "memory.current").read_text())
                stats = dict(line.split() for line in (v2 / "memory.stat").read_text().splitlines())
                inactive = int(stats.get("inactive_file",0))
                return dict(current_bytes=current,limit_bytes=int(raw),inactive_file_bytes=inactive,
                            effective_current_bytes=max(0,current-inactive),scope="cgroup_v2")
        values = dict((line.split(":",1)[0],int(line.split()[1])*1024) for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith(("MemTotal:","MemAvailable:")))
        if "MemTotal" in values and "MemAvailable" in values:
            effective = values["MemTotal"]-values["MemAvailable"]
            return dict(current_bytes=effective,limit_bytes=values["MemTotal"],inactive_file_bytes=0,
                        effective_current_bytes=effective,scope="system_available_memory")
    except (OSError,ValueError):
        pass
    return None


def _host_budget(connection, plan, *, desired_gib, extra_gib, excluding_job):
    config = plan["input_identity"]
    limit = config["host_memory_limit_gib"] * 1024**3
    reserve = config.get("host_memory_reserve_gib",20.) * 1024**3
    live = _read_host_memory()
    if live:
        limit = min(limit,live["limit_bytes"])
    capacity = limit-reserve
    used = connection.execute("SELECT COALESCE(SUM(gib),0) FROM leases WHERE job_id!=?",(excluding_job,)).fetchone()[0]*1024**3
    baseline = plan.get("runtime_baseline_bytes",0)
    if live:
        project_rss = 0
        try:
            identities = connection.execute("SELECT pid,identity FROM workers").fetchall()
        except sqlite3.OperationalError:
            identities = []
        for pid, raw in identities:
            if _process_identity(pid) == json.loads(raw):
                project_rss += _rss_bytes(pid)
        # Current other applications and exports are measured rather than
        # frozen into the launch baseline. Live+new independently checks
        # pressure even when reclaimable mmap pages inflate process RSS.
        baseline = max(0,live["effective_current_bytes"]-project_rss)
    predicted = baseline + used + desired_gib*1024**3
    live_plus_request = (live["effective_current_bytes"] if live else 0) + max(0,extra_gib)*1024**3
    return dict(admissible=predicted <= capacity and live_plus_request <= capacity,
        capacity_bytes=capacity,predicted_peak_bytes=predicted,live_plus_request_bytes=live_plus_request,
        live=live,reserve_bytes=reserve)


def _claim(connection, plan):
    """Reserve host workspace before claiming a mask, in the same SQLite transaction."""
    ready = [entry["name"] for entry in plan["chromosomes"]
             if (Path(entry["metadata_directory"]) / "COMPLETE").is_file()
             and (Path(entry["container_directory"]) / "COMPLETE").is_file()]
    if not ready:
        return None
    connection.execute("BEGIN IMMEDIATE")
    try:
        used = connection.execute("SELECT COALESCE(SUM(gib),0) FROM leases WHERE job_id!=?",(-os.getpid(),)).fetchone()[0]
        limit = plan["input_identity"]["host_memory_limit_gib"]
        preliminary = _host_budget(connection,plan,desired_gib=0,extra_gib=0,excluding_job=-os.getpid())
        capacity_gib = preliminary["capacity_bytes"]/1024**3
        selected = None
        marks = ",".join("?" for _ in ready)
        for job_id, payload in connection.execute(
                f"SELECT id,payload FROM jobs WHERE status='pending' AND json_extract(payload,'$.chromosome') IN ({marks}) ORDER BY id LIMIT 256",ready):
            job = json.loads(payload)
            count = plan["phenotypes"][job["trait"]]["alignment"]["analysis_samples"]
            reserve = job.get("host_memory_reserve_gib")
            if reserve is None:
                # Actual rare M is admitted by host_memory_guard before any
                # floating host genotype allocation. Unknown M reserves only
                # metadata/model preparation at this stage.
                maximum = job.get("max_variants")
                reserve = 8. if maximum is None or int(maximum) <= 0 else max(8., 2. * count * int(maximum) * 4 / 1024**3 + 8.)
                # Candidate bounds are not actual rare M. An oversized bound
                # enters metadata preparation and the actual-M guard decides.
                if reserve > capacity_gib:
                    reserve = 8.
            reserve = max(float(reserve),_rss_bytes(os.getpid())/1024**3)
            if not np.isfinite(reserve) or reserve <= 0 or reserve > capacity_gib:
                connection.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?",
                    (time.time(), "estimated host workspace exceeds configured host limit", job_id))
                continue
            budget = _host_budget(connection,plan,desired_gib=reserve,
                                  extra_gib=max(0,reserve-_rss_bytes(os.getpid())/1024**3),excluding_job=-os.getpid())
            if used + reserve <= limit and budget["admissible"]:
                selected = (job_id, job, reserve)
                break
        if selected is not None:
            job_id, job, reserve = selected
            connection.execute("UPDATE jobs SET status='running',owner_pid=?,owner_identity=?,started=? WHERE id=?", (os.getpid(),json.dumps(_process_identity(os.getpid())),time.time(),job_id))
            connection.execute("DELETE FROM leases WHERE job_id=?",(-os.getpid(),))
            connection.execute("INSERT INTO leases VALUES(?,?)", (job_id,reserve))
        connection.commit()
        return selected
    except BaseException:
        connection.rollback()
        raise


def _host_memory_guard(connection, plan, job_id, *, n, m, phase="prepare"):
    """Increase the live mask's shared host lease before materializing genotype arrays."""
    limit = plan["input_identity"]["host_memory_limit_gib"]
    copies = 3 if phase == "prepare_nonresident" else 2
    new_allocation = copies * int(n) * int(m) * 4 / 1024**3 + 2.
    required = max(8., _rss_bytes(os.getpid()) / 1024**3 + new_allocation)
    preliminary = _host_budget(connection,plan,desired_gib=0,extra_gib=0,excluding_job=job_id)
    if required*1024**3 > preliminary["capacity_bytes"]:
        raise MemoryError("actual mask host workspace exceeds the configured project host limit")
    before = time.perf_counter()
    connection.execute("BEGIN IMMEDIATE")
    others = connection.execute("SELECT COALESCE(SUM(gib),0) FROM leases WHERE job_id!=?",(job_id,)).fetchone()[0]
    current = connection.execute("SELECT gib FROM leases WHERE job_id=?",(job_id,)).fetchone()[0]
    required = max(required,current)
    budget = _host_budget(connection,plan,desired_gib=required,extra_gib=new_allocation,excluding_job=job_id)
    if others + required <= limit and budget["admissible"]:
        connection.execute("UPDATE leases SET gib=? WHERE job_id=?",(required,job_id))
        connection.commit()
        return time.perf_counter() - before
    connection.rollback()
    # Eight workers must not all retain smaller leases while waiting for
    # larger ones. Requeue this job with the now-known requirement and drop
    # its live preparation; admission will precede the next materialization.
    raise _HostLeaseDeferred(required)


def _single_outputs(pipeline, chromosome, directory, identity, arguments=None):
    """Flush only completed original 5000-variant groups; bounded host result storage."""
    from .r_output import write_association_batch
    directory.mkdir(parents=True, exist_ok=True)
    arguments = dict(arguments or {})
    if set(arguments)-{"start","end"}:
        raise ValueError("Single job arguments may only select inclusive start/end coordinates")
    if (arguments.get("start") is None) != (arguments.get("end") is None):
        raise ValueError("Single region requires both start and end")
    region = {key:int(value) for key,value in arguments.items() if value is not None}
    if region and (region["start"] < 1 or region["start"] > region["end"]):
        raise ValueError("invalid Single region coordinates")
    records, first_chunk, part, emitted, outputs = [], None, 0, 0, []
    levels = {"REF": [], "ALT": []}
    def flush():
        nonlocal records, part, emitted
        if not records:
            return
        part += 1
        tables = pipeline.individual_tables([records])
        table = tables[0]
        _probability_gate(table)
        table.row_names = [value + emitted for value in table.row_names]
        for key in levels:
            levels[key].extend(value for value in table.factor_levels[key] if value not in levels[key])
            table.factor_levels[key] = list(levels[key])
        path = directory / f"part_{part:06d}.Rdata"
        write_association_batch(path, [tables], kind="individual", layout="base")
        validation = _verify_native_output(path,kind="individual",expected_rows=len(table),expected_result=tables)
        outputs.append(dict(path=str(path), sha256=_sha256(path), rows=len(table),native_validation=validation))
        emitted += len(table)
        records = []
    for trait, batch in pipeline.iter_individual_records(chromosome,**region,
            mac_cutoff=identity["single_mac_cutoff"], variant_type="variant",
            subset_variants_num=identity["single_group_variants"]):
        if trait != 0:
            raise RuntimeError("single-phenotype worker returned another trait")
        for row in batch:
            chunk = row["_chunk"]
            if first_chunk is None:
                first_chunk = chunk
            if chunk - first_chunk >= identity["single_output_groups"]:
                flush()
                first_chunk = chunk
            records.append(row)
    flush()
    if not outputs:
        tables = pipeline.individual_tables([[]])
        path = directory / "part_000001.Rdata"
        write_association_batch(path,[tables],kind="individual",layout="base")
        outputs.append(dict(path=str(path),sha256=_sha256(path),rows=0,
            native_validation=_verify_native_output(path,kind="individual",expected_rows=0)))
    _json_write(directory / "index.private.json", dict(schema_version=1, rows=emitted, parts=outputs,
        final_factor_levels=levels,requested_region=region or None,
        scope="selected interval" if region else "whole chromosome",
        row_names="global analysed-scan ordinals; each part retains original group order"))
    return outputs, emitted


def _verify_native_output(path, *, kind, expected_rows=None, expected_result=None):
    """Read the real native file; Single additionally validates its complete table contract."""
    import rdata
    value = rdata.read_rda(path)
    names = {"coding":"results_coding","noncoding":"results_noncoding","ncrna":"results_ncRNA",
             "individual":"results_individual_analysis"}
    name = names[kind]
    if list(value) != [name]:
        raise ValueError("native output has an unexpected object name")
    if kind == "individual":
        table = value[name]
        if table is None:
            if expected_rows != 0:
                raise ValueError("native Single output is empty despite calculated rows")
        else:
            required = {"CHR","POS","REF","ALT","ALT_AF","MAF","N","pvalue","pvalue_log10","Score","Score_se","Est","Est_se"}
            if not required <= set(table.columns) or len(table) != expected_rows:
                raise ValueError("native Single columns or row count differ from calculated output")
            for field in ("pvalue","pvalue_log10","Score","Score_se","Est","Est_se"):
                array = np.asarray(table[field],dtype=float)
                if not np.isfinite(array).all():
                    raise ValueError("native Single table has a nonfinite statistic")
                if field == "pvalue" and ((array < 0) | (array > 1)).any():
                    raise ValueError("native Single table has an invalid probability")
            for field in ("REF","ALT"):
                if not hasattr(table[field].dtype,"categories"):
                    raise ValueError("native Single allele columns must preserve R factors")
            if expected_result is not None:
                expected = expected_result[0]
                if list(table.columns) != list(expected[0]) or [str(value) for value in table.index] != [str(value) for value in expected.row_names]:
                    raise ValueError("native Single columns or global row names changed during serialization")
                for field in table.columns:
                    saved = table[field].astype(str).to_numpy() if field in ("REF","ALT") else np.asarray(table[field],dtype=float)
                    wanted = np.asarray([row[field] for row in expected],dtype=str if field in ("REF","ALT") else float)
                    if not np.array_equal(saved,wanted):
                        raise ValueError("native Single serialization changed a computed value")
                for field in ("REF","ALT"):
                    if list(table[field].dtype.categories) != expected.factor_levels[field]:
                        raise ValueError("native Single serialization changed factor levels")
    elif expected_result is not None:
        from .r_output import association_object, RMatrix
        wanted = association_object(expected_result,kind=kind,layout="base")
        def exact(saved, reference):
            if reference is None:
                if saved is not None:
                    raise ValueError("native output contains an unexpected nonempty mask")
                return
            if isinstance(reference,Mapping):
                if not isinstance(saved,Mapping) or list(saved) != list(reference):
                    raise ValueError("native output mask categories differ")
                for category in reference:
                    exact(saved[category],reference[category])
            elif isinstance(reference,RMatrix):
                cells = reference.values.ravel(order="F")
                if not isinstance(saved,list) or len(saved) != len(cells):
                    raise ValueError("native result matrix shape differs")
                for cell,expected in zip(saved,cells):
                    raw = np.asarray(cell)
                    if raw.size != 1:
                        raise ValueError("native result matrix contains a nonscalar cell")
                    observed = raw.reshape(-1)[0]
                    if isinstance(expected,str):
                        same = str(observed) == expected
                    else:
                        same = np.isfinite(float(observed)) and float(observed) == float(expected)
                    if not same:
                        raise ValueError("native gene serialization changed a computed value")
            else:
                raise ValueError("unsupported expected native gene topology")
        exact(value[name],wanted)
    return dict(parsed=True,object_count=1,expected_rows=expected_rows,
                computed_values_exactly_preserved=expected_result is not None)


def _bind_chromosome(connection, plan, entry):
    """Bind late-finished metadata once; every worker checks against the same content hash."""
    expected = plan["input_identity"]["cache_bindings"][entry["name"]]
    for binding in expected.values():
        if _bound_digest(binding["path"]) != binding["sha256"]:
            raise ValueError("chromosome cache/catalog changed after input preparation")
    directory = Path(entry["metadata_directory"])
    digest = _bound_digest(directory/"manifest.json")
    initial = plan.get("initial_metadata_manifest_sha256",{}).get(entry["name"])
    if initial and initial != digest:
        raise ValueError("chromosome metadata changed after input preparation")
    if (directory/"COMPLETE").read_text().strip() != digest:
        raise ValueError("portable metadata completion marker differs")
    connection.execute("BEGIN IMMEDIATE")
    previous = connection.execute("SELECT metadata_sha256 FROM chromosome_bindings WHERE chromosome=?",(entry["name"],)).fetchone()
    if previous is None:
        connection.execute("INSERT INTO chromosome_bindings VALUES(?,?)",(entry["name"],digest))
    elif previous[0] != digest:
        connection.rollback()
        raise ValueError("chromosome metadata changed since its first dispatch")
    connection.commit()
    return digest


def _shared_annotation_index(pipeline, entry, plan, metadata_digest, *, kind):
    """Build candidates once per chromosome; other workers share read-only arrays."""
    import fcntl
    from .annotation_index import CandidateAnnotationIndex
    from .masks import NONCODING_CATEGORIES
    chromosome = entry["name"]
    variant_type = pipeline.options.variant_type
    key = (chromosome,variant_type)
    categories = list(NONCODING_CATEGORIES) if kind == "noncoding" else ["ncRNA"]
    requested = set(categories)|{"ncRNA"}
    if key in pipeline._annotation_indexes and requested <= pipeline._annotation_indexes[key].prepared_categories:
        return dict(already_loaded=True,seconds=0.)
    promoters = None
    if "promoter_intervals" in entry:
        promoters = json.loads(Path(entry["promoter_intervals"]).read_text())
    binding = dict(metadata_manifest_sha256=metadata_digest,variant_type=variant_type,
        categories=sorted(requested),
        chromosome=chromosome,annotation_catalog=pipeline.annotation_catalog,qc_path=pipeline.qc_path,
        promoter_sha256=_bound_digest(entry["promoter_intervals"]) if promoters is not None else None,
        mask_source_sha256={name:_sha256(Path(__file__).parent/name) for name in
                            ("masks.py","annotation_index.py","pipeline.py")})
    scope = "all" if kind == "noncoding" else "ncrna"
    directory = Path(plan["output_directory"])/"shared_indices"/f"chr{int(chromosome):02d}_{variant_type}_{scope}"
    directory.mkdir(parents=True,exist_ok=True)
    before = time.perf_counter()
    with (directory/"build.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest = directory/"manifest.private.json"
        if not manifest.exists():
            index = pipeline.prepare_annotation_index(chromosome,promoter_intervals=promoters,
                include_ncrna=True,categories=categories)
            entries,offset = [],0
            for gene in index.genes:
                for category in index.categories_for(gene):
                    rows = index.indices(gene,category)
                    entries.append([gene,category,offset,offset+len(rows)])
                    offset += len(rows)
            temporary = directory/"indices.preparing.npy"
            if offset:
                array = np.lib.format.open_memmap(temporary,mode="w+",dtype=np.uint32,shape=(offset,))
                for gene,category,start,end in entries:
                    rows = index.indices(gene,category)
                    if len(rows) and int(rows[-1]) >= 2**32:
                        raise ValueError("candidate variant indices exceed uint32 storage")
                    array[start:end] = rows
                array.flush()
                del array
            else:
                np.save(temporary,np.empty(0,dtype=np.uint32),allow_pickle=False)
            temporary.replace(directory/"indices.npy")
            np.save(directory/"base_mask.npy",pipeline._base_masks[key],allow_pickle=False)
            np.save(directory/"category_codes.npy",pipeline._category_codes,allow_pickle=False)
            document = dict(schema_version=1,binding=binding,entries=entries,candidate_rows=offset,
                prepared_categories=sorted(index.prepared_categories),promoter_signature=index.promoter_signature,
                files={name:_sha256(directory/name) for name in ("indices.npy","base_mask.npy","category_codes.npy")},
                build_seconds=time.perf_counter()-before)
            _json_write(manifest,document)
        else:
            document = json.loads(manifest.read_text())
        if document["schema_version"] != 1 or document["binding"] != binding:
            raise ValueError("shared annotation index is bound to other metadata or mask semantics")
        for name,digest in document["files"].items():
            if _bound_digest(directory/name) != digest:
                raise ValueError("shared annotation index array changed")
        array = np.load(directory/"indices.npy",mmap_mode="r" if document["candidate_rows"] else None,allow_pickle=False)
        if array.dtype != np.uint32 or array.ndim != 1 or len(array) != document["candidate_rows"]:
            raise ValueError("shared annotation index row geometry differs")
        index = CandidateAnnotationIndex(chromosome,variant_type)
        index.prepared_categories = set(document["prepared_categories"])
        signature = document["promoter_signature"]
        index.promoter_signature = tuple(map(tuple,signature)) if signature is not None else None
        for gene,category,start,end in document["entries"]:
            if not 0 <= start <= end <= len(array):
                raise ValueError("shared annotation index offset is invalid")
            rows = array[start:end]
            if len(rows) and (int(rows[-1]) >= pipeline.gds.n_variants or np.any(rows[1:] <= rows[:-1])):
                raise ValueError("shared annotation indices must be unique, ordered physical variant rows")
            index._groups.setdefault(gene,{})[category] = rows
        base = np.load(directory/"base_mask.npy",mmap_mode="r",allow_pickle=False)
        codes = np.load(directory/"category_codes.npy",mmap_mode="r",allow_pickle=False)
        if base.shape != (pipeline.gds.n_variants,) or base.dtype != np.bool_ or codes.shape != base.shape or codes.dtype != np.uint8:
            raise ValueError("shared annotation mask geometry differs")
        pipeline._annotation_indexes[key] = index
        pipeline._base_masks[key] = base
        pipeline._category_codes = codes
    return dict(already_loaded=False,seconds=time.perf_counter()-before,build_seconds=document["build_seconds"],
                candidate_rows=document["candidate_rows"],mmap_read_only=True)


def _run_worker(plan_path, device):
    from .cache_runtime.portable import PortableCachedGDS
    from .io import load_null_model
    from .pipeline import AnalysisOptions, PheWASPipeline
    from .profiling import StageProfiler
    from .r_output import write_association_batch
    from .tf32 import configure_tf32
    from .precision_audit import DenseProductAudit
    plan = json.loads(Path(plan_path).read_text())
    identity = plan["input_identity"]
    connection = sqlite3.connect(Path(plan["output_directory"]) / "jobs.sqlite", timeout=60)
    connection.execute("INSERT OR REPLACE INTO workers VALUES(?,?)",(os.getpid(),json.dumps(_process_identity(os.getpid()))))
    connection.execute("INSERT OR REPLACE INTO leases VALUES(?,?)",(-os.getpid(),max(1.,_rss_bytes(os.getpid())/1024**3)))
    connection.commit()
    torch.set_num_threads(identity["cpu_threads_per_worker"])
    configure_tf32(memory_limit_gib=identity["analysis_options"]["memory_limit_gib"])
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., identity["analysis_options"]["memory_limit_gib"] * 1024**3 / total),device=device)
    key, reader, pipeline = None, None, None
    backend = ExitStack()
    from . import _weighted_spectra
    backend.enter_context(_weighted_spectra.eigensolver_context(
        "cusolver_batched" if device.startswith("cuda") and identity["matmul_mode"] == "tf32" else "torch",
        memory_limit=int(identity["analysis_options"]["memory_limit_gib"] * 1024**3)))
    try:
        while True:
            claimed = _claim(connection, plan)
            if claimed is None:
                remaining = connection.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')").fetchone()[0]
                if remaining == 0:
                    break
                time.sleep(1)
                continue
            job_id, job, reserved = claimed
            print(json.dumps(dict(event="started",job_id=job_id,kind=job["kind"],chromosome=job["chromosome"],device=device)),flush=True)
            before = time.perf_counter()
            try:
                entry = next(item for item in plan["chromosomes"] if item["name"] == job["chromosome"])
                metadata_digest = _bind_chromosome(connection,plan,entry)
                current = (job["trait"], job["chromosome"])
                setup = time.perf_counter()
                if key != current:
                    if reader is not None:
                        reader.close()
                    reader, pipeline, key = None, None, None
                    model = load_null_model(plan["phenotypes"][job["trait"]]["path"], device=device, matmul_mode=identity["matmul_mode"])
                    reader = PortableCachedGDS(entry["container_directory"],entry["metadata_directory"],device=device)
                    model.gds_sample_ids = model.sample_ids.copy()
                    analysis = dict(plan["dataset_analysis"])
                    analysis.update(reader.manifest.get("analysis",{}))
                    analysis.update({name:entry[name] for name in ("annotation_catalog","annotation_names","qc_path") if name in entry})
                    pipeline = PheWASPipeline(reader,[model],qc_path=analysis.get("qc_path","annotation/filter"),
                        annotation_catalog=analysis.get("annotation_catalog",{}),annotation_names=analysis.get("annotation_names",[]),
                        options=AnalysisOptions(**identity["analysis_options"]))
                    pipeline.resident_genotypes = device.startswith("cuda")
                    pipeline.local_mask_reuse = True
                    pipeline.weight_batch_optimization = True
                    pipeline.statistics_tail_optimization = True
                    pipeline.single_batch_optimization = True
                    pipeline.individual_effective_block_size = identity["individual_effective_block_size"]
                    pipeline._portable_promoters = json.loads(Path(entry["promoter_intervals"]).read_text()) if "promoter_intervals" in entry else []
                    key = current
                index_report = None
                if job["kind"] in ("noncoding","ncrna"):
                    index_report = _shared_annotation_index(pipeline,entry,plan,metadata_digest,kind=job["kind"])
                setup_seconds = time.perf_counter() - setup
                waiting = []
                def guard(**arguments):
                    waiting.append(_host_memory_guard(connection,plan,job_id,**arguments))
                pipeline.host_memory_guard = guard
                pipeline.profiler = StageProfiler(device,enabled=True)
                pipeline.covariance_diagnostics = []
                pipeline.batch_diagnostics = []
                from .statistics import statistics_execution_metadata
                from .tf32 import execution_metadata as tf32_execution_metadata
                statistics_execution_metadata(reset=True)
                tf32_execution_metadata(reset=True)
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                output = Path(plan["output_directory"]) / "results" / f"trait_{job['trait'] + 1:04d}" / f"chr{job['chromosome']}" / job["kind"]
                output.mkdir(parents=True,exist_ok=True)
                with DenseProductAudit(forced=identity["matmul_mode"] == "tf32") as audit:
                    if job["kind"] == "individual":
                        old = pipeline.options
                        pipeline.options = replace(old, genotype_block_size=8192)
                        try:
                            outputs, rows = _single_outputs(pipeline,job["chromosome"],output / f"job_{job_id:06d}",identity,job["arguments"])
                        finally:
                            pipeline.options = old
                    else:
                        arguments = dict(job["arguments"])
                        if job["kind"] == "noncoding":
                            arguments["promoter_intervals"] = pipeline._portable_promoters
                        result = getattr(pipeline,job["kind"])(chromosome=job["chromosome"],**arguments)
                        _probability_gate(result)
                        path = output / f"job_{job_id:06d}.Rdata"
                        write_association_batch(path,[result],kind=job["kind"],layout="base")
                        outputs = [dict(path=str(path),sha256=_sha256(path),
                            native_validation=_verify_native_output(path,kind=job["kind"],expected_result=result))]
                        from .cli import _association_result_row_count
                        rows = _association_result_row_count(result,kind=job["kind"])
                        del result
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                report = dict(job_id=job_id,device=device,seconds=time.perf_counter()-before,setup_seconds=setup_seconds,
                    association_rows=rows,outputs=outputs,stage_profile=pipeline.profiler.report(),
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.startswith("cuda") else 0,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if device.startswith("cuda") else 0,
                    dense_product_audit=audit.report(),host_reserved_gib=reserved)
                report["statistics_execution_metadata"] = statistics_execution_metadata()
                report["tf32_execution"] = tf32_execution_metadata()
                report["covariance_diagnostics"] = list(pipeline.covariance_diagnostics)
                report["batch_diagnostics"] = list(pipeline.batch_diagnostics)
                report["genotype_reader"] = reader.reader_metadata
                report["metadata_manifest_sha256"] = metadata_digest
                report["scope"] = "selected interval" if job["kind"] == "individual" and job["arguments"].get("start") is not None else "whole chromosome" if job["kind"] == "individual" else "selected gene"
                report["shared_annotation_index"] = index_report
                report["host_memory_wait_seconds"] = sum(waiting)
                report["host_reserved_gib"] = connection.execute("SELECT gib FROM leases WHERE job_id=?",(job_id,)).fetchone()[0]
                if max(report["peak_allocated_mib"],report["peak_reserved_mib"]) > identity["analysis_options"]["memory_limit_gib"] * 1024:
                    raise MemoryError("worker exceeded configured CUDA allocator limit")
                connection.execute("UPDATE jobs SET status='completed',finished=?,report=?,error=NULL WHERE id=?",
                                   (time.time(),json.dumps(report,allow_nan=False),job_id))
                print(json.dumps(dict(event="completed",job_id=job_id,seconds=report["seconds"],rows=rows,device=device)),flush=True)
            except _HostLeaseDeferred as deferred:
                job["host_memory_reserve_gib"] = deferred.required
                connection.execute("UPDATE jobs SET status='pending',payload=?,owner_pid=NULL,owner_identity=NULL,deferrals=deferrals+1 WHERE id=?",
                    (json.dumps(job),job_id))
                if reader is not None:
                    reader.close()
                reader, pipeline, key = None, None, None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                connection.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?", (time.time(),traceback.format_exc(),job_id))
                print(json.dumps(dict(event="failed",job_id=job_id,device=device)),flush=True)
                if reader is not None:
                    reader.close()
                reader, pipeline, key = None, None, None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            finally:
                if pipeline is not None:
                    pipeline._test_set_cache.clear()
                    pipeline._local_union_host_prepared = None
                    pipeline._local_union_device_prepared = None
                gc.collect()
                connection.execute("DELETE FROM leases WHERE job_id=?",(job_id,))
                connection.execute("INSERT OR REPLACE INTO leases VALUES(?,?)",(-os.getpid(),max(1.,_rss_bytes(os.getpid())/1024**3)))
                connection.commit()
    finally:
        if reader is not None:
            reader.close()
        backend.close()
        connection.execute("DELETE FROM leases WHERE job_id=?",(-os.getpid(),))
        connection.execute("DELETE FROM workers WHERE pid=?",(os.getpid(),))
        connection.commit()
        connection.close()


def _exclusive_launcher(function):
    @wraps(function)
    def locked(plan_path, *args, **kwargs):
        lock = Path(plan_path).resolve().parent / "launcher.private.lock"
        identity = _process_identity(os.getpid()) or dict(pid=os.getpid())
        for attempt in range(2):
            try:
                descriptor = os.open(lock,os.O_WRONLY | os.O_CREAT | os.O_EXCL,0o600)
            except FileExistsError:
                previous = json.loads(lock.read_text())
                actual = _process_identity(previous["pid"])
                if actual == previous or (actual is None and _pid_alive(previous["pid"])):
                    raise RuntimeError("a launcher already owns this result directory")
                lock.unlink()
                continue
            with os.fdopen(descriptor,"w") as handle:
                json.dump(identity,handle)
            break
        else:
            raise RuntimeError("could not obtain the result-directory launcher lock")
        try:
            return function(plan_path,*args,**kwargs)
        finally:
            if lock.exists() and json.loads(lock.read_text()) == identity:
                lock.unlink()
    return locked


@_exclusive_launcher
def execute_plan(plan_path, *, devices=None, workers=8, retry_failed=False):
    """Execute a persisted plan on at most eight distinct GPUs, preserving completed outputs."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if _implementation_sha256() != plan["input_identity"].get("implementation_sha256"):
        raise ValueError("implementation changed after preparation; use a new result directory")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 to 8")
    if devices is None:
        count = min(workers,torch.cuda.device_count())
        if not count:
            if plan["input_identity"]["matmul_mode"] != "fp64":
                raise RuntimeError("TF32 production execution requires CUDA")
            devices = ["cpu"]
        else:
            devices = [f"cuda:{index}" for index in range(count)]
    else:
        devices = [str(value) for value in devices]
        if not devices or len(devices) > workers or len(set(devices)) != len(devices):
            raise ValueError("devices must select one to workers distinct devices")
        if plan["input_identity"]["matmul_mode"] == "tf32" and any(not value.startswith("cuda:") for value in devices):
            raise ValueError("TF32 production workers require explicit CUDA devices")
    database = Path(plan["output_directory"]) / "jobs.sqlite"
    with sqlite3.connect(database) as connection:
        for pid, raw in connection.execute("SELECT pid,identity FROM workers").fetchall():
            expected = json.loads(raw)
            actual = _process_identity(pid)
            if actual != expected and (actual is not None or not _pid_alive(pid)):
                connection.execute("DELETE FROM leases WHERE job_id=?",(-pid,))
                connection.execute("DELETE FROM workers WHERE pid=?",(pid,))
        for job_id, pid, raw_identity in connection.execute("SELECT id,owner_pid,owner_identity FROM jobs WHERE status='running'").fetchall():
            expected = json.loads(raw_identity) if raw_identity else None
            actual = _process_identity(pid)
            if (actual is not None and (expected is None or actual == expected)) or (actual is None and _pid_alive(pid)):
                raise RuntimeError("an existing worker still owns the run; do not start a second launcher")
            connection.execute("UPDATE jobs SET status='pending',owner_pid=NULL WHERE id=?",(job_id,))
            connection.execute("DELETE FROM leases WHERE job_id=?",(job_id,))
        if retry_failed:
            connection.execute("UPDATE jobs SET status='pending',owner_pid=NULL,error=NULL WHERE status='failed'")
        for job_id, raw in connection.execute("SELECT id,report FROM jobs WHERE status='completed'").fetchall():
            outputs = json.loads(raw)["outputs"]
            if any(not Path(item["path"]).is_file() or _sha256(item["path"]) != item["sha256"] for item in outputs):
                raise ValueError(f"completed task {job_id} has a missing or altered output; choose a new result directory")
    for model in plan["phenotypes"]:
        if _sha256(model["path"]) != model["sha256"]:
            raise ValueError("persisted null model changed after input preparation")
    before = time.perf_counter()
    context = mp.get_context("spawn")
    processes = [context.Process(target=_run_worker,args=(str(plan_path),device)) for device in devices]
    thread_names = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")
    previous_threads = {name:os.environ.get(name) for name in thread_names}
    try:
        for name in thread_names:
            os.environ[name] = str(plan["input_identity"]["cpu_threads_per_worker"])
        for process in processes:
            process.start()
    finally:
        for name,value in previous_threads.items():
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name] = value
    observed = set()
    while any(process.is_alive() for process in processes):
        for process in processes:
            process.join(timeout=.05)
            if process.exitcode is not None and process.exitcode != 0 and process.pid not in observed:
                observed.add(process.pid)
                with sqlite3.connect(database,timeout=60) as connection:
                    job_ids = [row[0] for row in connection.execute(
                        "SELECT id FROM jobs WHERE status='running' AND owner_pid=?",(process.pid,))]
                    for job_id in job_ids:
                        connection.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?",
                            (time.time(),"worker process exited unexpectedly",job_id))
                        connection.execute("DELETE FROM leases WHERE job_id=?",(job_id,))
                    connection.execute("DELETE FROM leases WHERE job_id=?",(-process.pid,))
                    connection.execute("DELETE FROM workers WHERE pid=?",(process.pid,))
    with sqlite3.connect(database) as connection:
        crashed = [process.pid for process in processes if process.exitcode != 0]
        for pid in crashed:
            connection.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE status='running' AND owner_pid=?",
                               (time.time(),"worker process exited unexpectedly",pid))
        counts = dict(connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        reports = [json.loads(raw) for raw, in connection.execute("SELECT report FROM jobs WHERE status='completed'")]
        cache_counts = dict(ready=sum((Path(entry["metadata_directory"])/"COMPLETE").is_file()
                               for entry in plan["chromosomes"]),total=len(plan["chromosomes"]))
        deferrals = connection.execute("SELECT COALESCE(SUM(deferrals),0) FROM jobs").fetchone()[0]
    report = dict(schema_version=1, counts=counts, task_count=plan["task_count"],devices=devices,
        launcher_seconds=time.perf_counter()-before,worker_process_failures=len(crashed),
        association_rows=sum(item["association_rows"] for item in reports),
        peak_allocated_mib=max((item["peak_allocated_mib"] for item in reports),default=0),
        peak_reserved_mib=max((item["peak_reserved_mib"] for item in reports),default=0),
        implementation_sha256=plan["input_identity"]["implementation_sha256"],
        cache_counts=cache_counts,host_admission_deferrals=deferrals,
        input_counts=[item["alignment"] for item in plan["phenotypes"]],
        null_fit_mode=plan["input_identity"]["null_fit_mode"],association_matmul_mode=plan["input_identity"]["matmul_mode"],
        precision_validation="association execution alone does not establish original-software accuracy",
        timing_contract="setup, CPU wall and CUDA stream stages overlap; do not sum them as end-to-end time",
        output_directory=plan["output_directory"])
    _json_write(Path(plan["output_directory"]) / "report.private.json",report)
    if counts.get("failed",0) or counts.get("pending",0) or counts.get("running",0) or crashed:
        raise RuntimeError(f"run is incomplete: {counts}; see jobs.sqlite and report.private.json")
    return report


def run(phenotype_csv, covariate_csv, cache_directory, *, devices=None, workers=8,
        retry_failed=False, **hyperparameters):
    """Run all cached chromosomes from the two eid CSVs and one portable cache folder.

    Each named phenotype is analysed separately after its complete-case sample
    selection. The default outputs include Single, coding, noncoding and ncRNA.
    Call inside ``if __name__ == '__main__':`` when using spawn workers.
    """
    plan = prepare_run(phenotype_csv,covariate_csv,cache_directory,**hyperparameters)
    return execute_plan(plan,devices=devices,workers=workers,retry_failed=retry_failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Gaussian WGS association from two eid CSVs and a portable cache directory")
    parser.add_argument("phenotype_csv")
    parser.add_argument("covariate_csv")
    parser.add_argument("cache_directory")
    parser.add_argument("--output-directory")
    parser.add_argument("--devices",nargs="+")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--chromosomes",nargs="+")
    parser.add_argument("--analyses",nargs="+",choices=_KINDS,default=list(_KINDS))
    parser.add_argument("--memory-limit-gib",type=float,default=40.)
    parser.add_argument("--transform",choices=("none","rint"),default="none")
    parser.add_argument("--retry-failed",action="store_true")
    args = vars(parser.parse_args(argv))
    print(json.dumps(run(**args),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
