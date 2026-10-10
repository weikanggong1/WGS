"""Configuration helpers for Python association pipelines."""
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from .pipeline import PheWASPipeline
from .masks import NONCODING_CATEGORIES

def _bounded_pipeline_type():
    """Keep cooperative constructor hooks when adding resident bounds.

    Cache-backed entry points temporarily replace ``PheWASPipeline`` with a
    subclass whose constructor restores and validates annotation indexes.
    The bounded subclass must retain that constructor in its MRO.
    """
    from .phewas_runtime.mask_limit import LimitedMaskPipeline
    from .pipeline import PheWASPipeline as BasePipeline
    current = PheWASPipeline
    if not isinstance(current, type) or not issubclass(current, BasePipeline):
        raise TypeError("bounded pipeline wrappers must inherit PheWASPipeline")
    if issubclass(current, LimitedMaskPipeline):
        return current
    if issubclass(LimitedMaskPipeline, current):
        return LimitedMaskPipeline

    class BoundedPipeline(LimitedMaskPipeline, current):
        pass

    return BoundedPipeline


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _association_result_row_count(result, *, kind):
    """Count computed trait rows, independently of their P values.

    Gene categories contain trait row lists; direct gene/Single outputs contain
    trait row lists without a category layer. Empty masks contribute no CSV rows.
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
    """Read prepared JSON intervals or an independent original TSV."""
    import json
    text = Path(filename).read_text(encoding="utf-8")
    intervals = []
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [line.split()[:3] for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if rows and rows[0][1:] == ["start", "end"]:
            rows = rows[1:]
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("promoter intervals require chromosome, start, end")
        chromosome, start, end = row
        if (isinstance(start, bool) or isinstance(end, bool)
                or not str(start).isdecimal() or not str(end).isdecimal()
                or int(start) < 1 or int(start) > int(end)):
            raise ValueError("promoter intervals require positive inclusive integer coordinates")
        intervals.append((str(chromosome), int(start), int(end)))
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


def _weighted_eigensolver_settings(config,mode,device):
    requested=config.get('weighted_eigensolver','auto')
    if not isinstance(requested,str) or requested not in ('auto','torch','cusolver_batched'):
        raise ValueError('weighted_eigensolver must be auto, torch, or cusolver_batched')
    weight_batch=config.get('weight_batch_optimization',True)
    if not isinstance(weight_batch,bool):
        raise ValueError('weight_batch_optimization must be a JSON boolean')
    eligible=mode=='tf32' and weight_batch and str(device).startswith('cuda')
    effective='cusolver_batched' if eligible and requested!='torch' else 'torch'
    reason=(None if effective=='cusolver_batched' else 'explicit_torch' if requested=='torch'
            else 'matmul_mode_not_tf32' if mode!='tf32' else 'weight_batch_optimization_disabled'
            if not weight_batch else 'device_not_cuda')
    return dict(requested=requested,effective=effective,eligible=eligible,inactive_reason=reason)


def _bind_genotype_samples(genotype, model, prepared_indices=None, rule="exact"):
    """Align complete FID/IID keys exactly, retaining cached logical sample order."""
    if rule != "exact":
        raise ValueError("sample_id_rule must be exact; family and individual IDs are never truncated")
    identifiers = np.asarray(model.sample_ids, dtype=str)
    canonical = getattr(model, "genotype_sample_ids", None)
    if canonical is not None and not np.array_equal(np.asarray(canonical, dtype=str), identifiers):
        raise ValueError("model canonical genotype identifiers differ from its sample keys")
    if identifiers.shape != (model.n,) or len(np.unique(identifiers)) != model.n:
        raise ValueError("model sample keys must be unique and aligned")
    available = np.asarray(genotype.sample_ids(), dtype=str)
    if prepared_indices is None:
        indices = np.asarray(genotype.sample_indices(identifiers), dtype=np.int64)
    else:
        raw = np.asarray(prepared_indices)
        if raw.shape != (model.n,) or raw.dtype.kind not in "iu":
            raise ValueError("prepared sample indices must be an aligned integer vector")
        indices = raw.astype(np.int64, copy=False)
        if len(np.unique(indices)) != model.n or np.any(indices < 0) or np.any(indices >= len(available)):
            raise ValueError("prepared sample indices are nonunique or out of range")
    if not np.array_equal(available[indices], identifiers):
        raise ValueError("genotype sample rows do not match complete model FID/IID keys")
    model.genotype_sample_ids = identifiers.copy()
    return indices


def run_configuration(config, *, cache_specs, device="cuda:0", cpu_threads=2, **options):
    """Run one configuration with the same shared runtime used for many traits."""
    from .phewas_runtime.runtime import run_configuration as run_shared
    return run_shared([config], cache_specs=cache_specs, device=device, cpu_threads=cpu_threads, **options)


def main(argv=None):
    """Invoke the public preparation/analysis command line interface."""
    from .run import main as public_main
    return public_main(argv)
