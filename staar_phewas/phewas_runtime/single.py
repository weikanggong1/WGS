"""Independent Single cores with shared pointwise tails and result transfer.

Every block is prepared for its own singleton sample axis and source AF. No
union padding, covariance sharing or matrix precision changes take place here.
"""
from __future__ import annotations

import operator
import numpy as np
import torch

from ..gds_device import DeviceMinorBlock
from ..pipeline import _individual_log_probabilities


_METRICS = dict(calls=0, active_trait_blocks=0, computed_trait_blocks=0,
                pointwise_tail_batches=0, result_transfer_batches=0,
                result_transfer_values=0, maximum_batch_variants=0)


def execution_metadata(*, reset=False):
    """Return aggregate counts without model or variant identifiers."""
    result = dict(_METRICS,
                  genotype_core="independent singleton row/K order",
                  tail_dtype="float64 scalar stability",
                  core_storage_dtype="float32",
                  covariance_shared=False, union_sample_padding=False)
    if reset:
        for key in _METRICS:
            _METRICS[key] = 0
    return result


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def process_single_batches(pipelines, blocks, ordinal_offsets, chromosome,
                           mac_cutoff=20, subset_variants_num=5000):
    """Return ``(records_by_trait, new_ordinals)`` for bounded ready blocks.

    ``pipelines`` contains one base-semantics Gaussian or ordinary binary
    model per entry. ``blocks`` contains singleton ``DeviceMinorBlock`` objects
    on that model's row axis; ``None`` marks a trait whose effective-column
    buffer is not ready. ``ordinal_offsets`` is independent for each trait.
    The caller has already selected the chromosome's QC/type variant axes.

    Each original Score/variance core executes separately and releases its
    floating genotype before the next trait. Only O(sum(M)) core vectors are
    concatenated for the unchanged pointwise log tail, SE, and one host copy.
    Records use the same helper as standalone Single, including original
    source AF and extraction/chunk markers. No phenotype values are imputed.
    """
    pipelines, blocks = list(pipelines), list(blocks)
    ordinal_offsets = list(ordinal_offsets)
    if not pipelines or not (len(pipelines) == len(blocks) == len(ordinal_offsets)):
        raise ValueError("pipelines, blocks and ordinal_offsets require equal nonempty lengths")
    subset_variants_num = _integer(subset_variants_num, "subset_variants_num", positive=True)
    ordinals = [_integer(value, "ordinal_offset") for value in ordinal_offsets]
    if not np.isscalar(mac_cutoff) or not np.isfinite(mac_cutoff) or mac_cutoff < 0:
        raise ValueError("mac_cutoff must be finite and nonnegative")
    if chromosome is None:
        raise ValueError("chromosome is required")

    # Admit the complete batch before updating counters or preparing dosage.
    device = None
    for pipeline, block in zip(pipelines, blocks):
        if len(pipeline.models) != 1 or pipeline.options.wrapper_semantics != "base":
            raise ValueError("Single PheWAS requires singleton base-semantics pipelines")
        model = pipeline.models[0]
        if (model.n_pheno != 1 or model.use_spa
                or getattr(model, "family", "gaussian") not in ("gaussian", "binomial")
                or getattr(model, "matmul_mode", "fp64") != "tf32"
                or not hasattr(model, "individual_score_variance")
                or model.x.dtype != torch.float32):
            raise ValueError("Single PheWAS requires a native FP32 ordinary single-trait core without SPA")
        model_device = torch.device(model.device)
        if device is None:
            device = model_device
        elif device != model_device:
            raise ValueError("Single PheWAS cores must use the same device")
        if block is not None:
            if not isinstance(block, DeviceMinorBlock):
                raise TypeError("ready Single blocks must be DeviceMinorBlock objects")
            if block.dosage.device != model_device:
                raise ValueError("Single block and model must use the same device")
            if (not np.array_equal(block.sample_indices, pipeline.union_rows)
                    or len(pipeline.union_rows) != model.n
                    or not np.array_equal(pipeline.trait_rows[0], np.arange(model.n))):
                raise ValueError("Single block must retain its exact singleton model sample order")

    _METRICS["calls"] += 1
    outputs = [[] for _ in pipelines]
    pending = []
    for index, (pipeline, block) in enumerate(zip(pipelines, blocks)):
        if block is None:
            continue
        _METRICS["active_trait_blocks"] += 1
        state, ordinals[index] = pipeline._prepare_individual_block(
            block, ordinals[index], mac_cutoff=mac_cutoff)
        if state is None:
            continue
        prepared = pipeline._prepare_individual_trait(state, 0, mac_cutoff=mac_cutoff)
        if prepared is None:
            continue
        core = pipeline._compute_individual_core(prepared)
        # Preserve the existing per-model core; never retain all traits' G.
        del prepared["genotype"]
        score, variance = core["score"], core["variance"]
        if (score.dtype != torch.float32 or variance.dtype != torch.float32
                or score.ndim != 1 or variance.shape != score.shape
                or score.numel() != len(prepared["selected"])
                or score.device != device or variance.device != device):
            raise RuntimeError("native Single core must return matching FP32 Score/variance vectors")
        _METRICS["computed_trait_blocks"] += 1
        pending.append((index, pipeline, prepared, score, variance))
    if not pending:
        return outputs, ordinals

    profiler = pending[0][1].profiler
    with profiler.measure("phewas_individual_tail", gpu=True):
        score = torch.cat([item[3] for item in pending])
        variance = torch.cat([item[4] for item in pending])
        standard_error = torch.sqrt(variance.to(torch.float64))
        log_probabilities = _individual_log_probabilities(score, variance)
        values = torch.stack((variance, score, standard_error, log_probabilities), dim=1)
    _METRICS["pointwise_tail_batches"] += 1
    _METRICS["maximum_batch_variants"] = max(_METRICS["maximum_batch_variants"], score.numel())
    with profiler.measure("phewas_result_d2h", gpu=True):
        values_cpu = values.detach().cpu().numpy()
    _METRICS["result_transfer_batches"] += 1
    _METRICS["result_transfer_values"] += values.numel()
    offset = 0
    for index, pipeline, prepared, score, _ in pending:
        stop = offset + score.numel()
        outputs[index] = pipeline._individual_records_from_values(
            prepared, values_cpu[offset:stop], subset_variants_num=subset_variants_num)
        offset = stop
    return outputs, ordinals
