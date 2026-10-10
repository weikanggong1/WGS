"""Stable native Single projection for a fixed diagonal precision state.

SPDX-License-Identifier: GPL-3.0-only
"""
import torch


def centered_projection_state(model, precision, precision_x, *, product, blocked=False):
    """Return sym(P)1 and its sum; no version counter means no cache.

    Centering removes the large intercept component before native TF32
    projection. The finite-state correction retains the same quadratic form
    when the imported state does not annihilate constants exactly. The caller
    supplies its own product function and handles Gaussian/SPA/precision modes.
    """
    fields = (model.x, precision_x, precision,
              model.fixed_effect_covariance, model.scaled_residuals)
    if (model.n_pheno != 1 or blocked or precision is None or precision_x is None
            or model.x.ndim != 2 or model.x.shape[0] != model.n
            or model.x.shape[1] == 0 or precision_x.shape != model.x.shape
            or precision.shape != (model.n,)
            or model.fixed_effect_covariance.shape != (model.x.shape[1],) * 2
            or model.scaled_residuals.shape != (model.n,)
            or any(value.dtype != torch.float32 or value.requires_grad
                   or value.layout != torch.strided or value.device != model.device
                   for value in fields)):
        model._centered_single_projection_cache = None
        return None
    try:
        stamp = tuple((id(value), value._version, tuple(value.shape),
                       value.stride(), value.storage_offset(), value.data_ptr(),
                       value.dtype, value.device) for value in fields)
    except RuntimeError:
        # torch.inference_mode() tensors cannot safely use versioned caches.
        stamp = None
    cached = getattr(model, "_centered_single_projection_cache", None)
    if stamp is not None and cached is not None and cached[0] == stamp:
        return cached[1]
    result = None
    if bool((model.x[:, 0] == 1).all()):
        # The quadratic form uses sym(P), including asymmetric import roundoff.
        covariance = model.fixed_effect_covariance
        symmetric_covariance = covariance * 0.5 + covariance.T * 0.5
        coefficient = product(symmetric_covariance, precision_x.sum(dim=0), mode="tf32")
        p_one = precision - product(precision_x, coefficient, mode="tf32")
        result = (p_one, p_one.sum())
    model._centered_single_projection_cache = (
        None if stamp is None else (stamp, result, fields))
    return result
