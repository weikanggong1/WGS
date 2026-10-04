"""STAAR/GMMAT native R output objects from fitted PyTorch state.

SPDX-License-Identifier: GPL-3.0-only
This module performs serialization only; it never launches R.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy import sparse
from .r_output import RAttributed, RMatrix, RS4, RCall, RSymbol, sparse_matrix, dense_s4_matrix, write_r_object


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _named(values, names):
    return RAttributed(_array(values), {"names": np.asarray(names, dtype=str)})


def _canonical_call():
    # Same match.call expression structure as fit_nullmodel's glmmkin wrapper.
    names = ("fixed", "data", "kins", "id", "random.slope", "groups", "family", "method",
             "method.optim", "maxiter", "tol", "taumin", "taumax", "tauregion", "verbose")
    return RCall("glmmkin", [(name, RSymbol(name)) for name in names])


def gaussian_null_r_object(model, *, original_sample_ids=None, covariate_names=None, covariate_assign=None, call=None):
    """Build the frozen GMMAT 1.3.2 glmmkin object, retaining sparse Matrix slots.

    Original GDS IDs are required when alignment used externally normalized
    identifiers. This is the formal obj_nullmodel object, not the NPZ cache.
    """
    if model.n_pheno != 1 or model.family != "gaussian":
        raise ValueError("gaussian_null_r_object requires a single Gaussian null model")
    if model.phenotype is None:
        raise ValueError("model has no stored phenotype; refit or load a complete model cache")
    n = model.n
    ids = np.asarray(model.sample_ids if original_sample_ids is None else original_sample_ids, dtype=str)
    if ids.shape != (n,) or len(set(ids)) != n:
        raise ValueError("original_sample_ids must be unique and aligned")
    x = _array(model.x)
    if covariate_names is None:
        covariate_names = ["(Intercept)"] if x.shape[1] == 1 and np.all(x[:, 0] == 1) else [f"V{i+1}" for i in range(x.shape[1])]
    if len(covariate_names) != x.shape[1]:
        raise ValueError("covariate names do not match the design matrix")
    assign = np.arange(x.shape[1], dtype=np.int32) if covariate_assign is None else np.asarray(covariate_assign, dtype=np.int32)
    if assign.shape != (x.shape[1],):
        raise ValueError("covariate_assign must give an R model.matrix term number per column")
    row_names = np.arange(1, n+1).astype(str)
    y = _array(model.phenotype)
    residual = _array(model.scaled_residuals) * float(model.theta[0])
    fitted = y - residual if model.fitted_values is None else _array(model.fitted_values)
    residual = y - fitted
    theta_names = ["dispersion", "kins1"] if model.has_kinship else ["dispersion"]
    theta = _array(model.theta)[:len(theta_names)]
    inverse = _array(model.inverse_variance)
    if model.has_kinship:
        coupled = np.zeros(n, dtype=bool)
        rows, columns, values = [], [], []
        for idx, rotation in model.spectrum.blocks:
            physical = (rotation * model.inverse_variance[idx][None, :]) @ rotation.T
            indices = _array(idx).astype(np.int64)
            coupled[indices] = True
            block = _array(physical)
            r, c = np.nonzero(np.triu(block))
            rows.extend(indices[r]); columns.extend(indices[c]); values.extend(block[r, c])
        singleton = np.flatnonzero(~coupled)
        rows.extend(singleton); columns.extend(singleton); values.extend(inverse[singleton])
        matrix = sparse.coo_matrix((values, (rows, columns)), shape=(n, n))
        sigma_i = sparse_matrix(matrix, class_name="dsCMatrix", dimnames=[row_names, row_names])
        sigma_ix = dense_s4_matrix(_array(model.precision_x), dimnames=[row_names, covariate_names])
    else:
        sigma_i = RS4("ddiMatrix", {"diag": "N", "Dim": np.asarray([n, n], dtype=np.int32),
            "Dimnames": [row_names, row_names], "x": inverse})
        sigma_ix = RMatrix(_array(model.precision_x), covariate_names, row_names)
    fields = {
        "theta": _named(theta, theta_names), "n.pheno": np.asarray([1.], dtype=np.float64),
        "n.groups": np.asarray([1], dtype=np.int32),
        "coefficients": _named(model.coefficients, covariate_names),
        "linear.predictors": fitted, "fitted.values": fitted,
        "Y": _named(y if model.working_phenotype is None else model.working_phenotype, row_names),
        "X": RAttributed(RMatrix(x, covariate_names, row_names), {"assign": assign}),
        "P": None, "residuals": _named(residual, row_names),
        "scaled.residuals": _named(model.scaled_residuals, row_names),
        "cov": RMatrix(_array(model.fixed_effect_covariance), covariate_names, covariate_names),
        "Sigma_i": sigma_i, "Sigma_iX": sigma_ix,
        "converged": bool(model.converged), "call": _canonical_call() if call is None else call,
        "id_include": ids, "sparse_kins": True, "relatedness": True, "use_SPA": False,
    }
    return RAttributed(fields, {"names": np.asarray(list(fields), dtype=str), "class": "glmmkin"}, object_flag=True)


def write_gaussian_null(path, model, *, original_sample_ids=None, covariate_names=None, covariate_assign=None, call=None):
    """Write obj_nullmodel.Rdata (or RDS) with the original object name."""
    value = gaussian_null_r_object(model, original_sample_ids=original_sample_ids,
                                  covariate_names=covariate_names, covariate_assign=covariate_assign, call=call)
    write_r_object(path, value, object_name="obj_nullmodel")
