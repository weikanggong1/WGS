"""Portable model files; no pickle or embedded deployment configuration."""
from pathlib import Path
import numpy as np
import torch
from .null_model import GaussianNullModel, KinshipSpectrum, fit_gaussian_null, rank_inverse_normal
from .multi import JointGaussianNullModel, fit_joint_gaussian_null
from .rint import rank_inverse_normal_tensor
from .binary_null import BinaryNullModel, fit_logistic_null, binary_prefitted_state
from .identity import sample_keys, validate_sample_pairs


def _sample_identity(model):
    pairs = getattr(model, "sample_pairs", None)
    if pairs is None:
        return {}
    pairs = validate_sample_pairs(pairs, where="model sample pairs")
    if not np.array_equal(sample_keys(pairs), np.asarray(model.sample_ids, dtype=str)):
        raise ValueError("model FID/IID pairs disagree with its ordered sample keys")
    return {"sample_pairs": pairs, "sample_identifier_format": "fid_iid_json"}


def _restore_sample_identity(model, values):
    if "sample_pairs" in values:
        pairs = validate_sample_pairs(values["sample_pairs"], where="stored model sample pairs")
        if not np.array_equal(sample_keys(pairs), np.asarray(model.sample_ids, dtype=str)):
            raise ValueError("stored model FID/IID pairs disagree with its sample keys")
        model.sample_pairs = pairs
    return model


def save_null_model(model: GaussianNullModel, path):
    """Write the complete fitted state to a private NPZ, including subject IDs."""
    def array(t):
        return t.detach().cpu().numpy()
    if isinstance(model, (JointGaussianNullModel, BinaryNullModel)):
        kind = "gaussian_joint" if isinstance(model, JointGaussianNullModel) else "binary_state"
        fields = ("covariates", "scaled_residuals", "coefficients", "theta", "working_theta",
                  "fixed_effect_covariance", "precision", "phenotype", "fitted_values") if kind=="gaussian_joint" else (
                  "x", "scaled_residuals", "fitted_probability", "xw", "projection_left",
                  "fixed_effect_covariance", "precision", "precision_x", "coefficients", "phenotype",
                  "working_phenotype")
        values = {"model_kind": kind, "sample_ids": np.asarray(model.sample_ids,dtype=str),
                  "iterations": model.iterations, "converged": model.converged,
                  "fit_method": model.fit_method, "use_spa": model.use_spa}
        for field in fields:
            value = getattr(model,field)
            if value is None: continue
            if value.layout != torch.strided:
                coo=value.to_sparse_coo().coalesce()
                values[field+"_indices"]=array(coo.indices());values[field+"_values"]=array(coo.values())
                values[field+"_shape"]=np.asarray(coo.shape,dtype=np.int64)
            else:values[field]=array(value)
        if kind=="gaussian_joint":
            values.update(relatedness=model.relatedness,residual_covariance_singular=model.residual_covariance_singular)
            if model.phenotype_names is not None:values["phenotype_names"]=np.asarray(model.phenotype_names,dtype=str)
        else:
            values["has_kinship"]=model.has_kinship
            values["matmul_mode"]=model.matmul_mode
            values["source_matmul_mode"]=getattr(model,"source_matmul_mode",model.matmul_mode)
        if getattr(model,"genotype_sample_ids",None) is not None:values["genotype_sample_ids"]=np.asarray(model.genotype_sample_ids,dtype=str)
        values.update(_sample_identity(model))
        np.savez_compressed(path,**values)
        return
    values = {name: array(getattr(model, name)) for name in (
        "x", "scaled_residuals", "coefficients", "theta", "precision_theta",
        "fixed_effect_covariance", "inverse_variance", "precision_x")}
    for name in ("phenotype", "fitted_values", "working_phenotype"):
        if getattr(model, name) is not None:
            values[name] = array(getattr(model, name))
    values["has_kinship"] = model.has_kinship
    values["model_kind"] = "gaussian_single"
    values["matmul_mode"] = getattr(model, "matmul_mode", "fp64")
    if getattr(model,"genotype_sample_ids",None) is not None:
        values["genotype_sample_ids"] = np.asarray(model.genotype_sample_ids,dtype=str)
    values.update(sample_ids=model.sample_ids.astype(str), eigenvalues=array(model.spectrum.eigenvalues),
                  number_blocks=len(model.spectrum.blocks), iterations=model.iterations,
                  converged=model.converged)
    for index, (rows, rotation) in enumerate(model.spectrum.blocks):
        values[f"block_{index}_rows"] = array(rows)
        values[f"block_{index}_rotation"] = array(rotation)
    values.update(_sample_identity(model))
    np.savez_compressed(path, **values)


def load_null_model(path, *, device="cpu", matmul_mode=None):
    """Load fitted state; optionally convert single-trait storage to native FP32.

    Without an override, historical FP64 caches retain their stored mode.
    """
    with np.load(path, allow_pickle=False) as values:
        stored_mode = str(values["matmul_mode"]) if "matmul_mode" in values else "fp64"
        selected_mode = stored_mode if matmul_mode is None else matmul_mode
        # Removed reconstruction caches are only reusable via an explicit mode.
        from .tf32 import validate_mode
        kind=str(values["model_kind"]) if "model_kind" in values else "gaussian_single"
        if kind in ("gaussian_single", "binary_state"):
            selected_mode = validate_mode(selected_mode)
        if kind == "binary_state" and selected_mode == "tf32" and bool(values["use_spa"]):
            raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
        storage_dtype = torch.float32 if selected_mode == "tf32" else torch.float64
        def tensor(name, dtype=None):
            dtype = storage_dtype if dtype is None else dtype
            return torch.as_tensor(values[name], dtype=dtype, device=device)
        kind=str(values["model_kind"]) if "model_kind" in values else "gaussian_single"
        if kind in ("gaussian_joint","binary_state"):
            def optional(name):
                if name in values:return tensor(name)
                if name+"_indices" in values:
                    return torch.sparse_coo_tensor(tensor(name+"_indices",torch.int64),tensor(name+"_values"),
                                                  tuple(values[name+"_shape"]),device=device).coalesce()
                return None
            if kind=="gaussian_joint":
                model=JointGaussianNullModel(tuple(values["sample_ids"].astype(str)),
                    *(tensor(name) for name in ("covariates","scaled_residuals","coefficients","theta","working_theta","fixed_effect_covariance","precision")),
                    iterations=int(values["iterations"]),converged=bool(values["converged"]),
                    relatedness=bool(values["relatedness"]),fit_method=str(values["fit_method"]),
                    residual_covariance_singular=bool(values["residual_covariance_singular"]),
                    phenotype=optional("phenotype"),fitted_values=optional("fitted_values"),
                    phenotype_names=tuple(values["phenotype_names"].astype(str)) if "phenotype_names" in values else None)
            else:
                model=binary_prefitted_state(sample_ids=values["sample_ids"],covariates=tensor("x"),residual=tensor("scaled_residuals"),
                    fitted_probability=tensor("fitted_probability"),xw=tensor("xw"),projection_left=tensor("projection_left"),
                    fixed_effect_covariance=tensor("fixed_effect_covariance"),precision=optional("precision"),
                    precision_covariates=optional("precision_x"),coefficients=optional("coefficients"),phenotype=optional("phenotype"),
                    has_kinship=bool(values["has_kinship"]),use_spa=bool(values["use_spa"]),provenance=str(values["fit_method"]),device=device,
                    matmul_mode=selected_mode,working_phenotype=optional("working_phenotype"))
                model.iterations=int(values["iterations"]);model.converged=bool(values["converged"])
                model.source_matmul_mode = str(values["source_matmul_mode"]) if "source_matmul_mode" in values else stored_mode
            if "genotype_sample_ids" in values:model.genotype_sample_ids=values["genotype_sample_ids"].astype(str)
            return _restore_sample_identity(model, values)
        if kind!="gaussian_single":raise ValueError("unknown fitted model cache kind")
        spectrum = KinshipSpectrum(tensor("eigenvalues"), [
            (tensor(f"block_{i}_rows", torch.int64), tensor(f"block_{i}_rotation"))
            for i in range(int(values["number_blocks"]))])
        model=GaussianNullModel(values["sample_ids"].astype(str),
            *(tensor(name) for name in ("x", "scaled_residuals", "coefficients", "theta", "precision_theta", "fixed_effect_covariance")),
            spectrum, tensor("inverse_variance"), tensor("precision_x"), int(values["iterations"]),
            bool(values["converged"]) if "converged" in values else True,
            phenotype=tensor("phenotype") if "phenotype" in values else None,
            has_kinship=bool(values["has_kinship"]) if "has_kinship" in values else True,
            fitted_values=tensor("fitted_values") if "fitted_values" in values else None,
            working_phenotype=tensor("working_phenotype") if "working_phenotype" in values else None,
            matmul_mode=selected_mode)
        model.source_matmul_mode = stored_mode
        if selected_mode == "tf32" and "phenotype" in values and "fitted_values" in values:
            # Preserve the saved residual subtraction without refitting.
            # Association tensors remain FP32.
            model.native_cached_residuals = np.subtract(values["phenotype"], values["fitted_values"]).astype(np.float64)
            model.native_cached_residuals.flags.writeable = False
        if "genotype_sample_ids" in values:model.genotype_sample_ids=values["genotype_sample_ids"].astype(str)
        return _restore_sample_identity(model, values)


def fit_prepared_input(path, *, device="cpu", transform="none", joint_mode=None,
                       family="gaussian", binary_mode=None, **kwargs):
    """Fit an aligned NPZ with y_raw, ids, optional covariates and sparse GRM.

    Optional genotype sample_indices are returned for the analysis configuration.
    Missing rows must have been removed before this preparation step.
    """
    if transform not in ("none", "rint"):
        raise ValueError("transform must be none or rint")
    with np.load(path, allow_pickle=False) as a:
        y = a["y_raw"]
        if family not in ("gaussian","binomial","binary"):
            raise ValueError("family must be gaussian or binomial")
        if transform == "rint" and family!="gaussian":
            raise ValueError("RINT is defined here for Gaussian phenotypes")
        if transform == "rint":
            y = rank_inverse_normal_tensor(y, device=device)
        options = {}
        for name, parameter in (("covariates", "covariates"), ("grm_diagonal", "kinship_diagonal"),
                                ("grm_edge_row", "edge_rows"), ("grm_edge_col", "edge_cols"), ("grm_edge_value", "edge_values")):
            if name in a:
                options[parameter] = a[name]
        if y.ndim==2 and y.shape[1]>1:
            if family!="gaussian" or joint_mode not in ("ordinary","strict","robust"):
                raise ValueError("joint Gaussian input requires explicit joint_mode: ordinary, strict, or robust")
            if joint_mode=="ordinary":
                options={key:value for key,value in options.items() if key=="covariates"}
            elif "kinship_diagonal" not in options:
                raise ValueError("joint mixed mode requires an explicitly supplied GRM diagonal")
            model=fit_joint_gaussian_null(y,sample_ids=a["ids"],device=device,robust=joint_mode=="robust",**options,**kwargs)
            if "phenotype_names" in a:model.phenotype_names=tuple(a["phenotype_names"].astype(str))
        elif family in ("binomial","binary"):
            if y.ndim==2 and y.shape[1]==1:y=y[:,0]
            if binary_mode!="ordinary":
                raise ValueError("native logistic fitting requires explicit binary_mode='ordinary'; mixed binary null fitting is unavailable, use an explicit prefitted state")
            model=fit_logistic_null(y,sample_ids=a["ids"],device=device,
                                    covariates=options.get("covariates"),**kwargs)
        else:
            if joint_mode is not None:raise ValueError("joint_mode requires multiple phenotype columns")
            if y.ndim==2 and y.shape[1]==1:y=y[:,0]
            model = fit_gaussian_null(y, sample_ids=a["ids"], device=device, **options, **kwargs)
        if "genotype_sample_ids" in a:model.genotype_sample_ids=a["genotype_sample_ids"].astype(str)
        _restore_sample_identity(model, a)
        rows = a["sample_indices"].copy() if "sample_indices" in a else None
    return model, rows
