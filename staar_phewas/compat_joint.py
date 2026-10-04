"""Native GMMAT glmmkin.multi serialization from fitted PyTorch state.

SPDX-License-Identifier: GPL-3.0-only
Serialization never starts an R process.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy import sparse
from .compat import _array, _canonical_call
from .r_output import RAttributed, RMatrix, RS4, sparse_matrix, write_r_object


def joint_gaussian_null_r_object(model, *, original_sample_ids=None,
                                 covariate_names=None, phenotype_names=None,
                                 covariate_assign=None, layout="phewas", call=None):
    """Build the actual glmmkin.multi fields, including sparse precision.

    ``multistaar`` retains the original MultiSTAAR null wrapper's fields;
    ``phewas`` appends the two flags used by STAARpipeline's fit_nullmodel.
    Trait-major precision has shape (n*t, n*t) with n*t*t nonzero entries.
    """
    if model.family!="gaussian" or model.n_pheno<2:
        raise ValueError("joint Gaussian output requires a joint Gaussian null")
    if model.phenotype is None or model.fitted_values is None:
        raise ValueError("joint cache lacks phenotype/fitted values; refit a complete cache")
    if layout not in ("phewas","multistaar"):
        raise ValueError("joint output layout must be phewas or multistaar")
    n,t=model.n,model.n_pheno
    ids=np.asarray(model.sample_ids if original_sample_ids is None else original_sample_ids,dtype=str)
    if ids.shape!=(n,) or len(set(ids))!=n:
        raise ValueError("original sample IDs must be unique and aligned")
    x=_array(model.covariates);p=x.shape[1]
    if covariate_names is None:
        covariate_names=["(Intercept)"] if p==1 and np.all(x[:,0]==1) else [f"V{j+1}" for j in range(p)]
    if phenotype_names is None:
        phenotype_names=model.phenotype_names or [f"trait{j+1}" for j in range(t)]
    if len(covariate_names)!=p or len(phenotype_names)!=t:
        raise ValueError("covariate and phenotype names must match model dimensions")
    row_names=np.arange(1,n+1).astype(str)
    expanded_rows=np.tile(row_names,t)
    fixed_names=[f"{trait}:{term}" for trait in phenotype_names for term in covariate_names]
    assign=np.arange(p,dtype=np.int32) if covariate_assign is None else np.asarray(covariate_assign,dtype=np.int32)
    if assign.shape!=(p,):raise ValueError("covariate_assign must match the design columns")
    precision=_array(model.precision)
    samples=np.tile(np.arange(n),t*t)
    a=np.repeat(np.arange(t),t*n);b=np.tile(np.repeat(np.arange(t),n),t)
    values=precision[samples,a,b]
    if model.relatedness:
        sigma_i=sparse_matrix(sparse.coo_matrix((values,(a*n+samples,b*n+samples)),shape=(n*t,n*t)),
            class_name="dsCMatrix",dimnames=[expanded_rows,expanded_rows])
    else:
        # Matrix 1.2's ordinary Kronecker precision is a triplet general matrix.
        sigma_i=RS4("dgTMatrix",{"i":(a*n+samples).astype(np.int32),"j":(b*n+samples).astype(np.int32),
            "Dim":np.asarray([n*t,n*t],dtype=np.int32),"Dimnames":[expanded_rows,expanded_rows],
            "x":values,"factors":[]})
    sx=torch.einsum("nab,np->anbp",model.precision,model.covariates).reshape(n*t,t*p)
    sigma_ix=sparse_matrix(_array(sx),dimnames=[expanded_rows,None])
    theta_values=[]
    for covariance in _array(model.theta):
        theta_values.append(sparse_matrix(covariance,class_name="dsCMatrix") if model.relatedness else
                            RMatrix(covariance,phenotype_names,phenotype_names))
    theta=theta_values if not model.relatedness else RAttributed(theta_values,{"names":np.asarray(["residuals"]+[f"kins{j}" for j in range(1,len(theta_values))],dtype=str)})
    y=_array(model.phenotype);fitted=_array(model.fitted_values)
    fields={"theta":theta,"n.pheno":np.asarray([t],dtype=np.int32),"n.groups":np.asarray([1.]),
        "coefficients":RMatrix(_array(model.coefficients),phenotype_names,covariate_names),
        "linear.predictors":RMatrix(fitted,phenotype_names,row_names),
        "fitted.values":RMatrix(fitted,phenotype_names,row_names),
        "Y":RMatrix(y,phenotype_names,None),
        "X":RAttributed(RMatrix(x,covariate_names,row_names),{"assign":assign}),"P":None,
        "residuals":RMatrix(y-fitted,phenotype_names,row_names),
        "scaled.residuals":RMatrix(_array(model.scaled_residuals),phenotype_names,row_names),
        "cov":RMatrix(_array(model.fixed_effect_covariance),fixed_names,fixed_names),
        "Sigma_i":sigma_i,"Sigma_iX":sigma_ix,"converged":bool(model.converged),
        "call":_canonical_call() if call is None else call,"id_include":ids,"sparse_kins":True}
    if layout=="phewas":fields.update(relatedness=model.relatedness,use_SPA=False)
    return RAttributed(list(fields.values()),{"names":np.asarray(list(fields),dtype=str),"class":"glmmkin.multi"},object_flag=True)


def write_joint_gaussian_null(path,model,**kwargs):
    """Write obj_nullmodel as native Rdata/RDS, independently of R runtime."""
    write_r_object(path,joint_gaussian_null_r_object(model,**kwargs),object_name="obj_nullmodel")
