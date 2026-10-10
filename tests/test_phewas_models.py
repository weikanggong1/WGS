"""Independent model/projection contracts; these are not population benchmarks."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary_null import _BlockKinship, _BlockPrecision, fit_logistic_mixed_null
from fudan_wgs_toolkit.phewas_models import (
    SparseKinshipData, continuous_paper_transform, infer_family, prepare_phewas_model)
from fudan_wgs_toolkit.rint import rank_inverse_normal_tensor
from fudan_wgs_toolkit.io import save_null_model, load_null_model


def design(n=64):
    return np.column_stack((np.ones(n), np.linspace(-1, 1, n)))


def grm(n=64):
    rows = np.arange(0, n, 2, dtype=np.int64)
    return SparseKinshipData(np.arange(n).astype(str), np.ones(n), rows, rows+1,
                            np.full(len(rows), .2))


def test_block_precision_matches_dense_matrix_and_trace():
    k = grm(8)
    kinship = _BlockKinship(k.diagonal, k.edge_rows, k.edge_cols, k.edge_values,
                           device="cpu", max_block_size=2)
    weights = torch.linspace(.07, .23, 8, dtype=torch.float64)
    precision = _BlockPrecision(kinship, weights, .43)
    dense_k = torch.eye(8, dtype=torch.float64)
    dense_k[k.edge_rows.copy(), k.edge_cols.copy()] = .2
    dense_k[k.edge_cols.copy(), k.edge_rows.copy()] = .2
    expected = torch.linalg.inv(torch.diag(1/weights) + .43*dense_k)
    z = torch.arange(24, dtype=torch.float64).reshape(8,3)/7
    torch.testing.assert_close(precision.apply(z), expected @ z, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(precision.trace_kinship(), torch.trace(expected @ dense_k), atol=1e-14, rtol=1e-14)
    sparse = precision.sparse_tensor()
    assert sparse.layout == torch.sparse_coo and sparse._nnz() == 16
    torch.testing.assert_close(sparse.to_dense(), expected, atol=1e-14, rtol=1e-14)


def test_grm_subset_retains_requested_order_and_exact_edges(tmp_path):
    k = grm(8)
    selected = k.subset(["5", "4", "0", "7"])
    np.testing.assert_array_equal(selected.sample_ids, ["5", "4", "0", "7"])
    np.testing.assert_array_equal(selected.edge_rows, [1])
    np.testing.assert_array_equal(selected.edge_cols, [0])
    np.testing.assert_array_equal(selected.edge_values, [.2])
    path = tmp_path / "grm.npz"
    np.savez(path, **{name:getattr(k,name) for name in (
        "sample_ids", "diagonal", "edge_rows", "edge_cols", "edge_values")})
    loaded = SparseKinshipData.load_npz(path)
    np.testing.assert_array_equal(loaded.diagonal, k.diagonal)
    with pytest.raises(ValueError, match="every analysis sample"):
        k.subset(["0", "999"])


def test_grm_rejects_nonintegral_rows_and_duplicate_ids():
    with pytest.raises(ValueError, match="integer"):
        SparseKinshipData(np.array(["a","b"]),np.ones(2),np.array([.2]),np.array([1]),np.array([.1]))
    with pytest.raises(ValueError, match="unique"):
        SparseKinshipData(np.array(["a","a"]),np.ones(2),np.array([],dtype=int),np.array([],dtype=int),np.array([]))


def test_paper_transform_residualizes_before_rint_and_uses_original_sd():
    x = design(10)
    y = np.array([5., 1., 7., 4., 9., 2., 8., 11., 3., 6.]) + 20*x[:,1]
    actual, metadata = continuous_paper_transform(y, x)
    residual = y-x @ np.linalg.lstsq(x,y,rcond=None)[0]
    expected = rank_inverse_normal_tensor(residual) * np.std(y,ddof=1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert metadata["rint_offset"] == .375 and metadata["sd_correction"] == 1
    assert not torch.allclose(actual,rank_inverse_normal_tensor(y)*np.std(y,ddof=1))


def test_per_trait_missingness_does_not_form_global_intersection():
    x=design(16); ids=np.arange(16).astype(str)
    y=np.sin(np.arange(16))
    a=y.copy();a[[1,3]]=np.nan
    b=y.copy();b[[2,4,6]]=np.nan
    left=prepare_phewas_model(a,ids,x,association_mode="fp64",continuous_transform="none")
    right=prepare_phewas_model(b,ids,x,association_mode="fp64",continuous_transform="none")
    np.testing.assert_array_equal(left.sample_indices,np.setdiff1d(np.arange(16),[1,3]))
    np.testing.assert_array_equal(right.sample_indices,np.setdiff1d(np.arange(16),[2,4,6]))
    assert left.model.n == 14 and right.model.n == 13
    assert left.metadata["shared_complete_case_required"] is False


def test_binary_preparation_keeps_exact_fp64_spa_sidecar():
    n=64;x=design(n);y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],8)
    output=prepare_phewas_model(y,np.arange(n).astype(str),x,association_mode="tf32")
    assert output.model.family == "binomial" and not output.model.use_spa
    assert output.model.x.dtype == torch.float32
    assert output.spa_model.use_spa and output.spa_model.x.dtype == torch.float64
    assert output.metadata["spa_p_filter"] == .05
    torch.testing.assert_close(output.model.scaled_residuals,output.spa_model.scaled_residuals.float(),atol=0,rtol=0)
    assert output.model.x.data_ptr()!=output.spa_model.x.data_ptr()


def test_mixed_binary_boundary_is_explicit_and_sparse_roundtrips(tmp_path):
    n=64;x=design(n);y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],8);k=grm(n)
    m=fit_logistic_mixed_null(y,x,sample_ids=k.sample_ids,device="cpu",**k.fit_parameters())
    assert m.converged and m.has_kinship and m.boundary_refit
    assert m.theta.tolist() == [1.,0.]
    assert m.precision.layout == torch.sparse_coo and m.precision._nnz() == 2*n
    assert m.working_phenotype.shape == y.shape and m.iterations > 1
    path=tmp_path/"model.npz";save_null_model(m,path);loaded=load_null_model(path)
    assert loaded.has_kinship and loaded.use_spa
    torch.testing.assert_close(loaded.precision.to_dense(),m.precision.to_dense(),atol=0,rtol=0)
    torch.testing.assert_close(loaded.scaled_residuals,m.scaled_residuals,atol=0,rtol=0)


def test_mixed_nonconvergence_and_oversized_family_do_not_fallback():
    n=64;x=design(n);y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],8);k=grm(n)
    with pytest.raises(ArithmeticError,match="converge"):
        fit_logistic_mixed_null(y,x,device="cpu",maxiter=2,**k.fit_parameters())
    with pytest.raises(ValueError,match="block exceeds"):
        fit_logistic_mixed_null(y,x,device="cpu",max_block_size=1,**k.fit_parameters())


def test_mixed_ai_iteration_matches_independent_dense_equations():
    n=64;x=design(n);y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],8);k=grm(n)
    trace=[]
    with pytest.raises(ArithmeticError,match="converge"):
        fit_logistic_mixed_null(y,x,device="cpu",maxiter=2,
            trace_callback=trace.append,**k.fit_parameters())
    first=trace[0]
    assert float(first["tau_old"])>0
    xd=torch.as_tensor(x,dtype=torch.float64)
    kd=torch.eye(n,dtype=torch.float64)
    kd[k.edge_rows.copy(),k.edge_cols.copy()]=.2
    kd[k.edge_cols.copy(),k.edge_rows.copy()]=.2
    inverse=torch.linalg.inv(torch.diag(1/first["weights_old"])+first["tau_old"]*kd)
    covariance=torch.linalg.inv(xd.T@inverse@xd)
    projection=inverse-inverse@xd@covariance@xd.T@inverse
    py=projection@first["Y_old"]
    coefficients=covariance@xd.T@inverse@first["Y_old"]
    score=py@kd@py-torch.trace(projection@kd)
    information=py@kd@projection@kd@py
    eta=first["Y_old"]-py/first["weights_old"]
    for key,expected in (("alpha",coefficients),("cov",covariance),("PY",py),
                         ("score",score),("AI",information),("eta",eta)):
        torch.testing.assert_close(first[key],expected,atol=2e-12,rtol=2e-12)


def test_family_detection_and_binary_no_spa_are_explicit():
    assert infer_family([0.,1.,np.nan]) == "binomial"
    assert infer_family([0.,1.000001]) == "gaussian"
    n=64;y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],8)
    out=prepare_phewas_model(y,np.arange(n).astype(str),design(n),association_mode="fp64",use_spa=False)
    assert out.spa_model is None and out.metadata["normal_only"]
    with pytest.raises(ValueError,match="exact 0 and 1"):
        prepare_phewas_model(y+.001,np.arange(n).astype(str),design(n),family="binomial")
