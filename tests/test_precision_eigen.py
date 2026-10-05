"""Precision dispatch/error controls; synthetic arithmetic is not a benchmark."""
import numpy as np
import pytest
import torch

from staar_phewas import _precision_eigen as precision
from staar_phewas.statistics import _ordered_sum

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_monotonic_brackets_cover_root_interval_with_declared_margin(device):
    roots = torch.tensor([-.02, -.01000002, -.010000005, -.009, 0., .009,
                          .010000005, .01000002, .02], dtype=torch.float64, device=device)
    spectrum = torch.tensor([.2, .7, 1.], dtype=torch.float64, device=device)
    spectra = spectrum.expand(len(roots), -1)
    statistics = (spectra / (1 - 2 * spectra * roots[:, None])).sum(dim=1)
    selected = precision.near_mean_mask(statistics, spectra)
    assert selected.cpu().tolist() == [False, False, True, True, True, True, True, False, False]
    # Integer scaling of q and every eigenvalue does not change the root.
    assert torch.equal(selected, precision.near_mean_mask(statistics * 32, spectra * 32))


@pytest.mark.parametrize("device", DEVICES)
def test_selection_preserves_zero_and_invalid_spectrum_paths(device):
    spectra = torch.tensor([[.2, 1.], [0., 0.], [1e-12, 1e-12], [float('nan'), 1.],
                            [.2, 1.], [.2, 1.]], dtype=torch.float64, device=device)
    statistic = torch.tensor([0., 1., 1., 1., -1., float('inf')], dtype=torch.float64, device=device)
    assert not bool(precision.near_mean_mask(statistic, spectra).any())


def test_missing_and_incorrect_reference_libraries_fail_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(precision, "_LIBRARY", None)
    monkeypatch.delenv("STAAR_REFERENCE_LAPACK_LIBRARY", raising=False)
    with pytest.raises(RuntimeError, match="STAAR_REFERENCE_LAPACK_LIBRARY"):
        precision.cpu_reference_eigenvalues(torch.eye(2, dtype=torch.float64))
    monkeypatch.setenv("STAAR_REFERENCE_LAPACK_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(RuntimeError, match="does not exist"):
        precision._library()
    supplied = tmp_path / "different.so"
    supplied.write_bytes(b"an unrelated library")
    monkeypatch.setenv("STAAR_REFERENCE_LAPACK_LIBRARY", str(supplied))
    with pytest.raises(RuntimeError, match="configured binary differs"):
        precision._library()



@pytest.mark.parametrize("matrix", [torch.empty((0, 0), dtype=torch.float64),
                                     torch.ones((2, 3), dtype=torch.float64),
                                     torch.eye(2, dtype=torch.float32),
                                     torch.tensor([[float('nan')]], dtype=torch.float64),
                                     torch.tensor([[float('inf')]], dtype=torch.float64)])
def test_invalid_matrix_is_rejected_before_loading_numeric_library(monkeypatch, matrix):
    monkeypatch.setattr(precision, "_library", lambda: pytest.fail("must reject before library load"))
    with pytest.raises(ValueError, match="Reference spectrum"):
        precision.cpu_reference_eigenvalues(matrix)

def test_raw_upper_is_only_weighted_after_sensitive_row_selection(monkeypatch):
    raw = torch.tensor([[1., np.nextafter(.1, 1.)], [.1, 2.]], dtype=torch.float64)
    weights = torch.tensor([2., 3.], dtype=torch.float64)
    eigenvalues = torch.tensor([1., 2.], dtype=torch.float64)
    products, solves = [], []

    class CUDAPlacement:
        is_cuda = True

        def __mul__(self, right):
            products.append(right.clone())
            return raw * right

    def solve(matrix):
        solves.append(matrix.clone())
        return eigenvalues

    monkeypatch.setattr(precision, "cpu_reference_eigenvalues", solve)
    _, selected = precision.refine_near_mean_spectrum(
        CUDAPlacement(), eigenvalues, eigenvalues.new_tensor(3.), weights=weights)
    assert selected and len(products) == len(solves) == 1
    torch.testing.assert_close(solves[0], raw * weights[:, None] * weights[None, :], atol=0., rtol=0.)
    before = len(products)
    result, selected = precision.refine_near_mean_spectrum(
        CUDAPlacement(), eigenvalues, eigenvalues.new_tensor(10.), weights=weights)
    assert not selected and result is eigenvalues
    assert len(products) == before


def test_cpu_path_requires_no_reference_library(monkeypatch):
    monkeypatch.delenv("STAAR_REFERENCE_LAPACK_LIBRARY", raising=False)
    matrix = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    eigenvalues = matrix.diagonal().clone()
    actual, refined = precision.refine_near_mean_spectrum(matrix, eigenvalues, matrix.new_tensor(3.))
    assert actual is eigenvalues and not refined


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_quadratic_rows_use_original_scalar_addition_order():
    products = torch.tensor([[1e16, 1.], [1., 1e16], [-1e16, 2.], [3., -1e16]],
                            dtype=torch.float64, device="cuda")
    # Increasing-row float64 sums are 3 and 2; a parallel tree can differ.
    torch.testing.assert_close(_ordered_sum(products).cpu(),
                               torch.tensor([3., 2.], dtype=torch.float64), atol=0., rtol=0.)


def test_metadata_reset_preserves_configuration_and_returns_independent_snapshot():
    saved = precision.precision_eigen_execution_metadata(reset=True)
    current = precision.precision_eigen_execution_metadata()
    assert current["cpu_eigen_calls"] == current["gpu_eigen_calls"] == 0
    assert current["nearmean_root_threshold"] == .01 and current["bracket_margin"] == 1e-8
    saved["gpu_eigen_routes"]["changed"] = 1
    assert "changed" not in precision.precision_eigen_execution_metadata()["gpu_eigen_routes"]


@pytest.mark.parametrize("info_code", [0, 1])
def test_reference_dsyev_has_fixed_original_workspace_and_one_call(monkeypatch, info_code):
    calls = []
    raw = torch.tensor([[1., np.nextafter(.1, 1.)], [.1, 2.]], dtype=torch.float64)
    precision.precision_eigen_execution_metadata(reset=True)
    def dsyev(job, upper, n_pointer, a_pointer, lda_pointer, output_pointer, work_pointer, lwork_pointer, info_pointer):
        n = n_pointer._obj.value
        lwork = lwork_pointer._obj.value
        assert job == b"N" and upper == b"U"
        assert lda_pointer._obj.value == n and lwork == 66*n
        matrix = np.ctypeslib.as_array(a_pointer, shape=(n*n,)).reshape((n,n), order="F")
        np.testing.assert_array_equal(matrix, raw.numpy())
        output = np.ctypeslib.as_array(output_pointer, shape=(n,))
        output[:] = [.5, 2.]
        info_pointer._obj.value = info_code
        calls.append(lwork)
    class Library:
        dsyev_ = staticmethod(dsyev)
    monkeypatch.setattr(precision, "_library", lambda: Library())
    if info_code:
        with pytest.raises(RuntimeError, match="eigen solve failed"):
            precision.cpu_reference_eigenvalues(raw)
    else:
        actual = precision.cpu_reference_eigenvalues(raw)
        torch.testing.assert_close(actual, torch.tensor([.5,2.], dtype=torch.float64), atol=0., rtol=0.)
    assert calls == [132]
    meta = precision.precision_eigen_execution_metadata()
    assert meta["cpu_eigen_calls"] == int(info_code == 0)
    assert meta["workspace_policy"] == "fixed_66n_original_armadillo"
