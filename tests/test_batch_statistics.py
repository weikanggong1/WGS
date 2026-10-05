"""Batch execution checks; small matrices are not a real-data benchmark."""
import numpy as np
import pytest
import torch

from staar_phewas.batch_statistics import _saddle_batch, staar_test_batch
from staar_phewas.null_model import GaussianNullModel, KinshipSpectrum
from staar_phewas.statistics import DegenerateTestError, annotation_weights, quadratic_form_sf, staar_test


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_batched_masks_preserve_filtering_fields_order_and_workspace_fallback(device):
    covariance = torch.tensor([[1., .2, .1], [.2, 1.2, .3], [.1, .3, .9]], dtype=torch.float64, device=device)
    three = dict(score=torch.tensor([.4, -.8, .6], dtype=torch.float64, device=device),
                 covariance=covariance, maf=[.001, .005, .009], mac=[2, 11, 4],
                 annotations=[[10.], [20.], [30.]], names=["functional"], cmac=17.25)
    filtered = dict(three, maf=[.001, .005, .1], acat_calibration="gaussian_glm", dof=30)
    items = [three, filtered, three]
    expected = [staar_test(**item) for item in items]
    actual, info = staar_test_batch(items, return_diagnostics=True)
    assert info["weighted_matrices"] == 12
    assert info["eigen_batches"] == 2
    for result, reference in zip(actual, expected):
        assert list(result) == list(reference)
        for field in result:
            assert result[field] == pytest.approx(reference[field], rel=1e-7, abs=1e-10)
    fallback, info = staar_test_batch(items, max_workspace_bytes=1, return_diagnostics=True)
    assert info["workspace_serial_masks"] == 3
    assert info["eigen_batches"] == 0
    assert fallback == expected
    assert staar_test_batch([]) == []


@pytest.mark.parametrize("device", DEVICES)
def test_batched_saddle_zero_mean_cutoffs_and_far_tails(device):
    spectra = torch.tensor([[.2, 1., 2.]] * 6, dtype=torch.float64, device=device)
    q = torch.tensor([0., .001, .1, 3.2, 20., 100.], dtype=torch.float64, device=device)
    values, compatible = _saddle_batch(q, spectra)
    assert compatible[3]
    for index in range(len(q)):
        # Compatibility rows are deliberately recomputed by the production
        # caller: eigensolver/reduction order matters near the spectral mean.
        if not compatible[index]:
            assert float(values[index]) == pytest.approx(quadratic_form_sf(q[index], spectra[index]), rel=1e-7, abs=1e-10)
    with pytest.raises(DegenerateTestError):
        _saddle_batch(q[:1], torch.zeros((1, 3), dtype=torch.float64, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_batched_negative_bound_matches_original_scalar_division(device):
    # Same public arithmetic fixture as the serial regression. No private
    # q/eigen values are needed to expose scalar/Tensor reverse division.
    eigenvalues = torch.tensor([[1/3, 2/3, 1.]], dtype=torch.float64, device=device)
    statistic = eigenvalues.new_tensor([1.9996])
    probability, compatibility = _saddle_batch(statistic, eigenvalues)
    assert bool(compatibility[0])
    expected = 0.39548257230568734  # Original STAAR 0.9.9 Saddle.
    assert float(probability[0]) == pytest.approx(expected, abs=1e-10, rel=1e-7)
    assert quadratic_form_sf(statistic[0], eigenvalues[0]) == pytest.approx(expected, abs=1e-10, rel=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_individual_variance_matches_explicit_related_projector(device):
    # A real eigensystem is exercised, without relying on fitted AI convergence.
    spectrum = KinshipSpectrum.from_sparse([.5, .6, .7, .8, .9], [0, 2], [1, 3], [.1, .15], device=device)
    x = torch.tensor([[1., -2.], [1., -1.], [1., 0.], [1., 1.], [1., 2.]], dtype=torch.float64, device=device)
    inverse = 1 / (.8 + .3 * spectrum.eigenvalues)
    precision = spectrum.rotate(inverse[:, None] * spectrum.rotate(torch.eye(5, dtype=torch.float64, device=device)), inverse=True)
    sx = precision @ x
    fixed = torch.linalg.inv(x.T @ sx)
    residual = torch.tensor([.4, -.1, .2, -.3, -.2], dtype=torch.float64, device=device)
    model = GaussianNullModel(np.arange(5).astype(str), x, residual, x.new_zeros(2), x.new_tensor([.8, .3]),
                              x.new_tensor([.8, .3]), fixed, spectrum, inverse, sx, 0, True)
    genotype = torch.tensor([[0., 1., 0.], [1., 0., 1.], [2., 1., 0.], [0., 2., 1.], [1., 0., 2.]], dtype=torch.float64, device=device)
    score, variance = model.individual_score_variance(genotype[:, ::2])
    projector = precision - sx @ fixed @ sx.T
    expected = (genotype[:, ::2].T @ projector @ genotype[:, ::2]).diagonal()
    torch.testing.assert_close(score, genotype[:, ::2].T @ residual, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(variance, expected, atol=1e-13, rtol=1e-13)
    with pytest.raises(ValueError, match="finite"):
        model.individual_score_variance(genotype[:3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_cuda_compatibility_tail_reuses_batched_spectrum(monkeypatch):
    # These genuine beta-weight matrices have statistics near their own means.
    # The scalar tail must reuse the batch spectrum rather than changing CUDA
    # eigensolver by solving each matrix a second time.
    diagonal = torch.tensor([.2, 1., 2.], dtype=torch.float64, device="cuda")
    score = torch.sqrt(diagonal * .995)
    covariance = torch.diag(diagonal)
    maf = torch.tensor([.001, .005, .009], dtype=torch.float64, device="cuda")
    item = dict(score=score, covariance=covariance, maf=maf, mac=torch.full_like(maf, 20))
    weights = annotation_weights(maf)[1]
    original = torch.linalg.eigvalsh
    calls, spectra = [], []

    def record(matrix, **kwargs):
        calls.append(tuple(matrix.shape))
        value = original(matrix, **kwargs)
        spectra.append(value.detach().clone())
        return value

    monkeypatch.setattr(torch.linalg, "eigvalsh", record)
    results, info = staar_test_batch([item], return_diagnostics=True)
    assert info["compatibility_rows"] == 2
    assert calls == [(2, 3, 3)]
    for column, field in enumerate(["SKAT(1,25)", "SKAT(1,1)"]):
        statistic = torch.sum(score.square() * weights[:, column].square())
        expected = quadratic_form_sf(statistic, spectra[0][column])
        assert results[0][field] == pytest.approx(expected, abs=1e-10, rel=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("matrix_capacity", [1, 3])
def test_small_cuda_workspace_singletons_use_natural_serial_mask(monkeypatch, matrix_capacity):
    # Four genuine weight columns give a singleton final chunk for capacity=3.
    # Capacity=1 cannot batch two matrices at all. Neither case may silently
    # switch to single-matrix SYEVD solely because of a workspace setting.
    diagonal = torch.tensor([1/3, 2/3, 1.], dtype=torch.float64, device="cuda")
    score = torch.sqrt(diagonal * .9998)
    maf = torch.tensor([.001, .005, .009], dtype=torch.float64, device="cuda")
    item = dict(score=score, covariance=torch.diag(diagonal), maf=maf,
                mac=torch.full_like(maf, 20), annotations=[[10.], [20.], [30.]],
                names=["functional"])
    reference = staar_test(**item)
    original = torch.linalg.eigvalsh
    calls = []

    def record(matrix, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigvalsh", record)
    results, info = staar_test_batch([item], max_workspace_bytes=matrix_capacity * 4 * 8 * 3**2,
                                    return_diagnostics=True)
    assert info["workspace_serial_masks"] == 1
    assert info["eigen_batches"] == 0
    assert info["weighted_matrices"] == 0
    assert calls == [(4, 3, 3)]
    assert list(results[0]) == list(reference)
    for field in reference:
        assert results[0][field] == pytest.approx(reference[field], abs=1e-10, rel=1e-7)
    # The unannotated beta(1,1) column also retains the original scalar tail.
    assert results[0]["SKAT(1,1)"] == pytest.approx(0.39548257230568734, abs=1e-10, rel=1e-7)
