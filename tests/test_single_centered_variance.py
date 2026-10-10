"""CPU algebra and cache contracts; no GPU or scientific benchmark claims."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.null_model import GaussianNullModel, KinshipSpectrum
import fudan_wgs_toolkit.null_model as null_module


FIELDS = ("x", "precision_x", "inverse_variance", "fixed_effect_covariance", "scaled_residuals")


@pytest.fixture
def products(monkeypatch):
    calls = []
    def selected_product(left, right, *, mode):
        dtype = torch.float32 if mode == "tf32" else torch.float64
        calls.append((mode, tuple(left.shape), tuple(right.shape)))
        return left.to(dtype=dtype) @ right.to(dtype=dtype)
    monkeypatch.setattr(null_module, "matmul", selected_product)
    return calls


def model_state(*, mode="tf32", rotations=False, n_pheno=1, nonsymmetric=False):
    n = 16
    dtype = torch.float32 if mode == "tf32" else torch.float64
    coordinate = torch.arange(n, dtype=dtype)
    design = torch.stack((torch.ones(n, dtype=dtype), torch.linspace(-.9, .8, n, dtype=dtype),
                          torch.cos(coordinate * .27)), dim=1)
    precision = .4 + (coordinate % 5) * .1
    precision_design = precision[:, None] * design
    # Fixed finite-tolerance state: P1 deliberately does not vanish.
    covariance = torch.linalg.inv(design.T @ precision_design) * .97
    if nonsymmetric:
        covariance = covariance.clone()
        covariance[0, 1] += .04
        covariance[1, 0] -= .04
    blocks = [(torch.tensor([0, 1]), torch.tensor([[.8, -.6], [.6, .8]], dtype=dtype))] if rotations else []
    return GaussianNullModel(sample_ids=np.arange(n), x=design,
        scaled_residuals=torch.linspace(-.15, .25, n, dtype=dtype) + .7,
        coefficients=torch.zeros(3, dtype=dtype), theta=torch.ones(2, dtype=dtype),
        precision_theta=torch.ones(2, dtype=dtype), fixed_effect_covariance=covariance,
        spectrum=KinshipSpectrum(torch.ones(n, dtype=dtype), blocks),
        inverse_variance=precision, precision_x=precision_design, iterations=0, converged=True,
        n_pheno=n_pheno, matmul_mode=mode)


def genotype(model, *, layout="c"):
    n = model.n
    g = torch.zeros((n, 8), dtype=torch.float32)
    g[:, 1] = 1
    g[:, 2] = 2
    g[:2, 3] = 1
    g[:, 4] = 1;g[0, 4] = 0;g[-1, 4] = 2
    g[:, 5] = torch.arange(n) % 3
    g[:, 6] = .125 + (torch.arange(n) % 4) * .25
    g[:, 7] = 2;g[:3, 7] = 1.5
    if layout == "f":
        return g.T.contiguous().T
    if layout == "strided":
        backing = torch.full((n, 16), -123., dtype=torch.float32)
        backing[:, ::2] = g
        return backing[:, ::2]
    return g


def projection_oracle(model):
    a = model.precision_x.double()
    c = model.fixed_effect_covariance.double()
    return torch.diag(model.inverse_variance.double()) - a @ c @ a.T


def legacy_single(model, g, mode):
    dtype = torch.float32 if mode == "tf32" else torch.float64
    original = g.to(dtype=dtype)
    rotated = original.clone() if model.spectrum.blocks else original
    for indices, rotation in model.spectrum.blocks:
        rotated[indices] = rotation.to(dtype=dtype).T @ original[indices]
    weighted = model.inverse_variance.to(dtype=dtype)[:, None] * rotated
    cross = model.precision_x.to(dtype=dtype).T @ original
    projected = cross.T @ model.fixed_effect_covariance.to(dtype=dtype)
    variance = (rotated * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
    return original.T @ model.scaled_residuals.to(dtype=dtype), variance


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_centering_keeps_finite_state_quadratic_and_original_score(products, layout, nonsymmetric):
    model = model_state(nonsymmetric=nonsymmetric)
    g = genotype(model, layout=layout)
    before_g = g.clone()
    before_state = {name: getattr(model, name).clone() for name in FIELDS}
    p = projection_oracle(model)
    expected = (g.double() * (p @ g.double())).sum(dim=0)
    expected_score = g.T @ model.scaled_residuals
    score, variance = model.individual_score_variance(g)
    assert score.dtype == variance.dtype == torch.float32
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance.double(), expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(g, before_g, rtol=0, atol=0)
    for name, value in before_state.items():
        torch.testing.assert_close(getattr(model, name), value, rtol=0, atol=0)
    # Simply dropping the means would incorrectly zero the constant columns.
    assert abs(float(p.sum())) > .01
    assert float(variance[1]) > .01 and float(variance[2]) > .01
    assert torch.count_nonzero(score[1:3]).item() == 2
    assert all(mode == "tf32" for mode, _, _ in products)


def test_p_one_uses_symmetric_part_of_nonsymmetric_supplied_covariance(products):
    model = model_state(nonsymmetric=True)
    p = projection_oracle(model)
    expected = ((p + p.T) * .5) @ torch.ones(model.n, dtype=torch.float64)
    unsymmetrized = p @ torch.ones(model.n, dtype=torch.float64)
    p_one, p_one_sum = model._centered_single_projection_state()
    torch.testing.assert_close(p_one.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(p_one_sum.double(), expected.sum(), rtol=2e-5, atol=2e-6)
    assert float(torch.max(torch.abs(expected - unsymmetrized))) > .01


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
def test_empty_variant_axis_returns_empty_fp32_vectors_without_mutating_input(products, layout):
    model = model_state()
    g = genotype(model, layout=layout)[:, :0]
    shape, stride = g.shape, g.stride()
    score, variance = model.individual_score_variance(g)
    assert score.shape == variance.shape == (0,)
    assert score.dtype == variance.dtype == torch.float32
    assert g.shape == shape and g.stride() == stride


@pytest.mark.parametrize("route", ["fp64_model", "fp64_override", "no_intercept", "near_intercept",
                                  "rotation", "joint", "genotype_grad"])
def test_other_single_paths_retain_original_reduction_exactly(products, monkeypatch, route):
    model = model_state(mode="fp64" if route == "fp64_model" else "tf32",
                        rotations=route == "rotation", n_pheno=2 if route == "joint" else 1)
    if route == "no_intercept":
        model.x[:, 0] = .5
    elif route == "near_intercept":
        model.x[0, 0] += torch.finfo(torch.float32).eps
    g = genotype(model)
    if route == "genotype_grad":
        g.requires_grad_(True)
    mode = "fp64" if route in ("fp64_model", "fp64_override") else "tf32"
    if route in ("fp64_model", "fp64_override", "genotype_grad"):
        def forbidden():
            raise AssertionError("this path must not invoke centered projection")
        monkeypatch.setattr(model, "_centered_single_projection_state", forbidden)
    else:
        assert model._centered_single_projection_state() is None
    expected_score, expected_variance = legacy_single(model, g, mode)
    score, variance = model.individual_score_variance(g, matmul_mode=mode)
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("change", ["float64", "requires_grad"])
def test_mixed_dtype_or_gradient_state_uses_original_single_formula(products, field, change):
    model = model_state()
    value = getattr(model, field)
    setattr(model, field, value.double() if change == "float64" else value.clone().requires_grad_(True))
    assert model._centered_single_projection_state() is None
    g = genotype(model)
    expected_score, expected_variance = legacy_single(model, g, "tf32")
    score, variance = model.individual_score_variance(g)
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)


def test_projection_cache_reuses_valid_tensor_state_and_keeps_original_fields_alive(products):
    model = model_state()
    first = model._centered_single_projection_state()
    calls = len(products)
    second = model._centered_single_projection_state()
    assert second is first
    assert len(products) == calls
    assert model._centered_single_projection_cache[1] is first


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("change", ["inplace", "reassigned"])
def test_projection_cache_invalidates_for_each_fitted_field_mutation(products, field, change):
    model = model_state()
    original = model._centered_single_projection_state()
    value = getattr(model, field)
    if change == "reassigned":
        setattr(model, field, value.clone())
    elif field == "x":
        value[:, 1].add_(.02)  # preserve the strict intercept
    else:
        value.add_(.002)
    calls = len(products)
    refreshed = model._centered_single_projection_state()
    assert refreshed is not original
    assert len(products) > calls
    p = projection_oracle(model)
    expected_p_one = ((p+p.T)*.5) @ torch.ones(model.n, dtype=torch.float64)
    torch.testing.assert_close(refreshed[0].double(), expected_p_one, rtol=2e-5, atol=2e-6)
    assert model._centered_single_projection_state() is refreshed


def test_negative_intercept_cache_invalidates_when_leading_column_becomes_ones(products):
    model = model_state()
    model.x[:, 0] = .5
    assert model._centered_single_projection_state() is None
    calls = len(products)
    assert model._centered_single_projection_state() is None
    assert len(products) == calls
    model.x[:, 0] = 1
    assert model._centered_single_projection_state() is not None
    assert len(products) > calls


def test_mode_conversion_releases_cached_projection(products):
    model = model_state()
    assert model._centered_single_projection_state() is not None
    model.set_matmul_mode("fp64")
    assert model._centered_single_projection_cache is None
    assert model._centered_single_projection_state() is None
    model.set_matmul_mode("tf32")
    assert model._centered_single_projection_cache is None
    assert model._centered_single_projection_state() is not None


@pytest.mark.parametrize("field", FIELDS)
def test_inference_tensor_without_version_counter_bypasses_projection_cache(products, field):
    model = model_state()
    cached = model._centered_single_projection_state()
    with torch.inference_mode():
        inference_value = getattr(model, field).clone()
    with pytest.raises(RuntimeError, match="version"):
        _ = inference_value._version
    setattr(model, field, inference_value)
    first = model._centered_single_projection_state()
    count = len(products)
    second = model._centered_single_projection_state()
    assert first is not cached and second is not first
    assert model._centered_single_projection_cache is None
    assert len(products) > count
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    score, variance = model.individual_score_variance(genotype(model))
    assert score.dtype == variance.dtype == torch.float32
    assert bool(torch.isfinite(variance).all())
