"""CPU algebra and fitted-state contracts for binary Single centering.

These fixtures test the supplied projection without fitting a replacement
model. They make no GPU timing or real-data accuracy claim.
"""
import numpy as np
import pytest
import torch

import staar_phewas.binary_null as binary_module
from staar_phewas.binary_null import BinaryNullModel


FIELDS = ("x", "precision_x", "precision", "fixed_effect_covariance",
          "scaled_residuals")


@pytest.fixture
def products(monkeypatch):
    calls = []

    def selected_product(left, right, *, mode):
        assert left.device.type == right.device.type == "cpu"
        dtype = torch.float32 if mode == "tf32" else torch.float64
        calls.append((mode, tuple(left.shape), tuple(right.shape)))
        return left.to(dtype=dtype) @ right.to(dtype=dtype)

    monkeypatch.setattr(binary_module, "matmul", selected_product)
    return calls


def model_state(*, mode="tf32", nonsymmetric=False, use_spa=False):
    n = 16
    dtype = torch.float32 if mode == "tf32" else torch.float64
    coordinate = torch.arange(n, dtype=dtype)
    design = torch.stack((torch.ones(n, dtype=dtype),
                          torch.linspace(-.9, .8, n, dtype=dtype),
                          torch.cos(coordinate * .27)), dim=1)
    precision = .2 + (coordinate % 5) * .05
    precision_design = precision[:, None] * design
    # Finite fitted state: its projection deliberately does not annihilate 1.
    covariance = torch.linalg.inv(design.T @ precision_design) * .97
    if nonsymmetric:
        covariance = covariance.clone()
        covariance[0, 1] += .04
        covariance[1, 0] -= .04
    return BinaryNullModel(sample_ids=np.arange(n), x=design,
        scaled_residuals=torch.linspace(-.15, .25, n, dtype=dtype) + .7,
        fitted_probability=torch.linspace(.2, .7, n, dtype=dtype),
        xw=precision_design.T, projection_left=design @ covariance,
        fixed_effect_covariance=covariance, precision=precision,
        precision_x=precision_design, phenotype=(coordinate % 2),
        use_spa=use_spa, matmul_mode=mode)


def genotype(model, *, layout="c"):
    n = model.n
    g = torch.zeros((n, 8), dtype=torch.float32)
    g[:, 1] = 1
    g[:, 2] = 2
    g[:2, 3] = 1
    g[:, 4] = 1
    g[0, 4] = 0
    g[-1, 4] = 2
    g[:, 5] = torch.arange(n) % 3
    g[:, 6] = .125 + (torch.arange(n) % 4) * .25
    g[:, 7] = 2
    g[:3, 7] = 1.5
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
    return torch.diag(model.precision.double()) - a @ c @ a.T


def legacy_single(model, g, mode):
    dtype = torch.float32 if mode == "tf32" else torch.float64
    original = g.to(dtype=dtype)
    if mode == "fp64":
        # FP64/SPA uses the original full covariance reduction, not a newly
        # chosen rowwise dot-product order.
        score, covariance = model.score_covariance(original, matmul_mode=mode)
        return score, covariance.diagonal()
    precision = model.precision
    if precision is None:
        precision = model.fitted_probability * (1 - model.fitted_probability)
    precision = precision.to(dtype=dtype)
    if precision.ndim == 1:
        weighted = precision[:, None] * original
    elif precision.layout != torch.strided:
        weighted = torch.sparse.mm(precision, original)
    else:
        weighted = precision @ original
    a = model.precision_x if model.precision_x is not None else model.xw.T
    cross = a.to(dtype=dtype).T @ original
    projected = cross.T @ model.fixed_effect_covariance.to(dtype=dtype)
    variance = (original * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
    return original.T @ model.scaled_residuals.to(dtype=dtype), variance


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_centering_keeps_supplied_quadratic_and_original_score(products, layout, nonsymmetric):
    model = model_state(nonsymmetric=nonsymmetric)
    g = genotype(model, layout=layout)
    before_g = g.clone()
    before_state = {name: getattr(model, name).clone() for name in FIELDS}
    p = projection_oracle(model)
    expected_variance = (g.double() * (p @ g.double())).sum(dim=0)
    expected_score = g.T @ model.scaled_residuals
    score, variance = model.individual_score_variance(g)
    assert score.dtype == variance.dtype == torch.float32
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance.double(), expected_variance,
                               rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(g, before_g, rtol=0, atol=0)
    for name, value in before_state.items():
        torch.testing.assert_close(getattr(model, name), value, rtol=0, atol=0)
    # Dropping means without the finite P1 terms would zero constant columns.
    assert abs(float(p.sum())) > .01
    assert float(variance[1]) > .01 and float(variance[2]) > .01
    assert torch.count_nonzero(score[1:3]).item() == 2
    assert all(mode == "tf32" for mode, _, _ in products)


def test_p_one_uses_symmetric_part_of_supplied_covariance(products):
    model = model_state(nonsymmetric=True)
    p = projection_oracle(model)
    ones = torch.ones(model.n, dtype=torch.float64)
    expected = ((p + p.T) * .5) @ ones
    unsymmetrized = p @ ones
    p_one, p_one_sum = model._centered_single_projection_state()
    torch.testing.assert_close(p_one.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(p_one_sum.double(), expected.sum(),
                               rtol=2e-5, atol=2e-6)
    assert float(torch.max(torch.abs(expected - unsymmetrized))) > .01


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
def test_empty_variant_axis_returns_empty_fp32_vectors(products, layout):
    model = model_state()
    g = genotype(model, layout=layout)[:, :0]
    shape, stride = g.shape, g.stride()
    score, variance = model.individual_score_variance(g)
    assert score.shape == variance.shape == (0,)
    assert score.dtype == variance.dtype == torch.float32
    assert g.shape == shape and g.stride() == stride


@pytest.mark.parametrize("route", ["no_precision", "no_precision_x", "dense_precision",
                                  "sparse_precision", "no_intercept", "near_intercept",
                                  "joint", "genotype_grad", "fp64_model", "fp64_override"])
def test_ineligible_single_retains_original_core_exactly(products, monkeypatch, route):
    model = model_state(mode="fp64" if route == "fp64_model" else "tf32")
    if route == "no_precision":
        model.precision = None
    elif route == "no_precision_x":
        model.precision_x = None
    elif route in ("dense_precision", "sparse_precision"):
        model.precision = torch.diag(model.precision)
        if route == "sparse_precision":
            model.precision = model.precision.to_sparse_coo()
    elif route == "no_intercept":
        model.x[:, 0] = .5
    elif route == "near_intercept":
        model.x[0, 0] += torch.finfo(torch.float32).eps
    elif route == "joint":
        model.n_pheno = 2
    g = genotype(model)
    if route == "genotype_grad":
        g.requires_grad_(True)
    mode = "fp64" if route in ("fp64_model", "fp64_override") else "tf32"
    if route in ("fp64_model", "fp64_override", "genotype_grad"):
        def forbidden():
            raise AssertionError("this route must not invoke centered projection")
        monkeypatch.setattr(model, "_centered_single_projection_state", forbidden)
    else:
        assert model._centered_single_projection_state() is None
    expected_score, expected_variance = legacy_single(model, g, mode)
    score, variance = model.individual_score_variance(g, matmul_mode=mode)
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)


def test_spa_retains_fp64_and_rejects_native_tf32_before_centering(products, monkeypatch):
    model = model_state(mode="fp64", use_spa=True)
    assert model._centered_single_projection_state() is None

    def forbidden():
        raise AssertionError("SPA must not invoke centered projection")
    monkeypatch.setattr(model, "_centered_single_projection_state", forbidden)
    g = genotype(model)
    expected_score, expected_variance = legacy_single(model, g, "fp64")
    score, variance = model.individual_score_variance(g)
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)
    with pytest.raises(ValueError, match="SPA requires fp64"):
        model.individual_score_variance(g, matmul_mode="tf32")


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("change", ["float64", "requires_grad"])
def test_mixed_dtype_or_gradient_state_uses_original_formula(products, field, change):
    model = model_state()
    value = getattr(model, field)
    setattr(model, field, value.double() if change == "float64"
            else value.clone().requires_grad_(True))
    assert model._centered_single_projection_state() is None
    g = genotype(model)
    expected_score, expected_variance = legacy_single(model, g, "tf32")
    score, variance = model.individual_score_variance(g)
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)


def test_projection_cache_reuses_valid_state(products):
    model = model_state()
    first = model._centered_single_projection_state()
    calls = len(products)
    assert model._centered_single_projection_state() is first
    assert len(products) == calls
    assert model._centered_single_projection_cache[1] is first


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("change", ["inplace", "reassigned"])
def test_cache_invalidates_each_fitted_field_mutation(products, field, change):
    model = model_state()
    original = model._centered_single_projection_state()
    value = getattr(model, field)
    if change == "reassigned":
        setattr(model, field, value.clone())
    elif field == "x":
        value[:, 1].add_(.02)  # Keep the strict intercept.
    else:
        value.add_(.002)
    calls = len(products)
    refreshed = model._centered_single_projection_state()
    assert refreshed is not original
    assert len(products) > calls
    p = projection_oracle(model)
    expected = ((p + p.T) * .5) @ torch.ones(model.n, dtype=torch.float64)
    torch.testing.assert_close(refreshed[0].double(), expected,
                               rtol=2e-5, atol=2e-6)
    assert model._centered_single_projection_state() is refreshed


def test_invalid_optional_state_drops_existing_cache(products):
    model = model_state()
    first = model._centered_single_projection_state()
    saved = model.precision_x
    model.precision_x = None
    assert model._centered_single_projection_state() is None
    assert model._centered_single_projection_cache is None
    model.precision_x = saved
    assert model._centered_single_projection_state() is not first


def test_negative_intercept_cache_invalidates_when_column_becomes_ones(products):
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
def test_inference_tensors_bypass_versioned_cache(products, field):
    model = model_state()
    cached = model._centered_single_projection_state()
    with torch.inference_mode():
        inference_value = getattr(model, field).clone()
    with pytest.raises(RuntimeError, match="version"):
        _ = inference_value._version
    setattr(model, field, inference_value)
    first = model._centered_single_projection_state()
    calls = len(products)
    second = model._centered_single_projection_state()
    assert first is not cached and second is not first
    assert model._centered_single_projection_cache is None
    assert len(products) > calls
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    score, variance = model.individual_score_variance(genotype(model))
    assert score.dtype == variance.dtype == torch.float32
    assert bool(torch.isfinite(variance).all())
