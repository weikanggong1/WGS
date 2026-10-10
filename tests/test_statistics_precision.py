"""Precision routing checks; CPU mocks are not TF32 accuracy benchmarks."""
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import fudan_wgs_toolkit.statistics as statistics


class _RejectTorchProducts(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in ('aten.mm.default', 'aten.mv.default', 'aten.dot.default',
                         'aten.addmm.default', 'aten.bmm.default'):
            raise AssertionError(f'Unexpected Torch matrix product: {func}')
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('mac,products', [([1., 2., 30.], 2),
                                         ([1., 2., 3.], 1),
                                         ([20., 30., 40.], 1)])
def test_all_burden_columns_route_through_backend_once(monkeypatch, mac, products):
    score = torch.tensor([.2, -.1, .5], dtype=torch.float64)
    covariance = torch.tensor([[2., .1, .3], [.1, 1., .2], [.3, .2, 3.]], dtype=torch.float64)
    maf = torch.tensor([.001, .002, .004], dtype=torch.float64)
    annotation = torch.tensor([[10., 20.], [20., 40.], [30., 50.]], dtype=torch.float64)
    skat_pvalues = torch.linspace(.1, .6, 6, dtype=torch.float64)
    payload = dict(score=score, covariance=covariance, maf=maf, mac=mac,
                   annotations=annotation, _skat_pvalues=skat_pvalues)
    expected = statistics.association_test(**payload)
    calls = []

    def mock_backend(left, right, **kwargs):
        mode = "ieee_fp32"
        # Independent CPU arithmetic solely exercises orchestration. Actual
        # Tensor Core precision is checked by the serial real-data GPU probe.
        calls.append((tuple(left.shape), tuple(right.shape), mode))
        return torch.as_tensor(np.matmul(left.numpy(), right.numpy()), dtype=torch.float32)

    from fudan_wgs_toolkit import _burden
    monkeypatch.setattr(_burden, 'ieee_burden_product', mock_backend)
    statistics.statistics_execution_metadata(reset=True)
    with _RejectTorchProducts():
        actual = statistics.association_test(**payload, matmul_mode='tf32')
    assert len(calls) == products
    assert all(shape[1][1] == 6 and shape[2] == 'ieee_fp32' for shape in calls)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=2e-6, rel=2e-6)
    metadata = statistics.statistics_execution_metadata()
    assert metadata['burden_matrix_products'] == 1
    assert metadata['rare_burden_matrix_products'] == products - 1
    assert metadata['matmul_mode_calls'] == {'tf32': 1}


def test_score_products_all_use_selected_backend(monkeypatch):
    calls = []
    def mock_backend(left, right, *, mode):
        calls.append(mode)
        return torch.as_tensor(np.matmul(left.numpy(), right.numpy()), dtype=torch.float32)
    genotype = torch.tensor([[0., 1.], [1., 0.], [2., 1.], [0., 2.]], dtype=torch.float64)
    residual = torch.tensor([.4, -.1, .2, -.5], dtype=torch.float64)
    covariates = torch.ones((4, 1), dtype=torch.float64)
    reference = statistics.score_covariance(genotype, residual, covariates=covariates)
    monkeypatch.setattr(statistics, 'matmul', mock_backend)
    with _RejectTorchProducts():
        actual = statistics.score_covariance(genotype, residual, covariates=covariates,
                                             matmul_mode='tf32')
    assert calls == ['tf32'] * 5
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value.double(), expected, rtol=2e-6, atol=2e-6)


def test_forced_mode_rejects_sparse_projector_without_densifying():
    with pytest.raises(ValueError, match='sparse precision/projector'):
        statistics.score_covariance(torch.ones((3, 2)), torch.ones(3),
            projector=torch.eye(3).to_sparse_coo(), matmul_mode='tf32')


def test_forced_mode_on_cpu_does_not_fall_back():
    with pytest.raises(ValueError, match='same CUDA device'):
        statistics.association_test([.1, .2], [[1., 0.], [0., 1.]], [.001, .002], [2., 4.],
                              _skat_pvalues=[.1, .2], matmul_mode='tf32')
