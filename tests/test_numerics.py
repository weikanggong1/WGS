"""R-generated numerical rules; these matrices are not benchmark data."""
import json
from pathlib import Path
import pytest
import torch
from staar_phewas.numerics import reference_crossprod, extended_variance


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_dot_first_call_and_strided_tails(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    fixtures = json.loads((Path(__file__).parent / "data/reference_dot.json").read_text())
    for fixture in fixtures["fixtures"]:
        # A view with stride 2 exercises input layout independently of the
        # sixteen-lane and eight-lane tail lengths in the genuine R oracle.
        left = torch.tensor(fixture["left"], dtype=torch.float64, device=device)
        right = torch.tensor(fixture["right"], dtype=torch.float64, device=device)
        storage = torch.empty((len(left), 2), dtype=left.dtype, device=device)
        storage[:, 0] = left
        storage[:, 1] = right
        a = storage[:, :1]
        first = reference_crossprod(a, storage[:, 1])
        second = reference_crossprod(a, storage[:, 1])
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        expected = torch.tensor([fixture["expected"]], dtype=left.dtype, device=device)
        torch.testing.assert_close(first, expected, atol=2e-14, rtol=2e-15)
        expected_variance = torch.tensor(fixture["variance_expected"], dtype=left.dtype, device=device)
        torch.testing.assert_close(extended_variance(right), expected_variance, atol=2e-15, rtol=5e-15)


def test_reference_crossprod_multiple_fixed_columns():
    x = torch.tensor([[1., -1.], [1., 0.], [1., 2.]], dtype=torch.float64)
    y = torch.tensor([3., 2., 7.], dtype=torch.float64)
    torch.testing.assert_close(reference_crossprod(x, y), torch.tensor([12., 11.], dtype=torch.float64), atol=0, rtol=0)
