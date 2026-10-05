"""Reference arithmetic checks; these small fixtures are not benchmarks."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from staar_phewas._reference_weights import (
    reference_annotation_weights,
    reference_complementary_phred,
    reference_weights_execution_metadata,
)
from staar_phewas.masks import annotation_phred_matrix


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_all_weight_types_match_independent_original_R_arithmetic(device):
    fixture = json.loads((Path(__file__).parent/'data/reference_weights.json').read_text())
    maf = torch.tensor(fixture['maf'], dtype=torch.float64, device=device)
    complete_phred = torch.tensor(fixture['phred'], dtype=torch.float64, device=device)
    outputs = reference_annotation_weights(maf, complete_phred)
    for result, kind in zip(outputs, ('burden', 'skat', 'acat')):
        expected = np.array([[float.fromhex(value) for value in row] for row in fixture[kind]])
        np.testing.assert_array_equal(result.cpu().numpy().view(np.uint64), expected.view(np.uint64))


def test_complementary_phred_matches_original_R_boundaries_and_rounding():
    # Includes the R NA payload, retained by the scalar transformation.
    r_na = np.array([0x7FF00000000007A2], dtype=np.uint64).view(np.float64)[0]
    values = np.array([r_na, np.nan, 0.0, -0.0, -1.0, -10000.0,
                       -np.inf, np.inf, 1e-300, .001, .5, 100.0, 10000.0])
    result = reference_complementary_phred(values)
    expected_bits = np.array([
        0x7FF80000000007A2, 0x7FF8000000000000,
        0x7FF0000000000000, 0x7FF0000000000000,
        0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000,
        0x8000000000000000, 0x7FF0000000000000,
        0x4042306D8BFBE839, 0x4023458057F136BA,
        0x3DFDD830A0F8DBC7, 0x8000000000000000,
    ], dtype=np.uint64)
    np.testing.assert_array_equal(result.view(np.uint64), expected_bits)


def test_raw_annotation_construction_preserves_order_and_missing_behavior():
    matrix, names = annotation_phred_matrix(
        {'CADD': [np.nan, 5.0, 10.0], 'aPC.LocalDiversity': [.001, .5, np.inf]},
        ['aPC.LocalDiversity', 'CADD'], indices=[1, 0, 2],
    )
    assert names == ['aPC.LocalDiversity', 'aPC.LocalDiversity(-)', 'CADD']
    np.testing.assert_array_equal(matrix[:, 0], [.5, .001, np.inf])
    expected = np.array([float.fromhex('0x1.3458057f136bap+3'),
                         float.fromhex('0x1.2306d8bfbe839p+5'), -0.0])
    np.testing.assert_array_equal(matrix[:, 1].copy().view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(matrix[:, 2], [5.0, 0.0, 10.0])


@pytest.mark.parametrize('maf', [[], [[.01]], [0.0], [-.01], [.5001], [np.nan], [np.inf]])
def test_weight_helper_preserves_frequency_validation(maf):
    with pytest.raises(ValueError):
        reference_annotation_weights(maf)


@pytest.mark.parametrize('annotations', [[1.0], [[1.0], [2.0]], [[-1.0]], [[np.nan]], [[np.inf]]])
def test_weight_helper_preserves_complete_phred_validation(annotations):
    with pytest.raises(ValueError):
        reference_annotation_weights([.01], annotations)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_weight_execution_metadata_records_conversion_boundary(device):
    reference_weights_execution_metadata(reset=True)
    maf = torch.tensor([.001, .5], dtype=torch.float64, device=device)
    complete_phred = torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float64, device=device)
    profile = {}
    outputs = reference_annotation_weights(maf, complete_phred, profile=profile)
    assert all(result.shape == (2, 6) and result.device.type == device for result in outputs)
    metadata = reference_weights_execution_metadata()
    assert metadata['weight_calls'] == 1
    assert metadata['weight_rows'] == 2
    assert metadata['weight_annotation_cells'] == 4
    assert metadata['weight_conversion_wall_seconds'] >= metadata['weight_scalar_transform_seconds'] >= 0
    assert metadata['weight_d2h_calls'] == (2 if device == 'cuda' else 0)
    assert metadata['weight_d2h_bytes'] == (48 if device == 'cuda' else 0)
    assert metadata['weight_h2d_calls'] == (3 if device == 'cuda' else 0)
    assert metadata['weight_h2d_bytes'] == (288 if device == 'cuda' else 0)
    assert metadata['complementary_phred_calls'] == 0  # Supplied PHRED stays supplied.
    assert 'host wall' in metadata['timing_scope']
    assert profile['total_seconds'] == metadata['weight_conversion_wall_seconds']
