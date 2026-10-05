"""Differential checks for block downloads and native result serialization."""
from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torchwgs.io import BedReader, Variant, write_bed
from torchwgs.output import RegenieWriter
from torchwgs.single import (_copy_result_columns, create_test_context,
                            iter_single_variant_results, SingleVariantConfig)


def original_download(statistics):
    columns = {name: values.detach().cpu().tolist()
               for name, values in statistics.items() if name != 'VALID'}
    return columns, statistics['VALID'].detach().cpu().tolist()


class SingleResultTransferTests(unittest.TestCase):
    def test_download_preserves_integer_counts_signed_zero_and_key_order(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                statistics = OrderedDict([
                    ('N', torch.tensor([2**53+1, 3], device=device, dtype=torch.int64)),
                    ('BETA', torch.tensor([-0., 1.125], device=device, dtype=dtype)),
                    ('SE', torch.tensor([float('inf'), float('nan')], device=device, dtype=dtype)),
                    ('VALID', torch.tensor([True, False], device=device)),
                ])
                actual, valid = _copy_result_columns(statistics)
                expected, expected_valid = original_download(statistics)
                self.assertEqual(list(actual), list(expected))
                self.assertEqual(actual['N'], expected['N'])
                self.assertIsInstance(actual['N'][0], int)
                self.assertEqual(actual['BETA'], expected['BETA'])
                self.assertTrue(np.signbit(actual['BETA'][0]))
                self.assertEqual(actual['SE'][0], expected['SE'][0])
                self.assertTrue(np.isnan(actual['SE'][1]))
                self.assertEqual([bool(value) for value in valid], expected_valid)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA result download required')
    def test_native_results_and_ids_match_original_download_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_ids = [(str(i), str(i)) for i in range(43)]
            variants = [Variant(i, '21', 'variant_'+str(i), i+1, 'A', 'G')
                        for i in range(9)]
            rng = np.random.default_rng(791)
            genotype = rng.integers(0, 3, size=(43, 9)).astype(np.float64)
            genotype[:, 0] = 0.; genotype[:, 1] = np.nan
            genotype[:4, 2] = np.nan
            genotype[:, 3] = 0.; genotype[:3, 3] = 1.
            prefix = root/'input'
            write_bed(prefix, genotype, variants, sample_ids)
            for dtype in ('float32', 'float64'):
                context = create_test_context(rng.normal(size=43), sample_ids=sample_ids[::-1],
                    device='cuda', dtype=dtype, apply_rint=False)
                configuration = SingleVariantConfig(device='cuda', dtype=dtype,
                    genotype_reader='cuda_packed', min_mac=3, maf_min=0., block_size=3)
                actual = list(iter_single_variant_results(BedReader(prefix), context,
                    config=configuration, variant_indices=[8, 3, 2, 0, 1, 5, 3]))
                with patch('torchwgs.single._copy_result_columns', side_effect=original_download):
                    expected = list(iter_single_variant_results(BedReader(prefix), context,
                        config=configuration, variant_indices=[8, 3, 2, 0, 1, 5, 3]))
                self.assertEqual(actual, expected)
                written = {}
                for label, rows in (('new', actual), ('old', expected)):
                    with RegenieWriter(root/(label+'_'+dtype), 'trait_01',
                            sample_ids=context.sample_ids, write_samples=True,
                            print_pheno_name=True) as writer:
                        for row in rows:
                            writer.write(row)
                        written[label] = (writer.path, writer.ids_path)
                for actual_path, expected_path in zip(written['new'], written['old']):
                    self.assertEqual(actual_path.read_bytes(), expected_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
