"""Mask BED byte-oracle checks; synthetic fixtures are not benchmarks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torchwgs.mask_output import MaskWriter, _pack_mask_cpu, _pack_mask_device


_SUFFIXES = ('.bed', '.bim', '.fam', '.snplist')
_INVALID = 'Max mask BED calls must lie in 0..2'


def _mask(values, index=0, *, raw=True):
    result = SimpleNamespace(name=f'M{index}', base_name=f'B{index}',
        aaf_upper=1 if index % 3 == 0 else .01, frequency='0.01',
        burden=values, variant_ids=(f'v{index}_2', f'v{index}_1'))
    if raw:
        result.raw_burden = values
        result.burden = torch.full_like(values, 9)
    return result


def _artifacts(masks, gene='SYNTHETIC_GENE', position=42):
    return SimpleNamespace(gene=SimpleNamespace(gene=gene, chrom='21', position=position),
                           masks=masks)


def _samples(n):
    return [(f'F{n-i}', f'S{n-i}') for i in range(n)], [i % 3 for i in range(n)]


def _files(prefix, partial=False):
    return {suffix: Path(str(prefix) + '_masks' + suffix + ('.partial' if partial else '')).read_bytes()
            for suffix in _SUFFIXES}


def _flush_files(writer):
    for stream in (writer.bed, writer.bim, writer.snplist):
        stream.flush()


def _values(n, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    choices = np.array([-.5, 0, .5, 1, 1.5, 2, 2.49, np.nan, np.inf, -np.inf], dtype=np_dtype)
    rng = np.random.default_rng(91)
    return torch.from_numpy(choices[rng.integers(0, len(choices), n)])


def _reused_objects(device):
    gene = SimpleNamespace(gene='INITIAL', chrom='1', position=1)
    mask = _mask(torch.zeros(5, device=device))
    def masks():
        for index in range(3):
            gene.gene = f'GENE_{index}'
            gene.chrom = str(index + 1)
            gene.position = 100 + index
            mask.name = f'M{index}'
            mask.base_name = f'B{index}'
            mask.aaf_upper = 1 if index == 0 else .01
            mask.variant_ids = [f'v{index}_first', f'v{index}_second']
            mask.raw_burden = torch.full((5,), index, dtype=torch.float64, device=device)
            yield mask
            # Also mutate the previously yielded membership list in place.
            mask.variant_ids[:] = ['changed_after_yield']
    return SimpleNamespace(gene=gene, masks=masks())


class MaskOutputCPUTests(unittest.TestCase):
    def test_device_operations_match_independent_numpy_bytes_and_padding(self):
        # Exercise exactly the device algorithm on CPU without starting CUDA.
        for dtype in (torch.float32, torch.float64):
            for n in (0, 1, 2, 3, 4, 5, 13, 4097):
                values = _values(n, dtype)
                expected = _pack_mask_cpu(values.numpy(), n)
                actual = _pack_mask_device(values, n)
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertEqual(actual.shape, ((n + 3) // 4 + 1,))
                self.assertEqual(int(actual[-1]), 0)
                np.testing.assert_array_equal(actual[:-1].numpy(), expected)
                if n % 4:
                    self.assertEqual(int(actual[-2]) >> (2 * (n % 4)), 0)

    def test_half_up_missing_and_order_have_explicit_known_bytes(self):
        # 0,.5,1.5,NA -> call 0,1,2,NA -> bits 11,10,00,01.
        values = torch.tensor([0., .5, 1.5, float('nan'), 0.], dtype=torch.float64)
        self.assertEqual(_pack_mask_cpu(values.numpy(), 5).tobytes(), b'\x4b\x03')
        self.assertEqual(_pack_mask_device(values, 5).tolist(), [0x4b, 0x03, 0])
        for dtype in (torch.float32, torch.float64):
            half = torch.tensor([.5, 1.5], dtype=dtype)
            below = torch.nextafter(half, torch.full_like(half, -float('inf')))
            above = torch.nextafter(half, torch.full_like(half, float('inf')))
            values = torch.cat((below, half, above))
            np.testing.assert_array_equal(_pack_mask_device(values, 6)[:-1],
                                          _pack_mask_cpu(values.numpy(), 6))

    def test_invalid_finite_calls_are_flagged_without_indexing_them(self):
        for dtype in (torch.float32, torch.float64):
            boundary = torch.tensor(-.5, dtype=dtype)
            lower = torch.nextafter(boundary, torch.tensor(-float('inf'), dtype=dtype))
            for invalid in (lower, torch.tensor(2.5, dtype=dtype),
                            torch.tensor(torch.finfo(dtype).max, dtype=dtype)):
                values = torch.stack((torch.tensor(0., dtype=dtype), invalid))
                with np.errstate(invalid='ignore'):
                    with self.assertRaisesRegex(ValueError, _INVALID):
                        _pack_mask_cpu(values.numpy(), 2)
                self.assertEqual(int(_pack_mask_device(values, 2)[-1]), 1)
        integer = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
        np.testing.assert_array_equal(_pack_mask_device(integer, 4)[:-1],
                                      _pack_mask_cpu(integer.numpy(), 4))

    def test_cpu_writer_preserves_raw_priority_all_frequency_and_attachments(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'output'
            ids, sex = _samples(5)
            raw = torch.tensor([0., .5, 1.5, float('nan'), 0.], dtype=torch.float64)
            first = _mask(raw, 0)
            second = _mask(torch.tensor([2., 1., 0., float('inf'), 2.]), 1, raw=False)
            second.raw_burden = None
            with MaskWriter(prefix, ids, sex) as writer:
                writer(_artifacts([first]))
                writer(_artifacts([second], gene='NEXT_GENE', position=43))
                writer(_artifacts([]))
                self.assertEqual(writer.n_masks, 2)
            files = _files(prefix)
            self.assertEqual(files['.bed'], b'\x6c\x1b\x01\x4b\x03\x78\x00')
            self.assertEqual(files['.bim'],
                b'21\tSYNTHETIC_GENE.M0.all\t0\t42\tB0.all\tref\n'
                b'21\tNEXT_GENE.M1.0.01\t0\t43\tB1.0.01\tref\n')
            self.assertEqual(files['.snplist'],
                b'SYNTHETIC_GENE.M0.all\tv0_2,v0_1\n'
                b'NEXT_GENE.M1.0.01\tv1_2,v1_1\n')
            self.assertEqual(files['.fam'], ''.join(
                f'{fid}\t{iid}\t0\t0\t{s}\t-9\n' for (fid, iid), s in zip(ids, sex)).encode())
            self.assertFalse(list(Path(directory).glob('*.partial')))
            writer.close()

    def test_batched_device_rows_preserve_first_invalid_partial_write(self):
        with TemporaryDirectory() as directory:
            cpu_prefix, batched_prefix = Path(directory) / 'cpu', Path(directory) / 'batched'
            ids, sex = _samples(5)
            masks = [_mask(torch.zeros(5), 0), _mask(torch.full((5,), 2.5), 1),
                     _mask(torch.ones(5), 2)]
            artifacts = _artifacts(masks)
            cpu = MaskWriter(cpu_prefix, ids, sex)
            batched = MaskWriter(batched_prefix, ids, sex)
            try:
                with self.assertRaisesRegex(ValueError, _INVALID):
                    cpu(artifacts)
                pending = [(batched._metadata(artifacts.gene, mask),
                            _pack_mask_device(mask.raw_burden, 5)) for mask in masks]
                with self.assertRaisesRegex(ValueError, _INVALID):
                    batched._flush_cuda(pending)
                self.assertEqual(pending, [])
                self.assertEqual(cpu.n_masks, 1)
                self.assertEqual(batched.n_masks, 1)
                _flush_files(cpu)
                _flush_files(batched)
                self.assertEqual(_files(cpu_prefix, True), _files(batched_prefix, True))
            finally:
                cpu.close(commit=False)
                batched.close(commit=False)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_exception_discards_partials_and_preserves_existing_final_files(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'output'
            ids, sex = _samples(5)
            with MaskWriter(prefix, ids, sex) as writer:
                writer(_artifacts([_mask(torch.zeros(5))]))
            original = _files(prefix)
            with self.assertRaisesRegex(ValueError, _INVALID):
                with MaskWriter(prefix, ids, sex) as writer:
                    writer(_artifacts([_mask(torch.ones(5)), _mask(torch.full((5,), 2.5), 1)]))
            self.assertEqual(_files(prefix), original)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_zero_samples_empty_masks_and_packed_batch_budget(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'empty'
            with MaskWriter(prefix, [], []) as writer:
                writer(_artifacts([]))
                writer(_artifacts([_mask(torch.empty(0))]))
                self.assertEqual(writer.n_masks, 1)
            self.assertEqual(_files(prefix)['.bed'], b'\x6c\x1b\x01')
            self.assertEqual(_files(prefix)['.fam'], b'')
            with patch('torchwgs.mask_output._CUDA_MASK_BATCH_BYTES', 25):
                writer = MaskWriter(Path(directory) / 'bounded', *_samples(13))
                self.assertEqual(writer._cuda_batch_limit, 5)
                writer.close(commit=False)
            with patch('torchwgs.mask_output._CUDA_MASK_BATCH_BYTES', 1):
                writer = MaskWriter(Path(directory) / 'single', *_samples(13))
                self.assertEqual(writer._cuda_batch_limit, 1)
                writer.close(commit=False)

    def test_metadata_is_immutable_when_gene_and_mask_objects_are_reused(self):
        with TemporaryDirectory() as directory:
            cpu_prefix, batch_prefix = Path(directory) / 'cpu', Path(directory) / 'batched'
            ids, sex = _samples(5)
            with MaskWriter(cpu_prefix, ids, sex) as writer:
                writer(_reused_objects('cpu'))
            artifacts = _reused_objects('cpu')
            with MaskWriter(batch_prefix, ids, sex) as writer:
                pending = []
                for mask in artifacts.masks:
                    pending.append((writer._metadata(artifacts.gene, mask),
                                    _pack_mask_device(mask.raw_burden, 5)))
                writer._flush_cuda(pending)
            self.assertEqual(_files(batch_prefix), _files(cpu_prefix))
            self.assertEqual(_files(batch_prefix)['.snplist'],
                b'GENE_0.M0.all\tv0_first,v0_second\n'
                b'GENE_1.M1.0.01\tv1_first,v1_second\n'
                b'GENE_2.M2.0.01\tv2_first,v2_second\n')
            self.assertEqual(_files(batch_prefix)['.bim'],
                b'1\tGENE_0.M0.all\t0\t100\tB0.all\tref\n'
                b'2\tGENE_1.M1.0.01\t0\t101\tB1.0.01\tref\n'
                b'3\tGENE_2.M2.0.01\t0\t102\tB2.0.01\tref\n')

    def test_write_metadata_errors_keep_range_priority_and_partial_stages(self):
        with TemporaryDirectory() as directory:
            ids, sex = _samples(5)
            for suffix in ('missing_chrom', 'bad_members'):
                prefix = Path(directory) / suffix
                artifacts = _artifacts([_mask(torch.zeros(5))])
                if suffix == 'missing_chrom':
                    del artifacts.gene.chrom
                    expected_type = AttributeError
                else:
                    artifacts.masks[0].variant_ids = [123]
                    expected_type = TypeError
                writer = MaskWriter(prefix, ids, sex)
                try:
                    # Invalid dosage is still checked before write metadata.
                    artifacts.masks[0].raw_burden.fill_(2.5)
                    with self.assertRaisesRegex(ValueError, _INVALID):
                        writer(artifacts)
                    _flush_files(writer)
                    self.assertEqual(_files(prefix, True)['.bed'], b'\x6c\x1b\x01')
                    artifacts.masks[0].raw_burden.zero_()
                    with self.assertRaises(expected_type):
                        writer(artifacts)
                    _flush_files(writer)
                    files = _files(prefix, True)
                    self.assertEqual(files['.bed'], b'\x6c\x1b\x01\xff\x03')
                    self.assertEqual(files['.bim'] == b'', suffix == 'missing_chrom')
                    self.assertEqual(files['.snplist'], b'')
                    self.assertEqual(writer.n_masks, 0)
                finally:
                    writer.close(commit=False)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for device mask writer')
class MaskOutputCUDATests(unittest.TestCase):
    def test_cuda_reused_gene_mask_and_membership_match_sequential_cpu_bytes(self):
        with TemporaryDirectory() as directory:
            cpu_prefix, cuda_prefix = Path(directory) / 'cpu', Path(directory) / 'cuda'
            ids, sex = _samples(5)
            with MaskWriter(cpu_prefix, ids, sex) as writer:
                writer(_reused_objects('cpu'))
            with MaskWriter(cuda_prefix, ids, sex) as writer:
                writer(_reused_objects('cuda'))
            self.assertEqual(_files(cuda_prefix), _files(cpu_prefix))

    def test_cuda_encoder_exact_bytes_across_padding_and_precision(self):
        for dtype in (torch.float32, torch.float64):
            for n in (0, 1, 2, 3, 4, 5, 13, 4103):
                values = _values(n, dtype)
                actual = _pack_mask_device(values.to('cuda'), n)
                self.assertEqual(actual.device.type, 'cuda')
                self.assertEqual(int(actual[-1]), 0)
                np.testing.assert_array_equal(actual[:-1].cpu(), _pack_mask_cpu(values.numpy(), n))

    def test_cuda_writer_matches_all_cpu_files_and_only_transfers_packed_batches(self):
        with TemporaryDirectory() as directory:
            cpu_prefix, cuda_prefix = Path(directory) / 'cpu', Path(directory) / 'cuda'
            ids, sex = _samples(13)
            for dtype in (torch.float32, torch.float64):
                masks = [_mask(_values(13, dtype).roll(index), index, raw=index % 2 == 0)
                         for index in range(17)]
                with MaskWriter(cpu_prefix, ids, sex) as writer:
                    writer(_artifacts(masks))
                cuda_masks = []
                for mask in masks:
                    copied = SimpleNamespace(**vars(mask))
                    copied.burden = mask.burden.to('cuda')
                    if hasattr(mask, 'raw_burden'):
                        copied.raw_burden = mask.raw_burden.to('cuda')
                    cuda_masks.append(copied)
                transfers = []
                original_cpu = torch.Tensor.cpu
                def logged_cpu(tensor, *args, **kwargs):
                    transfers.append((tensor.dtype, tuple(tensor.shape)))
                    return original_cpu(tensor, *args, **kwargs)
                with patch.object(torch.Tensor, 'cpu', new=logged_cpu):
                    with MaskWriter(cuda_prefix, ids, sex) as writer:
                        writer(_artifacts(cuda_masks))
                        self.assertEqual(writer.n_masks, 17)
                self.assertEqual(transfers, [(torch.uint8, (8, 5)),
                                             (torch.uint8, (8, 5)), (torch.uint8, (1, 5))])
                self.assertEqual(_files(cuda_prefix), _files(cpu_prefix))

    def test_cuda_invalid_partial_order_and_earlier_error_priority(self):
        with TemporaryDirectory() as directory:
            ids, sex = _samples(5)
            masks = [_mask(torch.zeros(5, device='cuda'), index) for index in range(12)]
            masks[9] = _mask(torch.full((5,), 2.5, device='cuda'), 9)
            prefix = Path(directory) / 'invalid'
            writer = MaskWriter(prefix, ids, sex)
            try:
                with self.assertRaisesRegex(ValueError, _INVALID):
                    writer(_artifacts(masks))
                self.assertEqual(writer.n_masks, 9)
                _flush_files(writer)
                self.assertEqual(len(_files(prefix, True)['.bed']), 3 + 9 * 2)
                self.assertEqual(len(_files(prefix, True)['.bim'].splitlines()), 9)
            finally:
                writer.close(commit=False)
            # A malformed later mask must not replace an earlier range error.
            with self.assertRaisesRegex(ValueError, _INVALID):
                with MaskWriter(Path(directory) / 'priority', ids, sex) as writer:
                    writer(_artifacts([masks[9], SimpleNamespace()]))
            prefix = Path(directory) / 'generator'
            writer = MaskWriter(prefix, ids, sex)
            def interrupted_masks():
                yield masks[0]
                raise RuntimeError('mask iterator failed')
            try:
                with self.assertRaisesRegex(RuntimeError, 'mask iterator failed'):
                    writer(_artifacts(interrupted_masks()))
                self.assertEqual(writer.n_masks, 1)
                _flush_files(writer)
                self.assertEqual(len(_files(prefix, True)['.bed']), 5)
            finally:
                writer.close(commit=False)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_cuda_batch_byte_bound_and_mixed_cpu_device_order(self):
        with TemporaryDirectory() as directory:
            cpu_prefix, cuda_prefix = Path(directory) / 'cpu', Path(directory) / 'cuda'
            ids, sex = _samples(13)
            masks = [_mask(_values(13, torch.float64).roll(index), index, raw=False)
                     for index in range(12)]
            with MaskWriter(cpu_prefix, ids, sex) as writer:
                writer(_artifacts(masks))
            for index, mask in enumerate(masks):
                if index != 6:
                    mask.burden = mask.burden.to('cuda')
            transfers = []
            original_cpu = torch.Tensor.cpu
            def logged_cpu(tensor, *args, **kwargs):
                if tensor.device.type == 'cuda':
                    transfers.append((tensor.dtype, tuple(tensor.shape)))
                return original_cpu(tensor, *args, **kwargs)
            with patch('torchwgs.mask_output._CUDA_MASK_BATCH_BYTES', 25), \
                 patch.object(torch.Tensor, 'cpu', new=logged_cpu):
                with MaskWriter(cuda_prefix, ids, sex) as writer:
                    writer(_artifacts(masks))
            self.assertEqual(transfers, [(torch.uint8, (5, 5)),
                                         (torch.uint8, (1, 5)), (torch.uint8, (5, 5))])
            self.assertEqual(_files(cuda_prefix), _files(cpu_prefix))

    def test_cuda_exception_rollback_preserves_existing_outputs(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'output'
            ids, sex = _samples(5)
            with MaskWriter(prefix, ids, sex) as writer:
                writer(_artifacts([_mask(torch.zeros(5))]))
            original = _files(prefix)
            with self.assertRaisesRegex(ValueError, _INVALID):
                with MaskWriter(prefix, ids, sex) as writer:
                    writer(_artifacts([_mask(torch.ones(5, device='cuda')),
                        _mask(torch.full((5,), -.6, device='cuda'), 1)]))
            self.assertEqual(_files(prefix), original)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_cuda_malformed_shape_retains_numpy_error_and_preceding_writes(self):
        with TemporaryDirectory() as directory:
            ids, sex = _samples(5)
            cpu_prefix, cuda_prefix = Path(directory) / 'cpu', Path(directory) / 'cuda'
            cpu = MaskWriter(cpu_prefix, ids, sex)
            cuda = MaskWriter(cuda_prefix, ids, sex)
            try:
                malformed = torch.zeros(1)
                with self.assertRaises(IndexError) as expected:
                    cpu(_artifacts([_mask(torch.zeros(5)), _mask(malformed, 1)]))
                with self.assertRaises(type(expected.exception)) as actual:
                    cuda(_artifacts([_mask(torch.zeros(5, device='cuda')),
                                     _mask(malformed.to('cuda'), 1)]))
                self.assertEqual(str(actual.exception), str(expected.exception))
                self.assertEqual(cuda.n_masks, 1)
                _flush_files(cpu)
                _flush_files(cuda)
                self.assertEqual(_files(cuda_prefix, True), _files(cpu_prefix, True))
            finally:
                cpu.close(commit=False)
                cuda.close(commit=False)


if __name__ == '__main__':
    unittest.main()
