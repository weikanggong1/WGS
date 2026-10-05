"""Optional REGENIE _masks BED/BIM/FAM/snplist artifacts, streamed by gene."""
from pathlib import Path

import numpy as np
import torch


_INVALID_CALL_MESSAGE = 'Max mask BED calls must lie in 0..2'
_CUDA_MASK_BATCH_SIZE = 8
_CUDA_MASK_BATCH_BYTES = 1 << 20


def _pack_mask_cpu(g, sample_count):
    """Original NumPy encoder, kept independent of the device encoder."""
    stride = (sample_count + 3) // 4
    valid = np.isfinite(g)
    calls = np.floor(np.where(valid, g, 0) + .5).astype(np.int64)
    if np.any(valid & ((calls < 0) | (calls > 2))):
        raise ValueError(_INVALID_CALL_MESSAGE)
    codes = np.zeros(stride * 4, dtype=np.uint8)
    codes[:sample_count] = np.array([3, 2, 0], dtype=np.uint8)[calls]
    codes[:sample_count][~valid] = 1
    return (codes.reshape(-1, 4) << np.array([0, 2, 4, 6], dtype=np.uint8)).sum(1).astype(np.uint8)


def _pack_mask_device(g, sample_count):
    """Encode on the input device; the final byte flags an invalid mask.

    Each call uses one mask's sample vector as temporary workspace. Only its
    packed bytes and error flag survive until the bounded transfer batch.
    NumPy promotes integer inputs to float64 before adding .5; retain that
    behavior without changing the floating-point burden precision.
    """
    if not g.is_floating_point() and not g.is_complex():
        g = g.to(torch.float64)
    stride = (sample_count + 3) // 4
    valid = torch.isfinite(g)
    calls = torch.floor(torch.where(valid, g, 0) + .5)
    invalid = (valid & ((calls < 0) | (calls > 2))).any()
    codes = torch.zeros(stride * 4, dtype=torch.uint8, device=g.device)
    # Half-up rounding is intentional: torch.round uses ties-to-even. Calls
    # outside 0..2 are flagged above and never written, so no unsafe lookup is
    # needed. Padding remains 00, independent of the last actual sample.
    codes[:sample_count].masked_fill_(calls == 0, 3)
    codes[:sample_count].masked_fill_(calls == 1, 2)
    codes[:sample_count].masked_fill_(~valid, 1)
    groups = codes.reshape(-1, 4)
    packed = (groups[:, 0] | (groups[:, 1] << 2) |
              (groups[:, 2] << 4) | (groups[:, 3] << 6))
    return torch.cat((packed, invalid.reshape(1).to(torch.uint8)))


class MaskWriter:
    def __init__(self, prefix, sample_ids, sample_sex):
        self.prefix = str(prefix) + '_masks'
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        self.n = len(sample_ids)
        self.stride = (self.n + 3) // 4
        self.n_masks = 0
        self.closed = False
        # Keep at most eight packed rows and normally at most 1 MiB. A single
        # unusually large row is still processed alone, never a dense mask
        # matrix. No floating-point burdens are cached or transferred here.
        self._cuda_batch_limit = max(1, min(_CUDA_MASK_BATCH_SIZE,
            _CUDA_MASK_BATCH_BYTES // (self.stride + 1)))
        self.bed = open(self.prefix + '.bed.partial', 'wb')
        self.bed.write(b'\x6c\x1b\x01')
        self.bim = open(self.prefix + '.bim.partial', 'w')
        self.snplist = open(self.prefix + '.snplist.partial', 'w')
        with open(self.prefix + '.fam.partial', 'w') as f:
            for (fid, iid), sex in zip(sample_ids, sample_sex):
                f.write(f'{fid}\t{iid}\t0\t0\t{sex}\t-9\n')

    @staticmethod
    def _metadata(gene, mask):
        frequency = 'all' if mask.aaf_upper == 1 else mask.frequency
        identifier = f'{gene.gene}.{mask.name}.{frequency}'
        allele = f'{mask.base_name}.{frequency}'
        # A generator may reuse and mutate gene/mask objects before a batch is
        # flushed. Snapshot output text now, without retaining those objects.
        # These fields were originally read after call validation and BED
        # writing. Save their exceptions for the same write stage so a range
        # error still wins, and preceding partial bytes remain identical.
        bim_line = snplist_line = None
        bim_error = snplist_error = None
        try:
            bim_line = f'{gene.chrom}\t{identifier}\t0\t{gene.position}\t{allele}\tref\n'
        except Exception as error:
            bim_error = (type(error), error.args)
        if bim_error is None:
            try:
                snplist_line = identifier + '\t' + ','.join(mask.variant_ids) + '\n'
            except Exception as error:
                snplist_error = (type(error), error.args)
        return bim_line, snplist_line, bim_error, snplist_error

    def _write_mask(self, metadata, packed):
        bim_line, snplist_line, bim_error, snplist_error = metadata
        self.bed.write(packed.tobytes())
        if bim_error is not None:
            kind, args = bim_error
            raise kind(*args)
        self.bim.write(bim_line)
        if snplist_error is not None:
            kind, args = snplist_error
            raise kind(*args)
        self.snplist.write(snplist_line)
        self.n_masks += 1

    def _flush_cuda(self, pending):
        if not pending:
            return
        # One D2H transfer contains only uint8 BED rows and their error flags.
        records = list(pending)
        pending.clear()
        rows = torch.stack([row for _, row in records]).cpu().numpy()
        for (metadata, _), row in zip(records, rows):
            if row[-1]:
                # Write all preceding valid rows before raising, as in the
                # original sequential encoder. Context-manager cleanup still
                # discards every partial file when an exception escapes.
                raise ValueError(_INVALID_CALL_MESSAGE)
            self._write_mask(metadata, row[:-1])

    def __call__(self, artifacts):
        pending = []
        pending_device = None
        try:
            for mask in artifacts.masks:
                metadata = self._metadata(artifacts.gene, mask)
                raw = getattr(mask, 'raw_burden', None)
                g = (mask.burden if raw is None else raw).detach()
                if g.device.type == 'cuda' and g.shape == (self.n,) and not g.is_complex():
                    if pending and g.device != pending_device:
                        self._flush_cuda(pending)
                    packed = _pack_mask_device(g, self.n)
                    pending_device = g.device
                    pending.append((metadata, packed))
                    if len(pending) == self._cuda_batch_limit:
                        self._flush_cuda(pending)
                else:
                    self._flush_cuda(pending)
                    # Pipeline burdens are real sample vectors. For unusual
                    # shapes or complex caller input, preserve NumPy's exact
                    # validation/error behavior rather than silently broadcast.
                    packed = _pack_mask_cpu(g.cpu().numpy(), self.n)
                    self._write_mask(metadata, packed)
            self._flush_cuda(pending)
        except BaseException:
            # A generator or a later mask may fail before the next transfer.
            # Preserve preceding writes and the earliest mask-range error.
            self._flush_cuda(pending)
            raise

    def close(self, commit=True):
        if self.closed:
            return
        self.closed = True
        self.bed.close()
        self.bim.close()
        self.snplist.close()
        for suffix in ('.bed', '.bim', '.fam', '.snplist'):
            partial = Path(self.prefix + suffix + '.partial')
            if commit:
                partial.replace(self.prefix + suffix)
            else:
                partial.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(commit=exc[0] is None)
