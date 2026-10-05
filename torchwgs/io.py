"""Streaming PLINK BED reader. Dosages count BIM A1 (the default REGENIE ALT)."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import itertools
import json
import os
import sqlite3
import tempfile
import time
import numpy as np
import torch


_BIM_INDEX_SCHEMA = 1
_BIM_INDEX_APPLICATION_ID = 0x57475342
_BIM_INDEX_INSERT_ROWS = 8192


@dataclass(frozen=True)
class Variant:
    index: int
    chrom: str
    id: str
    position: int
    allele1: str
    allele0: str


@dataclass
class PackedBedBlock:
    """One uploaded BED block; count hardcalls before decoding retained sites."""
    packed: torch.Tensor
    sample_bytes: torch.Tensor
    sample_shifts: torch.Tensor
    lookup: torch.Tensor

    def allele_counts(self):
        """Exact integer counts in the selected analysis-sample order."""
        from ._packed_gpu import packed_genotype_counts
        missing, hom_a1, het, hom_a0 = packed_genotype_counts(
            self.packed, self.sample_bytes, self.sample_shifts)
        return {'N': hom_a1 + het + hom_a0, 'AAC': 2 * hom_a1 + het,
                'N_MISSING': missing, 'N_HOM_A1': hom_a1,
                'N_HET': het, 'N_HOM_A0': hom_a0}

    def decode(self, columns=None):
        """Decode block-column indices, preserving their supplied order.

        ``columns=None`` decodes every site. Indices are validated on CPU;
        the single-variant path reuses its metadata selection for this check.
        """
        from ._packed_gpu import decode_packed_columns
        if columns is None:
            selected = torch.arange(self.packed.shape[0], device=self.packed.device)
        else:
            if isinstance(columns, torch.Tensor):
                columns = columns.detach().cpu().numpy()
            indices = np.asarray(columns, dtype=np.int64).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= self.packed.shape[0]):
                raise IndexError('Variant column outside packed block')
            selected = torch.as_tensor(indices, device=self.packed.device)
        return decode_packed_columns(self.packed, self.sample_bytes,
                                     self.sample_shifts, selected, self.lookup)


def read_sample_ids(path):
    """Read one-column IID or two-column FID/IID keep/remove files."""
    ids = set()
    with open(path) as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].lstrip('#').upper() in ('FID', 'IID', 'ID', 'EID') or fields[:2] == ['V1','V2']:
                continue
            ids.add(fields[-1] if len(fields) == 1 else (fields[0], fields[1]))
    return ids


class BedReader:
    """Memory-map SNP-major .bed; decode only requested samples and variants.

    This reader never loads the full BIM table. Blocks can be decoded on CPU,
    or transferred in their original packed representation and decoded on GPU.
    ``bim_index_path`` optionally stores a private SQLite ID index on disk.
    ``None`` keeps the original streaming lookup. Both paths retain BIM order
    and reject duplicates only when their ID is requested. SQL requests use
    bounded chunks; the positive/negative metadata cache stays bounded too.
    """
    def __init__(self, prefix, *, keep=None, remove=None, sample_ids=None,
                 metadata_cache_size=100000, bim_index_path=None,
                 bim_index_query_size=500):
        if not isinstance(metadata_cache_size, int) or metadata_cache_size < 0:
            raise ValueError('metadata_cache_size must be a nonnegative integer')
        if (isinstance(bim_index_query_size, bool) or not isinstance(bim_index_query_size, int)
                or not 1 <= bim_index_query_size <= 900):
            raise ValueError('bim_index_query_size must be an integer from 1 to 900')
        self.prefix = str(prefix)
        self.metadata_cache_size = metadata_cache_size
        self.bim_index_path = None if bim_index_path is None else Path(bim_index_path)
        if self.bim_index_path is not None:
            for suffix in ('.bed', '.bim', '.fam'):
                source = Path(str(prefix)+suffix)
                same = self.bim_index_path.resolve() == source.resolve()
                if not same and self.bim_index_path.exists() and source.exists():
                    same = os.path.samefile(self.bim_index_path, source)
                if same:
                    raise ValueError('BIM disk index must not overwrite a source BED/BIM/FAM file')
        self.bim_index_query_size = bim_index_query_size
        self._variant_metadata = OrderedDict()
        self._packed_decoder_cache = OrderedDict()
        self._bim_identity = None
        p = Path(prefix)
        if not Path(str(p)+'.bed').exists():
            raise FileNotFoundError(f'{p}.bed: use an uncompressed SNP-major BED file')
        fam = []
        sexes = []
        with open(str(p)+'.fam') as stream:
            for line in stream:
                f = line.split()
                if len(f) < 2:
                    raise ValueError('Malformed FAM row')
                fam.append((f[0], f[1]))
                sexes.append(int(f[4]) if len(f)>=5 else 0)
        if len(set(fam)) != len(fam):
            raise ValueError('Duplicate FID/IID in FAM')
        self._all_sample_ids = fam
        self._stride = (len(fam)+3)//4
        with open(str(p)+'.bed', 'rb') as stream:
            if stream.read(3) != b'\x6c\x1b\x01':
                raise ValueError('BED must have SNP-major PLINK header')
        nbytes = Path(str(p)+'.bed').stat().st_size - 3
        if nbytes % self._stride:
            raise ValueError('BED size does not match FAM')
        self.n_variants = nbytes//self._stride
        self._bed = np.memmap(str(p)+'.bed', mode='r', dtype=np.uint8,
                              offset=3, shape=(self.n_variants, self._stride))
        keep = read_sample_ids(keep) if isinstance(keep, (str, Path)) else keep
        remove = read_sample_ids(remove) if isinstance(remove, (str, Path)) else remove
        selected = [i for i, sid in enumerate(fam)
                    if (keep is None or sid in keep or sid[1] in keep)
                    and (remove is None or (sid not in remove and sid[1] not in remove))]
        if sample_ids is not None:
            positions = {fam[i]: i for i in selected}
            try:
                selected = [positions[tuple(s)] for s in sample_ids]
            except KeyError as error:
                raise ValueError('Requested sample missing from filtered FAM') from error
        self.sample_indices = np.asarray(selected, dtype=np.int64)
        # uint8 shifts keep NumPy's bitwise decoding in byte precision.  An
        # int64 shift array otherwise promotes every N x B BED code to int64.
        self._sample_bytes = self.sample_indices // 4
        self._sample_shifts = ((self.sample_indices % 4) * 2).astype(np.uint8)
        self.sample_ids = [fam[i] for i in selected]
        self.sample_sex = [sexes[i] for i in selected]
        self.n_samples = len(selected)
        if not self.n_samples:
            raise ValueError('No samples remain')
        self._variant_chromosomes = None

    def iter_variants(self) -> Iterator[Variant]:
        count = 0
        with open(self.prefix+'.bim') as stream:
            for i, line in enumerate(stream):
                f = line.split()
                if len(f) != 6:
                    raise ValueError(f'Malformed BIM row {i+1}')
                count += 1
                yield Variant(i, f[0].removeprefix('chr'), f[1], int(f[3]), f[4], f[5])
        if count != self.n_variants:
            raise ValueError('BIM variant count does not match BED')

    @property
    def variant_chromosomes(self):
        self._refresh_bim_metadata()
        if self._variant_chromosomes is None:
            self._variant_chromosomes = np.fromiter(
                (int(v.chrom) for v in self.iter_variants()), dtype=np.int16,
                count=self.n_variants)
        return self._variant_chromosomes

    def _refresh_bim_metadata(self):
        stat = Path(self.prefix+'.bim').stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size,
                    stat.st_mtime_ns, stat.st_ctime_ns)
        if identity != self._bim_identity:
            self._variant_metadata.clear()
            self._variant_chromosomes = None
            self._bim_identity = identity
        return identity

    def _bim_index_source(self, identity):
        return json.dumps({'path': str(Path(self.prefix+'.bim').resolve()),
                           'stat': identity}, sort_keys=True, separators=(',', ':'))

    def _bim_index_connection(self):
        # A reader never mutates a committed index. Concurrent builders publish
        # complete files by replacement; an open connection keeps its own inode.
        return sqlite3.connect(self.bim_index_path.resolve().as_uri()+'?mode=ro', uri=True)

    @staticmethod
    def _validated_bim_index_metadata(connection, source):
        if connection.execute('PRAGMA application_id').fetchone()[0] != _BIM_INDEX_APPLICATION_ID:
            return None
        if connection.execute('PRAGMA user_version').fetchone()[0] != _BIM_INDEX_SCHEMA:
            return None
        metadata = dict(connection.execute('SELECT name, value FROM metadata'))
        if metadata.get('source') != source or metadata.get('complete') != '1':
            return None
        rows, indexed, malformed = (int(metadata[name]) for name in
                                     ('rows', 'indexed_rows', 'first_malformed_row'))
        if (rows < 0 or not 0 <= indexed <= rows
                or not (malformed == -1 or 0 <= malformed < rows)):
            return None
        connection.execute('SELECT ordinal, chrom, identifier, position, allele1, allele0 FROM variants LIMIT 0')
        return rows, indexed, malformed

    def _read_bim_index_metadata(self, source):
        if not self.bim_index_path.is_file():
            return None
        try:
            connection = self._bim_index_connection()
            try:
                return self._validated_bim_index_metadata(connection, source)
            finally:
                connection.close()
        except (sqlite3.DatabaseError, ValueError, KeyError, OSError):
            # An interrupted/unrelated index is rebuilt, never used as a proof
            # that a requested ID is absent from the original BIM.
            return None

    def _build_bim_index(self, identity, source):
        path = self.bim_index_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.partial', dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        connection = None
        try:
            connection = sqlite3.connect(str(temporary))
            # Only an unpublished private temp file uses these write settings.
            # A failed build cannot alter an existing committed database.
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('PRAGMA synchronous=OFF')
            connection.execute('PRAGMA temp_store=FILE')
            connection.execute('PRAGMA cache_size=-8192')
            connection.execute(f'PRAGMA application_id={_BIM_INDEX_APPLICATION_ID}')
            connection.execute(f'PRAGMA user_version={_BIM_INDEX_SCHEMA}')
            connection.execute('CREATE TABLE variants (ordinal INTEGER PRIMARY KEY, chrom TEXT NOT NULL, '
                               'identifier TEXT NOT NULL, position TEXT NOT NULL, allele1 TEXT NOT NULL, allele0 TEXT NOT NULL)')
            connection.execute('CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
            connection.execute('BEGIN')
            rows, indexed, malformed, batch = 0, 0, -1, []
            with open(self.prefix+'.bim') as stream:
                for ordinal, line in enumerate(stream):
                    rows += 1
                    fields = line.split()
                    if len(fields) != 6:
                        # A lookup's first failure must stay in original row
                        # order. In particular a later malformed, unrequested
                        # row must not hide an earlier requested duplicate.
                        if malformed < 0:
                            malformed = ordinal
                        continue
                    batch.append((ordinal, fields[0].removeprefix('chr'), fields[1],
                                  fields[3], fields[4], fields[5]))
                    indexed += 1
                    if len(batch) == _BIM_INDEX_INSERT_ROWS:
                        connection.executemany('INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?)', batch)
                        batch.clear()
                if batch:
                    connection.executemany('INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?)', batch)
            # ID is deliberately non-unique: duplicate unrequested IDs were
            # legal in the old lookup and must remain legal in this one.
            connection.execute('CREATE INDEX variants_identifier ON variants(identifier)')
            connection.executemany('INSERT INTO metadata VALUES (?, ?)',
                [('source', source), ('complete', '1'), ('rows', str(rows)),
                 ('indexed_rows', str(indexed)), ('first_malformed_row', str(malformed))])
            connection.commit()
            connection.close()
            connection = None
            if self._refresh_bim_metadata() != identity:
                raise RuntimeError('BIM changed while building its disk index')
            # Persist the completed file before the atomic publication. The
            # database and any SQLite temporary pages contain private metadata.
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return rows, indexed, malformed
        finally:
            if connection is not None:
                connection.close()
            temporary.unlink(missing_ok=True)

    def prepare_bim_index(self):
        """Build/reuse the optional disk index; return anonymous preparation timing.

        Metadata binds the canonical BIM path and device/inode/size/mtime/ctime
        identity. The build streams rows in small batches, and atomically
        publishes a complete private file. No ID table is retained in RAM.
        Positions remain text until requested, matching the streaming parser.
        A malformed row is recorded so lookup errors retain original order;
        ``valid_bim`` checks row width only; requested positions are parsed by
        lookup. This report does not export the input's identifiers or paths.
        """
        start = time.perf_counter()
        if self.bim_index_path is None:
            return {'enabled': False, 'built': False, 'rows': None,
                    'indexed_rows': None, 'valid_bim': None, 'seconds': time.perf_counter()-start}
        identity = self._refresh_bim_metadata()
        source = self._bim_index_source(identity)
        built = False
        # Shared filesystems can return unchanged stat fields after an in-place
        # SQLite update. They do not prove the schema/source/completion record
        # is unchanged. Reopen the tiny metadata table on every prepare; this
        # never scans the BIM or loads the disk index into process memory.
        metadata = self._read_bim_index_metadata(source)
        if metadata is None:
            metadata = self._build_bim_index(identity, source)
            built = True
        if self._refresh_bim_metadata() != identity:
            raise RuntimeError('BIM changed while preparing its disk index')
        rows, indexed, malformed = metadata
        return {'enabled': True, 'built': built, 'rows': rows, 'indexed_rows': indexed,
                'valid_bim': malformed < 0, 'seconds': time.perf_counter()-start}

    def _find_indexed_variants(self, unknown, identity):
        self.prepare_bim_index()
        connection = self._bim_index_connection()
        found = {}
        try:
            # Validate the actual query connection too: between prepare and
            # this open, another builder may atomically replace the database.
            metadata = self._validated_bim_index_metadata(connection, self._bim_index_source(identity))
            if metadata is None:
                raise RuntimeError('BIM disk index changed while opening requested metadata')
            _, _, malformed = metadata
            error_ordinal = None if malformed < 0 else malformed
            error = None if malformed < 0 else ValueError(f'Malformed BIM row {malformed+1}')
            # SQLite TEXT affinity can equate an integer 12 with text '12'.
            # The streaming reader compares string IDs in Python, so ignore
            # non-string queries rather than widening that public equality.
            identifiers = (identifier for identifier in unknown if isinstance(identifier, str))
            while True:
                batch = list(itertools.islice(identifiers, self.bim_index_query_size))
                if not batch:
                    break
                slots = ','.join('?' for _ in batch)
                query = ('SELECT ordinal, chrom, identifier, position, allele1, allele0 '
                         f'FROM variants WHERE identifier IN ({slots})')
                arguments = batch
                if malformed >= 0:
                    query += ' AND ordinal < ?'
                    arguments = [*batch, malformed]
                cursor = connection.execute(query+' ORDER BY ordinal', arguments)
                while True:
                    rows = cursor.fetchmany(self.bim_index_query_size)
                    if not rows:
                        break
                    for ordinal, chrom, identifier, position, allele1, allele0 in rows:
                        if error_ordinal is not None and ordinal >= error_ordinal:
                            continue
                        try:
                            if identifier in found:
                                raise ValueError(f'Duplicate requested variant ID: {identifier}')
                            found[identifier] = Variant(ordinal, chrom, identifier, int(position), allele1, allele0)
                        except ValueError as failure:
                            error_ordinal, error = ordinal, failure
                cursor.close()
            if self._refresh_bim_metadata() != identity:
                raise RuntimeError('BIM changed while reading requested variant metadata')
            if error is not None:
                raise error
            return found
        finally:
            connection.close()

    def find_variants(self, ids: Iterable[str]):
        """Find requested IDs, caching only bounded positive/negative queries.

        A cached subset avoids a BIM rescan. With a disk index, new IDs use
        bounded SQL queries; otherwise they use the original complete scan.
        Neither path retains a full BIM index or genotype array in RAM.
        """
        needed = set(ids)
        identity = self._refresh_bim_metadata()
        unknown = needed.difference(self._variant_metadata)
        result = {identifier: self._variant_metadata[identifier]
                  for identifier in needed.intersection(self._variant_metadata)
                  if self._variant_metadata[identifier] is not None}
        if unknown:
            if self.bim_index_path is None:
                found = {}
                with open(self.prefix+'.bim') as stream:
                    for i,line in enumerate(stream):
                        f = line.split()
                        if len(f) != 6: raise ValueError(f'Malformed BIM row {i+1}')
                        if f[1] in unknown:
                            if f[1] in found: raise ValueError(f'Duplicate requested variant ID: {f[1]}')
                            found[f[1]] = Variant(i,f[0].removeprefix('chr'),f[1],int(f[3]),f[4],f[5])
            else:
                found = self._find_indexed_variants(unknown, identity)
            if self._refresh_bim_metadata() != identity:
                raise RuntimeError('BIM changed while reading requested variant metadata')
            result.update(found)
            if self.metadata_cache_size:
                for identifier, variant in found.items():
                    self._variant_metadata[identifier] = variant
                    while len(self._variant_metadata) > self.metadata_cache_size:
                        self._variant_metadata.popitem(last=False)
                for identifier in sorted(unknown.difference(found)):
                    self._variant_metadata[identifier] = None
                    while len(self._variant_metadata) > self.metadata_cache_size:
                        self._variant_metadata.popitem(last=False)
        # Preserve the previous public ordering: matching variants in BIM order.
        result = dict(sorted(result.items(), key=lambda item: item[1].index))
        for identifier in result:
            if identifier in self._variant_metadata:
                self._variant_metadata.move_to_end(identifier)
        for identifier in sorted(needed.intersection(self._variant_metadata)):
            if self._variant_metadata[identifier] is None:
                self._variant_metadata.move_to_end(identifier)
        while len(self._variant_metadata) > self.metadata_cache_size:
            self._variant_metadata.popitem(last=False)
        return result

    def read_variants(self, indices):
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(idx < 0) or np.any(idx >= self.n_variants):
            raise IndexError('Variant index outside BED')
        raw = self._bed[idx[:, None], self._sample_bytes[None, :]]
        codes = (raw >> self._sample_shifts[None, :]) & np.uint8(3)
        # PLINK 00=A1/A1; 01=missing; 10=A1/A2; 11=A2/A2.
        dosage = np.asarray([2., np.nan, 1., 0.], dtype=np.float32)[codes]
        return torch.from_numpy(dosage.T.copy())

    def read_packed_block(self, indices, *, sample_rows=None, device='cuda',
                          dtype=torch.float32):
        """Upload original BED bytes once for counts and selected-site decoding.

        The transferred block has B x ceil(N_source/4) bytes, rather than an
        expanded N_analysis x B float matrix.  Optional sample_rows refers to
        this reader's already filtered sample order, preserving arbitrary order.
        """
        import hashlib
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(idx < 0) or np.any(idx >= self.n_variants):
            raise IndexError('Variant index outside BED')
        if sample_rows is None:
            selection = self.sample_indices
        else:
            rows = np.asarray(sample_rows, dtype=np.int64).reshape(-1)
            if np.any(rows < 0) or np.any(rows >= self.n_samples):
                raise IndexError('Sample row outside filtered FAM')
            selection = self.sample_indices[rows]
        target_device = torch.device(device)
        if target_device.type == 'cuda' and target_device.index is None:
            target_device = torch.device('cuda',torch.cuda.current_device())
        target_dtype = getattr(torch,dtype) if isinstance(dtype,str) else dtype
        if target_dtype not in (torch.float32,torch.float64):
            raise ValueError('Packed BED decoding supports float32 and float64')
        selection_key = hashlib.sha256(selection.tobytes()).digest()
        cache_key = (str(target_device),target_dtype,selection_key)
        cached = self._packed_decoder_cache.get(cache_key)
        if cached is None:
            sample_bytes = torch.as_tensor(selection//4,device=target_device,dtype=torch.long)
            sample_shifts = torch.as_tensor(((selection%4)*2).astype(np.uint8),device=target_device)
            lookup = torch.tensor([2.,float('nan'),1.,0.],device=target_device,dtype=target_dtype)
            cached = (sample_bytes,sample_shifts,lookup)
            self._packed_decoder_cache[cache_key] = cached
            while len(self._packed_decoder_cache)>4:
                self._packed_decoder_cache.popitem(last=False)
        self._packed_decoder_cache.move_to_end(cache_key)
        sample_bytes,sample_shifts,lookup = cached
        # Integer-array indexing copies only original packed rows into an owned,
        # contiguous CPU buffer.  No expanded participant matrix exists on CPU.
        raw = np.ascontiguousarray(self._bed[idx,:])
        packed = torch.from_numpy(raw).to(target_device)
        return PackedBedBlock(packed, sample_bytes, sample_shifts, lookup)

    def read_packed_variants(self, indices, *, sample_rows=None, device='cuda',
                             dtype=torch.float32):
        """Decode an uploaded BED block directly into sample-major genotypes."""
        return self.read_packed_block(indices, sample_rows=sample_rows,
                                      device=device, dtype=dtype).decode()

    def iter_blocks(self, block_size=1000, indices=None):
        if block_size < 1:
            raise ValueError('block_size must be positive')
        selected = range(self.n_variants) if indices is None else indices
        chunk = []
        for i in selected:
            chunk.append(int(i))
            if len(chunk) == block_size:
                yield torch.tensor(chunk, dtype=torch.int64), self.read_variants(chunk)
                chunk = []
        if chunk:
            yield torch.tensor(chunk, dtype=torch.int64), self.read_variants(chunk)

    def _iter_variant_metadata_blocks(self, block_size, indices):
        if block_size<1:
            raise ValueError('block_size must be positive')
        selected = None if indices is None else set(map(int, indices))
        chunk = []
        for v in self.iter_variants():
            if selected is not None and v.index not in selected:
                continue
            chunk.append(v)
            if len(chunk) == block_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def iter_packed_variant_blocks(self, block_size=1000, indices=None, *,
                                   sample_rows=None, device='cuda', dtype=torch.float32):
        """Stream BIM metadata with uploaded blocks, without expanding hardcalls."""
        for variants in self._iter_variant_metadata_blocks(block_size, indices):
            yield variants, self.read_packed_block([v.index for v in variants],
                sample_rows=sample_rows, device=device, dtype=dtype)

    def iter_variant_blocks(self, block_size=1000, indices=None, *,
                            genotype_reader='cpu', sample_rows=None, device='cuda',
                            dtype=torch.float32):
        if block_size < 1:
            raise ValueError('block_size must be positive')
        if genotype_reader not in ('cpu','cuda_packed'):
            raise ValueError('genotype_reader must be cpu or cuda_packed')
        for variants in self._iter_variant_metadata_blocks(block_size, indices):
            indices = [v.index for v in variants]
            if genotype_reader == 'cuda_packed':
                values = self.read_packed_variants(indices, sample_rows=sample_rows,
                                                   device=device, dtype=dtype)
            else:
                values = self.read_variants(indices)
                if sample_rows is not None:
                    values = values[sample_rows]
            yield variants, values


def load_phenotype(path, column, sample_ids, *, missing_values=(-9,)):
    import pandas as pd
    table = pd.read_csv(path, sep=r'\s+', dtype={'FID': str, 'IID': str, '#FID': str},
                        usecols=lambda c: c in ('FID', '#FID', 'IID', column))
    table.rename(columns={'#FID': 'FID'}, inplace=True)
    if not {'FID', 'IID', column}.issubset(table.columns):
        raise ValueError('Phenotype requires FID, IID and the requested column')
    if table.duplicated(['FID', 'IID']).any():
        raise ValueError('Duplicate phenotype FID/IID')
    table[column] = pd.to_numeric(table[column], errors='coerce')
    table.loc[table[column].isin(missing_values), column] = np.nan
    values = table.set_index(['FID', 'IID'])[column].reindex(sample_ids).to_numpy(dtype=np.float64)
    return torch.from_numpy(values)


def resolve_variant_include(reader, path):
    ids = {line.strip().split()[0] for line in open(path) if line.strip()}
    return np.fromiter((v.index for v in reader.iter_variants() if v.id in ids), dtype=np.int64)


def write_bed(prefix, genotypes, variants, sample_ids):
    """Write a real hardcall subset or optional gene masks in PLINK format."""
    prefix = str(prefix)
    g = torch.as_tensor(genotypes).detach().cpu().numpy()
    if g.shape != (len(sample_ids),len(variants)):
        raise ValueError('BED output genotype dimensions differ from samples/variants')
    finite = np.isfinite(g)
    if np.any(finite & ((g < 0)|(g > 2)|(g != np.rint(g)))):
        raise ValueError('BED supports 0/1/2 hardcalls; do not silently round dosages')
    stride = (len(sample_ids)+3)//4
    with open(prefix+'.bed','wb') as stream:
        stream.write(b'\x6c\x1b\x01')
        for j in range(len(variants)):
            padded = np.zeros(stride*4,dtype=np.uint8)
            values = g[:,j]
            codes = np.full(len(values),1,dtype=np.uint8)
            codes[np.isfinite(values)&(values==2)] = 0
            codes[np.isfinite(values)&(values==1)] = 2
            codes[np.isfinite(values)&(values==0)] = 3
            padded[:len(values)] = codes
            packed = (padded.reshape(-1,4) << np.array([0,2,4,6],dtype=np.uint8)).sum(1).astype(np.uint8)
            stream.write(packed.tobytes())
    with open(prefix+'.bim','w') as stream:
        for v in variants: stream.write(f'{v.chrom}\t{v.id}\t0\t{v.position}\t{v.allele1}\t{v.allele0}\n')
    with open(prefix+'.fam','w') as stream:
        for fid,iid in sample_ids: stream.write(f'{fid} {iid} 0 0 0 -9\n')


def materialize_bed(prefix, cache_directory):
    """Expand .bed.gz into a private cache without changing the source files."""
    import gzip, hashlib, json, shutil, fcntl
    source=Path(prefix)
    if Path(str(source)+'.bed').exists(): return str(source)
    compressed=Path(str(source)+'.bed.gz')
    if not compressed.exists(): raise FileNotFoundError(str(source)+'.bed[.gz]')
    directory=Path(cache_directory)/hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/source.name
    identity={suffix:[Path(str(source)+suffix).stat().st_size,Path(str(source)+suffix).stat().st_mtime_ns]
              for suffix in ('.bed.gz','.bim','.fam')}
    marker=directory/'source.json'
    with (directory/'materialize.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:cached_identity=json.loads(marker.read_text())
        except (OSError,ValueError):cached_identity=None
        if not (Path(str(target)+'.bed').exists() and cached_identity==identity):
            with open(str(source)+'.fam') as f:n_samples=sum(1 for line in f if line.strip())
            n_variants=0
            with open(str(source)+'.bim','rb') as f:
                while True:
                    chunk=f.read(16*1024**2)
                    if not chunk:break
                    n_variants+=chunk.count(b'\n')
            # Accept a final BIM line without newline.
            with open(str(source)+'.bim','rb') as f:
                f.seek(-1,2)
                if f.read(1)!=b'\n':n_variants+=1
            expected=3+((n_samples+3)//4)*n_variants
            if shutil.disk_usage(directory).free < expected+1024**3:
                raise OSError(f'Insufficient cache space for {expected} bytes of decompressed BED')
            partial=Path(str(target)+'.bed.partial')
            try:
                with gzip.open(compressed,'rb') as src,partial.open('wb') as dst:
                    shutil.copyfileobj(src,dst,length=16*1024**2)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            if partial.stat().st_size!=expected:
                partial.unlink()
                raise ValueError('Expanded BED size does not match BIM/FAM')
            partial.replace(str(target)+'.bed')
            marker.write_text(json.dumps(identity))
        for suffix in ('.bim','.fam'):
            link=Path(str(target)+suffix)
            if link.is_symlink() or link.exists():link.unlink()
            link.symlink_to(Path(str(source)+suffix).resolve())
    return str(target)


def materialize_discovery_bed(prefix, cache_directory, *, keep=None, remove=None,
                              sample_ids=None, block_variants=1000):
    """Stream all BED variants into a private, sample-filtered BED cache.

    A compressed source is read once in physical variant order.  Its original
    two-bit calls are copied exactly, including missing calls; no dosage
    conversion or variant selection is performed.  Source FAM lines retain
    their sex/pedigree fields.  The returned BIM links to the entire source BIM.
    """
    import gzip, hashlib, json, shutil, fcntl, os
    if not isinstance(block_variants, int) or block_variants < 1:
        raise ValueError('block_variants must be a positive integer')
    source = Path(prefix)
    bed = Path(str(source)+'.bed')
    if not bed.exists():
        bed = Path(str(source)+'.bed.gz')
    if not bed.exists():
        raise FileNotFoundError(str(source)+'.bed[.gz]')
    fam = []
    fam_lines = []
    with open(str(source)+'.fam') as stream:
        for line in stream:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError('Malformed FAM row')
            fam.append((fields[0], fields[1]))
            fam_lines.append(line if line.endswith('\n') else line+'\n')
    if not fam or len(set(fam)) != len(fam):
        raise ValueError('Empty FAM or duplicate FID/IID in FAM')
    keep = read_sample_ids(keep) if isinstance(keep, (str, Path)) else keep
    remove = read_sample_ids(remove) if isinstance(remove, (str, Path)) else remove
    selected = [i for i, sid in enumerate(fam)
                if (keep is None or sid in keep or sid[1] in keep)
                and (remove is None or (sid not in remove and sid[1] not in remove))]
    if sample_ids is not None:
        positions = {fam[i]: i for i in selected}
        try:
            selected = [positions[tuple(sid)] for sid in sample_ids]
        except KeyError as error:
            raise ValueError('Requested sample missing from filtered FAM') from error
        if len(set(selected)) != len(selected):
            raise ValueError('Duplicate requested FID/IID')
    if not selected:
        raise ValueError('No samples remain')
    input_stride = (len(fam)+3)//4
    output_stride = (len(selected)+3)//4
    n_variants = 0
    bim = Path(str(source)+'.bim')
    with bim.open('rb') as stream:
        last_byte = b''
        while True:
            chunk = stream.read(16*1024**2)
            if not chunk:
                break
            n_variants += chunk.count(b'\n')
            last_byte = chunk[-1:]
        if last_byte and last_byte != b'\n':
            n_variants += 1
    if not n_variants:
        raise ValueError('Empty BIM')
    selection = np.asarray(selected, dtype=np.int64)
    selection_sha = hashlib.sha256(selection.astype('<i8', copy=False).tobytes()).hexdigest()
    key = hashlib.sha256((str(source.resolve())+'\0'+selection_sha).encode()).hexdigest()[:20]
    directory = Path(cache_directory)/('discovery_'+key)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/(source.name+'_discovery')
    marker = directory/'source.json'
    expected = 3+output_stride*n_variants
    source_paths = [bed, bim, Path(str(source)+'.fam')]
    identity = {'format_version':1, 'source':str(source.resolve()),
                'source_files':{path.suffix:[path.stat().st_size,path.stat().st_mtime_ns]
                                for path in source_paths},
                'sample_selection_sha256':selection_sha,
                'n_source_samples':len(fam), 'n_discovery_samples':len(selected),
                'n_variants':n_variants, 'expected_bed_bytes':expected}
    target_bed = Path(str(target)+'.bed')
    with (directory/'materialize.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached_identity = json.loads(marker.read_text())
        except (OSError, ValueError):
            cached_identity = None
        cached = (cached_identity == identity and target_bed.is_file()
                  and target_bed.stat().st_size == expected
                  and Path(str(target)+'.fam').is_file())
        if not cached:
            if shutil.disk_usage(directory).free < expected+1024**3:
                raise OSError(f'Insufficient cache space for {expected} bytes of discovery BED')
            partial_bed = Path(str(target_bed)+'.partial')
            partial_fam = Path(str(target)+'.fam.partial')
            partial_marker = Path(str(marker)+'.partial')
            sample_bytes = selection//4
            sample_shifts = ((selection % 4)*2).astype(np.uint8)
            open_source = gzip.open if bed.name.endswith('.gz') else open
            written = 0
            try:
                with open_source(bed, 'rb') as src, partial_bed.open('wb') as dst:
                    if src.read(3) != b'\x6c\x1b\x01':
                        raise ValueError('BED must have SNP-major PLINK header')
                    dst.write(b'\x6c\x1b\x01')
                    while True:
                        chunk = src.read(input_stride*block_variants)
                        if not chunk:
                            break
                        if len(chunk) % input_stride:
                            raise ValueError('Source BED size does not match source FAM')
                        raw = np.frombuffer(chunk, dtype=np.uint8).reshape(-1,input_stride)
                        codes = (raw[:,sample_bytes] >> sample_shifts[None,:]) & np.uint8(3)
                        packed = np.zeros((len(raw),output_stride), dtype=np.uint8)
                        for offset in range(4):
                            part = codes[:,offset::4]
                            packed[:,:part.shape[1]] |= part << np.uint8(2*offset)
                        dst.write(packed.tobytes())
                        written += len(raw)
                    dst.flush()
                    os.fsync(dst.fileno())
                if written != n_variants or partial_bed.stat().st_size != expected:
                    raise ValueError('Source BED variant count differs from complete BIM')
                # Reading gzip through EOF validates its CRC before committing.
                current_files = {path.suffix:[path.stat().st_size,path.stat().st_mtime_ns]
                                 for path in source_paths}
                if current_files != identity['source_files']:
                    raise RuntimeError('Source genotype files changed while streaming discovery BED')
                partial_fam.write_text(''.join(fam_lines[i] for i in selected))
                partial_marker.write_text(json.dumps(identity,indent=2)+'\n')
                partial_bed.replace(target_bed)
                partial_fam.replace(str(target)+'.fam')
                partial_marker.replace(marker)
            except BaseException:
                partial_bed.unlink(missing_ok=True)
                partial_fam.unlink(missing_ok=True)
                partial_marker.unlink(missing_ok=True)
                raise
        target_bim = Path(str(target)+'.bim')
        if target_bim.is_symlink() or target_bim.exists():
            target_bim.unlink()
        target_bim.symlink_to(bim.resolve())
    return str(target)
