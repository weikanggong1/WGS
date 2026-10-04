"""Streaming PLINK BED reader. Dosages count BIM A1 (the default REGENIE ALT)."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np
import torch


@dataclass(frozen=True)
class Variant:
    index: int
    chrom: str
    id: str
    position: int
    allele1: str
    allele0: str


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

    This reader never loads the full BIM table. BED decoding is I/O work on CPU;
    association matrix operations are performed by the PyTorch CUDA kernels.
    """
    def __init__(self, prefix, *, keep=None, remove=None, sample_ids=None,
                 metadata_cache_size=100000):
        if not isinstance(metadata_cache_size, int) or metadata_cache_size < 0:
            raise ValueError('metadata_cache_size must be a nonnegative integer')
        self.prefix = str(prefix)
        self.metadata_cache_size = metadata_cache_size
        self._variant_metadata = OrderedDict()
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

    def find_variants(self, ids: Iterable[str]):
        """Find requested IDs, caching only bounded positive/negative queries.

        A cached subset avoids a BIM rescan. Newly requested IDs still require
        a complete scan, preserving detection of duplicate requested IDs.
        No full BIM index or genotype array is retained by this cache.
        """
        needed = set(ids)
        identity = self._refresh_bim_metadata()
        unknown = needed.difference(self._variant_metadata)
        result = {identifier: self._variant_metadata[identifier]
                  for identifier in needed.intersection(self._variant_metadata)
                  if self._variant_metadata[identifier] is not None}
        if unknown:
            found = {}
            with open(self.prefix+'.bim') as stream:
                for i,line in enumerate(stream):
                    f = line.split()
                    if len(f) != 6: raise ValueError(f'Malformed BIM row {i+1}')
                    if f[1] in unknown:
                        if f[1] in found: raise ValueError(f'Duplicate requested variant ID: {f[1]}')
                        found[f[1]] = Variant(i,f[0].removeprefix('chr'),f[1],int(f[3]),f[4],f[5])
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
        raw = self._bed[idx[:, None], (self.sample_indices//4)[None, :]]
        codes = (raw >> ((self.sample_indices % 4)*2)[None, :]) & 3
        # PLINK 00=A1/A1; 01=missing; 10=A1/A2; 11=A2/A2.
        dosage = np.asarray([2., np.nan, 1., 0.], dtype=np.float32)[codes]
        return torch.from_numpy(dosage.T.copy())

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

    def iter_variant_blocks(self, block_size=1000, indices=None):
        selected = None if indices is None else set(map(int, indices))
        chunk = []
        for v in self.iter_variants():
            if selected is not None and v.index not in selected:
                continue
            chunk.append(v)
            if len(chunk) == block_size:
                yield chunk, self.read_variants([v.index for v in chunk])
                chunk = []
        if chunk:
            yield chunk, self.read_variants([v.index for v in chunk])


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
