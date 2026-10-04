"""Single quantitative-trait REGENIE Step 1, computed with PyTorch.

Independent mathematical implementation of the default 5-fold algorithm in
REGENIE 3.4.1. Genotypes are streamed by chromosome; level-0 held-out predictions
can live in a disk memmap. No REGENIE executable is used by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
import contextlib
import gzip
import hashlib
import json
import shutil
import tempfile
import time

import numpy as np
import torch


@dataclass(frozen=True)
class Step1Config:
    block_size: int = 1000
    folds: int = 5
    ridge_l0: tuple[float, ...] = (0.01, 0.25, 0.5, 0.75, 0.99)
    ridge_l1: tuple[float, ...] = (0.01, 0.25, 0.5, 0.75, 0.99)
    dtype: str = "float32"
    device: str = "cuda"
    apply_rint: bool = True
    preserve_masked_rows: bool = True
    tf32: bool = True
    l0_storage: str = "memmap"
    keep_l0: bool = False
    sample_chunk_size: int = 4096
    max_gpu_gb: float = 20.0
    variance_tolerance: float = 1e-6
    covariate_eigen_tolerance: float = 1e-15
    output_chromosomes: tuple[int, ...] = tuple(range(1, 23))

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.folds < 2 or self.sample_chunk_size < 1:
            raise ValueError("block_size/sample_chunk_size must be positive and folds >= 2")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("Only float32 and float64 are supported; float16 is not supported")
        if self.l0_storage not in ("memmap", "memory"):
            raise ValueError("l0_storage must be 'memmap' or 'memory'")
        for name in ("ridge_l0", "ridge_l1"):
            values = getattr(self, name)
            if len(values) < 2 or any(not 0 < value < 1 for value in values):
                raise ValueError(f"{name} needs >= 2 heritability values strictly between 0 and 1")
        if self.max_gpu_gb <= 0 or self.variance_tolerance <= 0:
            raise ValueError("Memory budget and variance_tolerance must be positive")
        if not 0 < self.covariate_eigen_tolerance < 1:
            raise ValueError("covariate_eigen_tolerance must be strictly between 0 and 1")
        if not self.output_chromosomes or len(set(self.output_chromosomes)) != len(self.output_chromosomes):
            raise ValueError("output_chromosomes must be nonempty and unique")
        if any(chrom not in range(1, 23) for chrom in self.output_chromosomes):
            raise ValueError("This quantitative-trait implementation supports autosomes 1-22")


def _ids(values: Sequence[Any], n: int) -> list[tuple[str, str]]:
    if len(values) != n:
        raise ValueError("sample_ids length does not match genotype samples")
    result = []
    for value in values:
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            result.append((str(value[0]), str(value[1])))
        else:
            # An IID alone is supported explicitly by assigning the same FID.
            result.append((str(value), str(value)))
    if len(set(result)) != n:
        raise ValueError("Duplicate FID/IID pairs are not allowed")
    if any(any(char.isspace() for char in fid + iid) for fid, iid in result):
        raise ValueError("FID/IID values cannot contain whitespace")
    return result


@contextlib.contextmanager
def _atomic_outputs(targets: Sequence[Path]):
    """Prepare sibling files and restore existing outputs if a commit fails."""
    temporary, backups, committed = [], {}, []
    retain_backups = False
    try:
        for target in targets:
            descriptor = tempfile.NamedTemporaryFile(
                prefix=target.name + ".", suffix=".partial", dir=target.parent, delete=False)
            temporary.append(Path(descriptor.name))
            descriptor.close()
        yield temporary
        # Prepare every rollback copy before replacing any completed output.
        for target in targets:
            if target.exists():
                descriptor = tempfile.NamedTemporaryFile(
                    prefix=target.name + ".", suffix=".backup", dir=target.parent, delete=False)
                backup = Path(descriptor.name)
                descriptor.close()
                backups[target] = backup
                shutil.copy2(target, backup)
            else:
                backups[target] = None
        try:
            for partial, target in zip(temporary, targets):
                partial.replace(target)
                committed.append(target)
        except BaseException:
            try:
                for target in reversed(committed):
                    backup = backups[target]
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        backup.replace(target)
            except BaseException as recovery_error:
                retain_backups = True
                raise RuntimeError("Output rollback failed; recovery backups were retained") from recovery_error
            raise
    finally:
        for partial in temporary:
            partial.unlink(missing_ok=True)
        if not retain_backups:
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)


@dataclass
class NullModel:
    sample_ids: list[tuple[str, str]]
    loco: torch.Tensor
    chromosomes: tuple[int, ...] = tuple(range(1, 23))
    y_scale: float = 1.0
    sample_indices: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.loco = torch.as_tensor(self.loco).detach().cpu().contiguous()
        if self.loco.ndim != 2:
            raise ValueError("LOCO must be a samples x chromosomes matrix")
        self.sample_ids = _ids(self.sample_ids, self.loco.shape[0])
        self.chromosomes = tuple(int(chrom) for chrom in self.chromosomes)
        if self.loco.ndim != 2 or self.loco.shape[1] != len(self.chromosomes):
            raise ValueError("LOCO dimensions must be samples x chromosomes")
        if len(set(self.chromosomes)) != len(self.chromosomes):
            raise ValueError("Duplicate LOCO chromosomes")
        if not torch.isfinite(self.loco).all():
            raise ValueError("LOCO predictions must be finite")
        if not np.isfinite(self.y_scale) or self.y_scale <= 0:
            raise ValueError("y_scale must be finite and positive")
        if self.sample_indices is None:
            self.sample_indices = torch.arange(len(self.sample_ids), dtype=torch.long)
        else:
            self.sample_indices = torch.as_tensor(self.sample_indices, dtype=torch.long).cpu()
        if self.sample_indices.shape != (len(self.sample_ids),):
            raise ValueError("sample_indices must contain one index per LOCO sample")
        if self.prs is not None:
            self.prs = torch.as_tensor(self.prs).detach().cpu().reshape(-1)
            if self.prs.shape != (len(self.sample_ids),) or not torch.isfinite(self.prs).all():
                raise ValueError("prs must have one finite prediction per LOCO sample")

    def chromosome_prediction(self, chromosome: int) -> torch.Tensor:
        return self.loco[:, self.chromosomes.index(int(chromosome))]

    def align(self, sample_ids: Sequence[Any]) -> torch.Tensor:
        """Return predictions in requested FID/IID order; missing IDs are errors."""
        requested = _ids(sample_ids, len(sample_ids))
        lookup = {sample: index for index, sample in enumerate(self.sample_ids)}
        try:
            indices = [lookup[sample] for sample in requested]
        except KeyError as error:
            raise ValueError("A requested sample has no LOCO prediction") from error
        return self.loco[indices]

    def save(self, path: str | Path) -> Path:
        """Save a tensor-only .pt payload and readable parameter metadata."""
        target = Path(path)
        if target.suffix != ".pt":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "null_model.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_ids": self.sample_ids, "loco": self.loco,
            "chromosomes": self.chromosomes, "y_scale": self.y_scale,
            "sample_indices": self.sample_indices, "metadata": self.metadata, "prs": self.prs,
        }
        metadata_target = target.with_suffix(".json")
        with _atomic_outputs([target, metadata_target]) as (model_partial, metadata_partial):
            torch.save(payload, model_partial)
            metadata_partial.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NullModel":
        target = Path(path)
        if target.is_dir():
            target = target / "null_model.pt"
        try:
            payload = torch.load(target, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch 2.0 compatibility; only load trusted local files.
            payload = torch.load(target, map_location="cpu")
        return cls(**payload)

    def export_regenie(self, prefix: str | Path, *, phenotype_name: str | None = None,
                       sort_ids: bool = True, pheno_index: int = 1,
                       n_chromosomes: int = 23, compressed: bool = False) -> tuple[Path, Path]:
        """Write REGENIE-compatible .loco and _pred.list without rescaling LOCO."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        name = phenotype_name or self.metadata.get("phenotype_name", "phenotype")
        if not name or any(char.isspace() for char in name):
            raise ValueError("phenotype_name must be one nonempty token")
        tokens = [f"{fid}_{iid}" for fid, iid in self.sample_ids]
        if len(set(tokens)) != len(tokens):
            raise ValueError("FID_IID concatenation is ambiguous for these sample IDs")
        order = sorted(range(len(tokens)), key=tokens.__getitem__) if sort_ids else list(range(len(tokens)))
        if pheno_index < 1 or n_chromosomes < 1:
            raise ValueError("pheno_index and n_chromosomes must be positive")
        loco_path = Path(str(prefix) + f"_{pheno_index}.loco" + (".gz" if compressed else ""))
        list_path = Path(str(prefix) + "_pred.list")
        opener = gzip.open if compressed else open
        with _atomic_outputs([loco_path, list_path]) as (loco_partial, list_partial):
            with opener(loco_partial, "wt", encoding="utf-8") as handle:
                # C++ std::map ID order, default stream precision and trailing spaces.
                handle.write("FID_IID " + " ".join(tokens[index] for index in order) + " \n")
                for chromosome in range(1, n_chromosomes + 1):
                    if chromosome in self.chromosomes:
                        prediction = self.chromosome_prediction(chromosome)
                    elif self.prs is not None:
                        prediction = self.prs
                    elif len(self.chromosomes) > 1:
                        prediction = self.loco.sum(1) / (len(self.chromosomes) - 1)
                    else:
                        raise ValueError("Full PRS is needed to export a chromosome with no input markers")
                    handle.write(str(chromosome) + " " + " ".join(
                        format(float(prediction[index]), ".6g") for index in order) + " \n")
            list_partial.write_text(f"{name} {loco_path.resolve()}\n", encoding="utf-8")
        return loco_path, list_path

    @classmethod
    def from_regenie(cls, path: str | Path, *, phenotype_name: str = "phenotype",
                     sample_ids: Sequence[Any] | None = None,
                     chromosomes: Sequence[int] | None = None,
                     y_scale: float = 1.0) -> "NullModel":
        """Import a .loco[.gz] or _pred.list, optionally align supplied sample IDs.

        Imported predictions are already in the source Step1 standardized units.
        y_scale is provenance only and must not multiply the imported predictions.
        """
        source = Path(path)
        if source.name.endswith(".list"):
            matches = [line.split(maxsplit=1) for line in source.read_text().splitlines() if line.strip()]
            matches = [parts for parts in matches if len(parts) == 2 and parts[0] == phenotype_name]
            if len(matches) != 1:
                raise ValueError("Prediction list must contain exactly one matching phenotype")
            source_file = Path(matches[0][1])
            source = source_file if source_file.is_absolute() else source.parent / source_file
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as handle:
            header = handle.readline().split()
            if not header or header[0] != "FID_IID" or len(header) < 2:
                raise ValueError("Expected REGENIE FID_IID header")
            tokens = header[1:]
            if len(set(tokens)) != len(tokens):
                raise ValueError("Duplicate IDs in LOCO header")
            rows = {}
            for line in handle:
                if not line.strip():
                    continue
                parts = line.split()
                chromosome = int(parts[0])
                if chromosome in rows or len(parts) != len(tokens) + 1:
                    raise ValueError("Invalid or duplicate chromosome row in LOCO")
                rows[chromosome] = [float(value) if value.upper() != "NA" else float("nan") for value in parts[1:]]
        selected_chromosomes = tuple(sorted(rows) if chromosomes is None else chromosomes)
        if any(chromosome not in rows for chromosome in selected_chromosomes):
            raise ValueError("Requested chromosome is absent from imported LOCO")
        if sample_ids is None:
            parsed = [token.partition("_") for token in tokens]
            if any(not separator for _, separator, _ in parsed):
                raise ValueError("LOCO IDs lack FID_IID separator; supply explicit sample_ids")
            ids = [(fid, iid) for fid, _, iid in parsed]
            indices = list(range(len(tokens)))
        else:
            ids = _ids(sample_ids, len(sample_ids))
            lookup = {token: index for index, token in enumerate(tokens)}
            try:
                indices = [lookup[f"{fid}_{iid}"] for fid, iid in ids]
            except KeyError as error:
                raise ValueError("A requested sample is absent from imported LOCO") from error
        matrix = torch.tensor([rows[chromosome] for chromosome in selected_chromosomes], dtype=torch.float64).T
        # Missing source phenotypes are excluded rather than silently imputed.
        matrix = matrix[indices]
        valid = torch.isfinite(matrix).all(dim=1)
        kept = torch.nonzero(valid, as_tuple=False).flatten()
        if not len(kept):
            raise ValueError("No finite LOCO samples remain")
        ids = [ids[index] for index in kept.tolist()]
        return cls(ids, matrix[kept], selected_chromosomes, y_scale, kept,
                   {"phenotype_name": phenotype_name, "source": "regenie_loco",
                    "source_file": str(source.resolve()), "loco_units": "standardized_step1_phenotype",
                    "excluded_missing_predictions": int((~valid).sum())},
                   prs=torch.tensor(rows[23], dtype=torch.float64)[indices][kept] if 23 in rows else None)


def rank_inverse_normal(values: torch.Tensor) -> torch.Tensor:
    """Blom RINT, with average ranks for exact ties and nonfinite values retained."""
    values = torch.as_tensor(values)
    result = values.to(dtype=torch.float64).clone()
    finite = torch.isfinite(result)
    observations = result[finite]
    if not observations.numel():
        return result
    sorted_values, order = torch.sort(observations, stable=True)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    average_ranks = (starts.to(torch.float64) + ends.to(torch.float64) + 1) / 2
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    probabilities = (sorted_ranks - 0.375) / (len(observations) + 0.25)
    transformed = torch.special.ndtri(probabilities)
    original_order = torch.empty_like(transformed)
    original_order[order] = transformed
    result[finite] = original_order
    return result


# Public alias used by phenotype preparation and Step2 context.
inverse_normal_transform = rank_inverse_normal


def contiguous_folds(n_samples: int, n_folds: int = 5) -> tuple[slice, ...]:
    """REGENIE folds preserve genotype order; only the last fold takes the remainder."""
    if n_folds < 2 or n_samples < n_folds:
        raise ValueError("Need >= 2 folds and at least one sample per fold")
    size = n_samples // n_folds
    return tuple(slice(fold * size, (fold + 1) * size if fold < n_folds - 1 else n_samples)
                 for fold in range(n_folds))


def _masked_folds(valid: torch.Tensor, n_folds: int) -> tuple[slice, ...]:
    """Match genotype-row CV boundaries when REGENIE masks missing observations."""
    active = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if n_folds < 2 or len(active) < n_folds:
        raise ValueError("Need >= 2 folds and at least one active sample per fold")
    target = len(active) // n_folds
    boundaries = [0] + [active[fold * target - 1] + 1 for fold in range(1, n_folds)] + [len(valid)]
    return tuple(slice(boundaries[i], boundaries[i + 1]) for i in range(n_folds))


def covariate_basis(covariates: torch.Tensor | None, n_samples: int | None = None, *,
                    device: str | torch.device | None = None,
                    dtype: torch.dtype = torch.float64, eigen_tolerance: float = 1e-15) -> torch.Tensor:
    """Orthonormalize numerical covariates plus intercept via eigenvalues of X'X."""
    if n_samples is None:
        if covariates is None:
            raise ValueError("n_samples is required when covariates=None")
        n_samples = len(covariates)
    if device is None:
        device = covariates.device if isinstance(covariates, torch.Tensor) else "cpu"
    if covariates is None:
        return torch.full((n_samples, 1), n_samples ** -0.5, dtype=dtype, device=device)
    covariates = torch.as_tensor(covariates, dtype=torch.float64, device=device)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    if covariates.ndim != 2 or covariates.shape[0] != n_samples or not torch.isfinite(covariates).all():
        raise ValueError("Covariates must be finite numerical samples x covariates")
    design = torch.cat((covariates, torch.ones((n_samples, 1), dtype=torch.float64, device=device)), dim=1)
    eigenvalues, eigenvectors = torch.linalg.eigh(design.T @ design)
    keep = eigenvalues > eigenvalues[-1] * eigen_tolerance
    if not keep.any() or int(keep.sum()) >= n_samples:
        raise ValueError("Covariate rank must be positive and smaller than sample size")
    basis = (design @ eigenvectors[:, keep]) / eigenvalues[keep].sqrt()
    return basis.to(dtype=dtype)


@contextlib.contextmanager
def _tf32(enabled: bool):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _shape(genotypes: Any) -> tuple[int, int]:
    if isinstance(genotypes, (torch.Tensor, np.ndarray)):
        if genotypes.ndim != 2:
            raise ValueError("genotypes must be samples x variants")
        return tuple(genotypes.shape)
    return int(genotypes.n_samples), int(genotypes.n_variants)


def _read_block(genotypes: Any, indices: np.ndarray, sample_indices: torch.Tensor) -> torch.Tensor:
    if isinstance(genotypes, torch.Tensor):
        block = genotypes.index_select(1, torch.as_tensor(indices, device=genotypes.device))
        return block.index_select(0, sample_indices.to(block.device))
    if isinstance(genotypes, np.ndarray):
        return torch.from_numpy(np.asarray(genotypes[np.ix_(sample_indices.numpy(), indices)]))
    pieces = []
    seen = []
    for block_indices, matrix in genotypes.iter_blocks(len(indices), indices=indices):
        seen.extend(np.asarray(block_indices, dtype=np.int64).tolist())
        matrix = torch.as_tensor(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != genotypes.n_samples:
            raise ValueError("Reader yielded invalid genotype block shape")
        pieces.append(matrix.index_select(0, sample_indices.to(matrix.device)))
    if seen != indices.tolist() or not pieces:
        raise ValueError("Genotype reader changed requested variant ordering")
    return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]


def _ridge_solutions(gram: torch.Tensor, target: torch.Tensor, penalties: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * 0.5)
    denominator = eigenvalues[:, None] + penalties[None, :]
    if torch.any(denominator <= 0):
        raise RuntimeError("Ridge Gram is numerically indefinite; rerun in float64")
    return eigenvectors @ ((eigenvectors.T @ target)[:, None] / denominator)


@torch.no_grad()
def fit_null(genotypes: Any, phenotype: torch.Tensor | np.ndarray,
             chromosomes: Sequence[int] | torch.Tensor | np.ndarray | None = None, *,
             covariates: torch.Tensor | np.ndarray | None = None,
             config: Step1Config | None = None, output_dir: str | Path | None = None,
             sample_ids: Sequence[Any] | None = None, phenotype_name: str = "phenotype",
             variant_indices: Sequence[int] | torch.Tensor | np.ndarray | None = None,
             progress_callback: Callable[[dict[str, Any]], None] | None = None) -> NullModel:
    """Fit single-trait two-level ridge and held-out LOCO predictions.

    The supplied variant set must already pass the intended chip QC. No implicit
    LD pruning/MAF filtering is performed here. Missing y/covariates are removed;
    genotype NaNs are mean-imputed on retained samples. By default masked rows
    are retained internally exactly as in REGENIE 3.4.1; output includes active samples only. All projection and Gram
    calculations run on config.device; CUDA is required by the default config.
    """
    config = config or Step1Config()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose device='cpu' explicitly for validation")
    started = time.perf_counter()
    n_original, m_original = _shape(genotypes)
    phenotype_cpu = torch.as_tensor(phenotype, dtype=torch.float64).detach().cpu()
    if phenotype_cpu.ndim == 2 and phenotype_cpu.shape[1] == 1:
        phenotype_cpu = phenotype_cpu[:, 0]
    if phenotype_cpu.shape != (n_original,):
        raise ValueError("phenotype must contain exactly one value per genotype sample")
    valid = torch.isfinite(phenotype_cpu)
    covariates_cpu = None
    if covariates is not None:
        covariates_cpu = torch.as_tensor(covariates, dtype=torch.float64).detach().cpu()
        if covariates_cpu.ndim == 1:
            covariates_cpu = covariates_cpu[:, None]
        if covariates_cpu.ndim != 2 or covariates_cpu.shape[0] != n_original:
            raise ValueError("covariates must be samples x numerical covariates")
        valid &= torch.isfinite(covariates_cpu).all(dim=1)
    kept_samples = torch.nonzero(valid, as_tuple=False).flatten()
    n_samples = len(kept_samples)
    if config.preserve_masked_rows:
        working_indices = torch.arange(n_original, dtype=torch.long)
        working_valid = valid
        folds = _masked_folds(valid, config.folds)
        output_rows = kept_samples
    else:
        working_indices = kept_samples
        working_valid = torch.ones(n_samples, dtype=torch.bool)
        folds = contiguous_folds(n_samples, config.folds)
        output_rows = torch.arange(n_samples, dtype=torch.long)
    n_rows = len(working_indices)
    ids_source = sample_ids if sample_ids is not None else getattr(genotypes, "sample_ids", None)
    all_ids = _ids(ids_source if ids_source is not None else [(str(i + 1), str(i + 1)) for i in range(n_original)], n_original)
    ids = [all_ids[index] for index in kept_samples.tolist()]
    if chromosomes is None:
        chromosomes = getattr(genotypes, "variant_chromosomes", None)
    if chromosomes is None:
        raise ValueError("Provide one chromosome per input genotype variant")
    chromosome_array = np.asarray(torch.as_tensor(chromosomes, dtype=torch.long).cpu())
    if chromosome_array.shape != (m_original,):
        raise ValueError("chromosomes must contain one entry per genotype variant")
    selected = np.arange(m_original, dtype=np.int64) if variant_indices is None else np.asarray(
        torch.as_tensor(variant_indices, dtype=torch.long).cpu())
    if selected.ndim != 1 or not len(selected) or np.any(selected < 0) or np.any(selected >= m_original):
        raise ValueError("variant_indices must be a nonempty valid one-dimensional index vector")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Duplicate variant_indices are not allowed")
    if np.any(~np.isin(chromosome_array[selected], config.output_chromosomes)):
        raise ValueError("Selected variants must belong to output_chromosomes (autosomes 1-22)")
    blocks: list[tuple[int, np.ndarray]] = []
    # Preserve genotype-file order within each chromosome, as in REGENIE.
    selected = np.sort(selected)
    for chromosome in config.output_chromosomes:
        chromosome_indices = selected[chromosome_array[selected] == chromosome]
        blocks.extend((chromosome, chromosome_indices[start:start + config.block_size])
                      for start in range(0, len(chromosome_indices), config.block_size))
    m_total = len(selected)
    l0_count = len(config.ridge_l0)
    n_predictors = len(blocks) * l0_count
    bytes_per_value = 4 if dtype == torch.float32 else 8
    # Includes genotype/projection temporaries, every fold Gram, eigensolver
    # workspace and streamed L1 prediction rows. It is deliberately conservative.
    estimated_bytes = bytes_per_value * (3 * n_rows * min(config.block_size, m_total) +
        (config.folds + 8) * n_predictors ** 2 + 3 * config.sample_chunk_size * n_predictors)
    if device.type == "cuda" and estimated_bytes > config.max_gpu_gb * 1e9:
        raise MemoryError("Estimated GPU memory exceeds max_gpu_gb; reduce block/sample chunk size")
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    temp_directory = None
    l0_file = None
    if config.l0_storage == "memmap":
        if destination is None:
            if config.keep_l0:
                raise ValueError("keep_l0 requires output_dir")
            temp_directory = tempfile.TemporaryDirectory(prefix="torchwgs-step1-")
            scratch = Path(temp_directory.name)
        else:
            scratch = destination
        # Unique path prevents concurrent phenotypes from overwriting each other.
        descriptor = tempfile.NamedTemporaryFile(prefix="l0_", suffix=".dat", dir=scratch, delete=False)
        l0_file = Path(descriptor.name)
        descriptor.close()
        features = np.memmap(l0_file, mode="w+", dtype=config.dtype, shape=(n_rows, n_predictors))
    else:
        features = np.empty((n_rows, n_predictors), dtype=config.dtype)
    block_chromosomes = [chromosome for chromosome, _ in blocks for _ in config.ridge_l0]
    stage_timings = {}
    try:
        with _tf32(config.tf32):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            active_y = phenotype_cpu[kept_samples].to(device)
            if config.apply_rint:
                active_y = rank_inverse_normal(active_y)
            cov = covariates_cpu[kept_samples] if covariates_cpu is not None else None
            # Covariate rank/RINT and N-C normalization are evaluated on active
            # samples. REGENIE retains masked genotype rows as zeros internally.
            active_basis = covariate_basis(cov, n_samples, device=device,
                                           eigen_tolerance=config.covariate_eigen_tolerance)
            active_y -= active_basis @ (active_basis.T @ active_y)
            residual_df = n_samples - active_basis.shape[1]
            y_scale_tensor = active_y.norm() / residual_df ** 0.5
            if y_scale_tensor < config.variance_tolerance:
                raise ValueError("Phenotype has zero/low residual variance")
            y_scale = float(y_scale_tensor)
            y = torch.zeros(n_rows, dtype=dtype, device=device)
            basis = torch.zeros((n_rows, active_basis.shape[1]), dtype=dtype, device=device)
            device_output_rows = output_rows.to(device)
            y[device_output_rows] = (active_y / y_scale_tensor).to(dtype)
            basis[device_output_rows] = active_basis.to(dtype)
            mask = working_valid.to(device)
            del active_y, active_basis
            penalties0 = torch.tensor([m_total * (1 - h) / h for h in config.ridge_l0], dtype=dtype, device=device)
            l0_started = time.perf_counter()
            for block_number, (chromosome, indices) in enumerate(blocks):
                raw = _read_block(genotypes, indices, working_indices)
                g = raw.to(device=device, dtype=dtype)
                del raw
                if torch.isinf(g).any() or ((torch.isfinite(g)) & ((g < 0) | (g > 2))).any():
                    raise ValueError("Autosomal ALT dosage must be in [0,2], or NaN for missing")
                observed = torch.isfinite(g) & mask[:, None]
                count = observed.sum(0)
                if torch.any(count == 0):
                    raise ValueError("A chip variant is missing for all retained samples")
                means = torch.where(observed, g, 0).sum(0, dtype=torch.float64) / count
                g = torch.where(observed, g, means.to(dtype)[None, :])
                del observed
                g *= mask[:, None]
                g -= basis @ (basis.T @ g)
                scale = g.norm(dim=0) / residual_df ** 0.5
                if torch.any(scale < config.variance_tolerance):
                    raise ValueError("A chip variant has zero/low residual variance; check chip QC")
                g /= scale
                full_gram = g.T @ g
                full_target = g.T @ y
                predictions = torch.empty((n_rows, l0_count), dtype=dtype, device=device)
                for fold in folds:
                    heldout_g = g[fold]
                    gram = full_gram - heldout_g.T @ heldout_g
                    target = full_target - heldout_g.T @ y[fold]
                    coefficients = _ridge_solutions(gram, target, penalties0)
                    predictions[fold] = heldout_g @ coefficients
                # The original centers all held-out predictions together; the
                # L0 feature standard deviation uses N-1, unlike genotype N-C.
                mean = predictions.sum(0, dtype=torch.float64) / n_samples
                centered_ss = predictions.to(torch.float64).square().sum(0) - n_samples * mean.square()
                if torch.any(centered_ss <= 0) or not torch.isfinite(centered_ss).all():
                    raise ValueError("A level-0 prediction is constant/nonfinite")
                predictions -= mean.to(dtype)
                predictions *= ((n_samples - 1) / centered_ss).sqrt().to(dtype)
                begin = block_number * l0_count
                features[:, begin:begin + l0_count] = predictions.cpu().numpy()
                if progress_callback is not None:
                    progress_callback({"stage": "level0", "completed_blocks": block_number + 1,
                                       "total_blocks": len(blocks), "chromosome": chromosome})
                del g, predictions, full_gram, full_target, gram, target, coefficients, heldout_g
            if isinstance(features, np.memmap):
                features.flush()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_timings["level0_seconds"] = time.perf_counter() - l0_started
            l1_started = time.perf_counter()
            grams = [torch.zeros((n_predictors, n_predictors), dtype=dtype, device=device) for _ in folds]
            targets = [torch.zeros(n_predictors, dtype=dtype, device=device) for _ in folds]
            for fold_number, fold in enumerate(folds):
                for start in range(fold.start, fold.stop, config.sample_chunk_size):
                    end = min(start + config.sample_chunk_size, fold.stop)
                    chunk = torch.from_numpy(np.asarray(features[start:end])).to(device=device, dtype=dtype)
                    grams[fold_number].add_(chunk.T @ chunk)
                    targets[fold_number].add_(chunk.T @ y[start:end])
                if progress_callback is not None:
                    progress_callback({"stage": "level1_gram", "completed_folds": fold_number + 1,
                                       "total_folds": config.folds})
            total_gram = sum(grams)
            total_target = sum(targets)
            penalties1 = torch.tensor([n_predictors * (1 - h) / h for h in config.ridge_l1], dtype=dtype, device=device)
            mse_numerator = torch.zeros(len(config.ridge_l1), dtype=torch.float64, device=device)
            fold_coefficients = []
            for fold_number, fold in enumerate(folds):
                coefficients = _ridge_solutions(total_gram - grams[fold_number],
                                                total_target - targets[fold_number], penalties1)
                fold_coefficients.append(coefficients)
                for start in range(fold.start, fold.stop, config.sample_chunk_size):
                    end = min(start + config.sample_chunk_size, fold.stop)
                    chunk = torch.from_numpy(np.asarray(features[start:end])).to(device=device, dtype=dtype)
                    difference = (chunk @ coefficients).to(torch.float64) - y[start:end, None].to(torch.float64)
                    mse_numerator += difference.square().sum(0)
            mse = mse_numerator / n_samples
            selected_l1 = int(torch.argmin(mse))
            del grams, targets, total_gram, total_target
            # Default k-fold REGENIE does not refit L1 using all samples here.
            # Its final chromosome contributions use each held-out fold's beta.
            contribution = torch.zeros((n_rows, len(config.output_chromosomes)), dtype=dtype, device="cpu")
            chromosome_columns = {chromosome: np.flatnonzero(np.asarray(block_chromosomes) == chromosome)
                                  for chromosome in config.output_chromosomes}
            for fold_number, fold in enumerate(folds):
                beta = fold_coefficients[fold_number][:, selected_l1]
                for start in range(fold.start, fold.stop, config.sample_chunk_size):
                    end = min(start + config.sample_chunk_size, fold.stop)
                    chunk = torch.from_numpy(np.asarray(features[start:end])).to(device=device, dtype=dtype)
                    for column, chromosome in enumerate(config.output_chromosomes):
                        columns = chromosome_columns[chromosome]
                        if len(columns):
                            index_tensor = torch.as_tensor(columns, dtype=torch.long, device=device)
                            contribution[start:end, column] = (chunk.index_select(1, index_tensor) @ beta[index_tensor]).cpu()
            loco = (contribution.sum(1, keepdim=True) - contribution)[output_rows]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_timings["level1_seconds"] = time.perf_counter() - l1_started
            peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        metadata = {
            "implementation": "torchwgs_single_qt_step1", "algorithm_reference": "REGENIE 3.4.1 default k-fold QT",
            "phenotype_name": phenotype_name, "config": asdict(config),
            "n_original_samples": n_original, "n_samples": n_samples, "n_internal_rows": n_rows,
            "n_covariates_including_intercept": int(basis.shape[1]),
            "n_variants": m_total, "n_blocks": len(blocks), "n_level0_predictors": n_predictors,
            "fold_sizes": [fold.stop - fold.start for fold in folds],
            "fold_active_sizes": [int(working_valid[fold].sum()) for fold in folds],
            "selected_ridge_l1_index": selected_l1, "selected_ridge_l1_h": float(config.ridge_l1[selected_l1]),
            "ridge_l0_penalties": penalties0.cpu().tolist(), "ridge_l1_penalties": penalties1.cpu().tolist(),
            "cv_mse": mse.cpu().tolist(), "y_scale": y_scale,
            "loco_units": "standardized_step1_phenotype", "final_l1_refit_full_samples": False,
            "variant_index_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "timings": stage_timings, "total_seconds": time.perf_counter() - started,
            "estimated_gpu_bytes": estimated_bytes, "peak_gpu_allocated_bytes": peak_memory,
            "level0_file": str(l0_file.resolve()) if config.keep_l0 and l0_file is not None else None,
            "level0_shape": [n_rows, n_predictors],
        }
        model = NullModel(ids, loco, config.output_chromosomes, y_scale, kept_samples, metadata,
                          prs=contribution.sum(1)[output_rows])
        if destination is not None:
            model.save(destination)
        return model
    finally:
        if isinstance(features, np.memmap):
            features.flush()
            features._mmap.close()
        if l0_file is not None and not config.keep_l0:
            l0_file.unlink(missing_ok=True)
        if temp_directory is not None:
            temp_directory.cleanup()
