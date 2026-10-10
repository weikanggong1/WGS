"""Export independent gene and promoter references with Python only.

Inputs are local, independently downloaded reference archives. This script
does not download resources or launch an external language interpreter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import numbers
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile

GENE_SOURCE_SHA256 = "4bfa7ab5fe2c25f8aaf6e4319e683c3e29b97f62e687b01036501cffd32b484f"
PROMOTER_ARCHIVE_SHA256 = "feb61126a6d874949423703c90e92e6073d22fecf2db18c44aa9859a43d53f4e"
PROMOTER_SQLITE_SHA256 = "7fab4f12a779f3917f84f19e6fe10b66fd5564ad015da55163d5c17e6c86f573"
GENE_COLUMNS = ("hgnc_symbol", "chromosome_name", "start_position", "end_position")
NCRNA_COLUMNS = ("chr", "ncRNA")
PROMOTER_COLUMNS = ("chromosome", "start", "end", "strand", "gene_id")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_source(path, expected_sha256):
    path = Path(path).resolve()
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("an explicit lowercase SHA-256 checksum is required")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError("reference source SHA-256 differs from the expected version")
    return path, actual


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(label + " must be a positive integer")
    return int(value)


def _name(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(label + " must be a nonempty, unmodified reference string")
    return value


def export_gene_tables(objects, output_directory):
    """Preserve official table values and row order, without inferred names."""
    directory = Path(output_directory)
    outputs = {}
    for key, filename, columns in (("genes_info", "genes.csv", GENE_COLUMNS),
                                   ("ncRNA_gene", "ncrna.csv", NCRNA_COLUMNS)):
        table = objects.get(key)
        if table is None or tuple(table.columns) != columns or table.empty or table.isna().any().any():
            raise ValueError("gene reference table is missing or has an unexpected schema")
        path = directory / filename
        chromosome_counts = {}
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for row in table.itertuples(index=False, name=None):
                if key == "genes_info":
                    name = _name(row[0], "gene name")
                    chromosome = _integer(row[1], "gene chromosome")
                    start, end = _integer(row[2], "gene start"), _integer(row[3], "gene end")
                    if start > end:
                        raise ValueError("gene start must not exceed gene end")
                    values = [name, chromosome, start, end]
                else:
                    chromosome = _integer(row[0], "noncoding RNA chromosome")
                    values = [chromosome, _name(row[1], "noncoding RNA name")]
                writer.writerow(values)
                chromosome_counts[str(chromosome)] = chromosome_counts.get(str(chromosome), 0) + 1
        outputs[filename] = dict(rows=len(table), header=list(columns),
                                 sha256=sha256(path), chromosome_counts=chromosome_counts)
    return outputs


def export_promoters(database_path, output_path):
    """Apply the official single-range gene and strand-aware promoter rules."""
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        metadata = dict(connection.execute("SELECT name, value FROM metadata"))
        if metadata.get("Genome") != "hg38" or metadata.get("Db type") != "TxDb":
            raise ValueError("promoter reference must be a TxDb database for hg38")
        groups = {}
        transcript_gene_rows = 0
        for gene, chromosome, strand, start, end in connection.execute("""
            SELECT g.gene_id, t.tx_chrom, t.tx_strand, t.tx_start, t.tx_end
            FROM gene AS g INNER JOIN transcript AS t ON g._tx_id = t._tx_id
        """):
            transcript_gene_rows += 1
            gene, chromosome = _name(gene, "gene identifier"), _name(chromosome, "transcript chromosome")
            if strand not in ("+", "-"):
                raise ValueError("transcript strand must be + or -")
            start, end = _integer(start, "transcript start"), _integer(end, "transcript end")
            if start > end:
                raise ValueError("transcript start must not exceed transcript end")
            ranges = groups.setdefault(gene, {})
            key = (chromosome, strand)
            if key in ranges:
                previous_start, previous_end = ranges[key]
                ranges[key] = min(previous_start, start), max(previous_end, end)
            else:
                ranges[key] = start, end
    chromosome_counts, nonpositive = {}, 0
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PROMOTER_COLUMNS)
        # Character gene identifiers are ordered by the original split rule.
        for gene in sorted(groups):
            ranges = groups[gene]
            if len(ranges) != 1:
                continue
            (chromosome, strand), (start, end) = next(iter(ranges.items()))
            if strand == "+":
                promoter_start, promoter_end = start - 3000, start + 2999
            else:
                promoter_start, promoter_end = end - 2999, end + 3000
            # Keep out-of-bound coordinates: the reference method does not trim.
            nonpositive += promoter_start <= 0
            chromosome = chromosome[3:] if chromosome.startswith("chr") else chromosome
            chromosome_counts[chromosome] = chromosome_counts.get(chromosome, 0) + 1
            writer.writerow([chromosome, promoter_start, promoter_end, strand, gene])
    return dict(rows=sum(chromosome_counts.values()), header=list(PROMOTER_COLUMNS),
                sha256=sha256(output_path), chromosome_counts=chromosome_counts,
                transcript_gene_rows=transcript_gene_rows, gene_identifiers=len(groups),
                excluded_multi_range_genes=sum(len(ranges) != 1 for ranges in groups.values()),
                nonpositive_start_rows=nonpositive, width=6000, upstream=3000, downstream=3000,
                coordinate_system="1-based-closed", trim_out_of_bound=False,
                sqlite_sha256=sha256(database_path), database_metadata=metadata)


def _database(source, temporary_directory, expected_package_version):
    if not tarfile.is_tarfile(source):
        return source
    with tarfile.open(source, "r:gz") as archive:
        databases = [member for member in archive.getmembers()
                     if member.isfile() and member.name.endswith(".sqlite")]
        descriptions = [member for member in archive.getmembers()
                        if member.isfile() and member.name.endswith("/DESCRIPTION")]
        if len(databases) != 1 or len(descriptions) != 1:
            raise ValueError("promoter archive must contain exactly one SQLite database and package description")
        description = archive.extractfile(descriptions[0]).read().decode("utf-8")
        versions = [line.partition(":")[2].strip() for line in description.splitlines()
                    if line.startswith("Version:")]
        if versions != [expected_package_version]:
            raise ValueError("promoter package version differs from the expected version")
        database = Path(temporary_directory) / "promoter.sqlite"
        with archive.extractfile(databases[0]) as handle, database.open("wb") as output:
            shutil.copyfileobj(handle, output)
    return database


def export_references(gene_reference_archive, promoter_database, output_directory, *,
                      gene_source_sha256=GENE_SOURCE_SHA256, promoter_source_sha256=None,
                      promoter_package_version="3.22.0"):
    """Export local official references and retain source/output integrity proof."""
    directory = Path(output_directory).resolve()
    if directory.exists():
        raise FileExistsError("use a fresh reference output directory")
    genes, gene_hash = checked_source(gene_reference_archive, gene_source_sha256)
    promoter_source = Path(promoter_database).resolve()
    if promoter_source_sha256 is None:
        promoter_source_sha256 = (PROMOTER_ARCHIVE_SHA256 if tarfile.is_tarfile(promoter_source)
                                  else PROMOTER_SQLITE_SHA256)
    promoter_source, promoter_hash = checked_source(promoter_source, promoter_source_sha256)
    try:
        import rdata
    except ImportError as error:
        raise ImportError("install the optional annotations dependency before exporting references") from error
    objects = rdata.conversion.convert(rdata.parser.parse_file(genes))
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-export-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "references"
        staging.mkdir()
        database = _database(promoter_source, temporary, promoter_package_version)
        outputs = export_gene_tables(objects, staging)
        outputs["promoter_intervals.tsv"] = export_promoters(database, staging / "promoter_intervals.tsv")
        if sha256(genes) != gene_hash or sha256(promoter_source) != promoter_hash:
            raise ValueError("reference source changed during export")
        proof = dict(completed=True, method="Python reference archive and SQLite export",
                     gene_source=dict(path=str(genes), sha256=gene_hash),
                     promoter_source=dict(path=str(promoter_source), sha256=promoter_hash,
                                          expected_package_version=promoter_package_version),
                     reader_version=rdata.__version__, outputs=outputs)
        (staging / "reference_export.private.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
        staging.rename(directory)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-reference-archive", required=True, type=Path)
    parser.add_argument("--promoter-database", required=True, type=Path,
                        help="Original local package tar.gz or its independently extracted SQLite")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--gene-source-sha256", default=GENE_SOURCE_SHA256)
    parser.add_argument("--promoter-source-sha256", default=None)
    parser.add_argument("--promoter-package-version", default="3.22.0")
    proof = export_references(**vars(parser.parse_args(argv)))
    print(json.dumps(dict(completed=proof["completed"], reader_version=proof["reader_version"],
                          outputs={name:dict(rows=record["rows"], sha256=record["sha256"])
                                   for name, record in proof["outputs"].items()}), indent=2))


if __name__ == "__main__":
    main()
