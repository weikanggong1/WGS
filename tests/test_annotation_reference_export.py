"""Independent reference export checks; anonymous inputs, no benchmark claims."""
import csv
import importlib.util
from pathlib import Path
import sqlite3
import tarfile

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "annotation_reference_example", Path(__file__).parents[1] / "examples/export_annotation_references.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def database(path):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (name TEXT, value TEXT);
            INSERT INTO metadata VALUES ('Genome','hg38'),('Db type','TxDb');
            CREATE TABLE transcript (_tx_id INTEGER,tx_chrom TEXT,tx_strand TEXT,tx_start INTEGER,tx_end INTEGER);
            CREATE TABLE gene (gene_id TEXT,_tx_id INTEGER);
        """)
        # One gene has two transcripts; two have incompatible chromosome/strand.
        rows = [(1, "g_b", "chr21", "+", 8000, 8300),
                (2, "g_b", "chr21", "+", 7500, 8600),
                (3, "g_a", "chr21", "-", 9000, 9900),
                (4, "g_multichrom", "chr21", "+", 10000, 10500),
                (5, "g_multichrom", "chr22", "+", 10000, 10500),
                (6, "g_multistrand", "chr21", "+", 11000, 11500),
                (7, "g_multistrand", "chr21", "-", 11000, 11500),
                (8, "g_out_of_bounds", "chrM", "+", 100, 300)]
        for ordinal, gene, chromosome, strand, start, end in rows:
            connection.execute("INSERT INTO transcript VALUES (?,?,?,?,?)", (ordinal, chromosome, strand, start, end))
            connection.execute("INSERT INTO gene VALUES (?,?)", (gene, ordinal))
    return path


def gene_objects():
    return dict(genes_info=pd.DataFrame([['gene_b',21,8000,8100], ['gene_a',21,5000,5300]],
                                        columns=example.GENE_COLUMNS),
                ncRNA_gene=pd.DataFrame([[21,'rna_b'],[22,'rna_a']], columns=example.NCRNA_COLUMNS))


def test_promoters_merge_transcripts_and_keep_strand_boundaries(tmp_path):
    source = database(tmp_path / "original.sqlite")
    output = tmp_path / "promoters.tsv"
    proof = example.export_promoters(source, output)
    rows = list(csv.DictReader(output.open(), delimiter="\t"))
    assert [row['gene_id'] for row in rows] == ['g_a', 'g_b', 'g_out_of_bounds']
    assert [(int(row['start']), int(row['end'])) for row in rows] == [(6901, 12900), (4500, 10499), (-2900, 3099)]
    assert all(int(row['end']) - int(row['start']) + 1 == 6000 for row in rows)
    assert proof['excluded_multi_range_genes'] == 2
    assert proof['nonpositive_start_rows'] == 1
    assert proof['chromosome_counts'] == {'21': 2, 'M': 1}


def test_gene_export_preserves_source_rows_and_integer_cells(tmp_path):
    proof = example.export_gene_tables(gene_objects(), tmp_path)
    assert (tmp_path / 'genes.csv').read_bytes() == b'hgnc_symbol,chromosome_name,start_position,end_position\ngene_b,21,8000,8100\ngene_a,21,5000,5300\n'
    assert (tmp_path / 'ncrna.csv').read_bytes() == b'chr,ncRNA\n21,rna_b\n22,rna_a\n'
    assert proof['genes.csv']['rows'] == 2


@pytest.mark.parametrize('change', ['schema', 'missing', 'coordinates'])
def test_invalid_gene_inputs_are_rejected(tmp_path, change):
    objects = gene_objects()
    if change == 'schema':
        objects['genes_info'] = objects['genes_info'].drop(columns='hgnc_symbol')
    elif change == 'missing':
        objects['genes_info'].loc[0, 'hgnc_symbol'] = None
    else:
        objects['genes_info'].loc[0, 'end_position'] = 10
    with pytest.raises(ValueError):
        example.export_gene_tables(objects, tmp_path)


def test_source_hash_mismatch_does_not_create_output(tmp_path):
    source = tmp_path / 'source.reference'
    source.write_bytes(b'changed-input')
    with pytest.raises(ValueError, match='SHA-256 differs'):
        example.export_references(source, source, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_archive_description_version_is_checked(tmp_path):
    sql = database(tmp_path / 'input.sqlite')
    description = tmp_path / 'DESCRIPTION'
    description.write_text('Package: anonymous\nVersion: 9.0.0\n')
    archive = tmp_path / 'input.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(sql, arcname='anonymous/inst/extdata/reference.sqlite')
        handle.add(description, arcname='anonymous/DESCRIPTION')
    with pytest.raises(ValueError, match='package version'):
        example._database(archive, tmp_path, '3.22.0')


def test_wrong_build_is_rejected(tmp_path):
    sql = database(tmp_path / 'input.sqlite')
    with sqlite3.connect(sql) as connection:
        connection.execute("UPDATE metadata SET value='hg19' WHERE name='Genome'")
    with pytest.raises(ValueError, match='hg38'):
        example.export_promoters(sql, tmp_path / 'output.tsv')
