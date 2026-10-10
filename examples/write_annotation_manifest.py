"""Create an input manifest for independently downloaded annotation files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_manifest(output_directory, chromosome, variant_archive,
                   gene_csv, ncrna_csv, promoter_tsv, *, default_qc=None,
                   reference_allele_source="annotation", allow_missing_non_snv=False):
    """Map original FAVOR columns without accepting prepared metadata.

    The files stay external inputs. ``default_qc`` is an explicit project
    policy; omit it and add the actual QC column mapping when it is available.
    """
    if reference_allele_source not in ("annotation", "bim_variant_id"):
        raise ValueError("reference_allele_source must be annotation or bim_variant_id")
    if allow_missing_non_snv and (reference_allele_source != "bim_variant_id" or default_qc is None):
        raise ValueError("missing non-SNV policy requires explicit BIM identity and QC declarations")
    weight_columns = {
        "CADD": "cadd_phred",
        "LINSIGHT": "linsight",
        "FATHMM.XF": "fathmm_xf",
        "aPC.EpigeneticActive": "apc_epigenetics_active",
        "aPC.EpigeneticRepressed": "apc_epigenetics_repressed",
        "aPC.EpigeneticTranscription": "apc_epigenetics_transcription",
        "aPC.Conservation": "apc_conservation_v2",
        "aPC.LocalDiversity": "apc_local_nucleotide_diversity_v3",
        "aPC.Mappability": "apc_mappability",
        "aPC.TF": "apc_transcription_factor",
        "aPC.Protein": "apc_protein_function_v3",
    }
    category_columns = {
        "GENCODE.Category": "genecode_comprehensive_category",
        "GENCODE.Info": "genecode_comprehensive_info",
        "GENCODE.EXONIC.Category": "genecode_comprehensive_exonic_category",
        "MetaSVM": "metasvm_pred", "GeneHancer": "genehancer",
        "CAGE": "cage_tc", "DHS": "rdhs",
    }
    all_columns = dict(category_columns, **weight_columns)
    catalog = {name: "annotation/" + column for name, column in all_columns.items()}
    mapping = dict(chromosome="chromosome", position="position",
                   reference="ref_vcf", alternate="alt_vcf")
    mapping.update({catalog[name]: column for name, column in all_columns.items()})
    manifest = dict(
        schema_version=1, format="raw-wgs-annotations-v1",
        annotation_catalog=catalog, annotation_names=list(weight_columns),
        reference_allele_source=reference_allele_source,
        allow_missing_non_snv=allow_missing_non_snv,
        column_mapping=mapping, qc_path="annotation/filter",
        chromosomes=[dict(name=str(chromosome), variants=str(Path(variant_archive).resolve()),
                         gene_reference=str(Path(gene_csv).resolve()),
                         ncrna_reference=str(Path(ncrna_csv).resolve()),
                         promoter_intervals=str(Path(promoter_tsv).resolve()))],
    )
    if default_qc is not None:
        manifest["default_qc"] = default_qc
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "annotations.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--variant-archive", required=True)
    parser.add_argument("--gene-csv", required=True)
    parser.add_argument("--ncrna-csv", required=True)
    parser.add_argument("--promoter-tsv", required=True)
    parser.add_argument("--default-qc", help="Explicit constant QC policy, not inferred QC")
    parser.add_argument("--reference-allele-source", choices=("annotation", "bim_variant_id"), default="annotation")
    parser.add_argument("--allow-missing-non-snv", action="store_true",
                        help="Explicitly retain uncovered non-SNVs; restricts gene analysis to covered SNVs")
    print(write_manifest(**vars(parser.parse_args())))
