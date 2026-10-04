"""Create editable input/config JSON files for the complete discovery analysis."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from torchwgs import WGSConfig, DiscoveryInputs, study_gene_analyses
from dataclasses import asdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array-prefix',required=True)
    parser.add_argument('--array-variant-include',required=True)
    parser.add_argument('--phenotype-file',required=True)
    parser.add_argument('--phenotype-column',required=True)
    parser.add_argument('--discovery-samples',required=True)
    parser.add_argument('--sample-remove')
    parser.add_argument('--wgs-prefix-template',required=True,
                        help='BED prefix with {chromosome}, e.g. /data/wgs/Image_Q3_c{chromosome}')
    parser.add_argument('--annotation-root',required=True,help='Existing Anno_New directory')
    parser.add_argument('--json-directory',required=True)
    parser.add_argument('--no-rint',action='store_true')
    parser.add_argument('--float64',action='store_true')
    parser.add_argument('--write-masks',action='store_true')
    args=parser.parse_args()
    inputs=DiscoveryInputs(
        array_prefix=args.array_prefix,array_variant_include=args.array_variant_include,
        phenotype_file=args.phenotype_file,phenotype_column=args.phenotype_column,
        discovery_samples=args.discovery_samples,sample_remove=args.sample_remove,
        wgs_prefixes={str(chromosome):args.wgs_prefix_template.format(chromosome=chromosome)
                      for chromosome in range(1,23)},
        gene_analyses=study_gene_analyses(args.annotation_root),
    )
    configuration=WGSConfig.paper(apply_rint=not args.no_rint)
    configuration.write_masks=args.write_masks
    if args.float64:
        configuration.step1=replace(configuration.step1,dtype='float64',tf32=False)
        configuration.single_variant.dtype='float64'
        configuration.single_variant.tf32=False
    output=Path(args.json_directory)
    output.mkdir(parents=True,exist_ok=True)
    (output/'discovery_inputs.json').write_text(json.dumps(asdict(inputs),indent=2)+'\n')
    (output/'analysis_config.json').write_text(json.dumps(configuration.to_dict(),indent=2)+'\n')


if __name__=='__main__':main()
