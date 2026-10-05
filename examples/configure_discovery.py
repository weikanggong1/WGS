"""Create editable input/config JSON files for the complete discovery analysis."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from torchwgs import WGSConfig, DiscoveryInputs, study_gene_analyses, ExecutionConfig
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
                        help='BED prefix with {chromosome}, e.g. /data/wgs/chr{chromosome}')
    parser.add_argument('--annotation-root',required=True,
                        help='Annotation directory, e.g. /data/annotation, with annotation, setlist and mask files')
    parser.add_argument('--json-directory',required=True)
    parser.add_argument('--chromosomes',type=int,nargs='+',default=list(range(1,23)))
    parser.add_argument('--imported-loco',help='Compatible full-chip Step1 prediction list')
    parser.add_argument('--parallel-level',choices=['serial','mask','chromosome'],default='serial')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--max-gpu-gb',type=float,default=20.)
    parser.add_argument('--genotype-reader',choices=['cpu','cuda_packed'],default='cuda_packed')
    parser.add_argument('--no-rint',action='store_true')
    parser.add_argument('--float64',action='store_true')
    mask_output=parser.add_mutually_exclusive_group()
    mask_output.add_argument('--write-masks',dest='write_masks',action='store_true',
                             help='Write BED/BIM/FAM/snplist mask files (paper default)')
    mask_output.add_argument('--no-write-masks',dest='write_masks',action='store_false',
                             help='Omit BED/BIM/FAM/snplist mask files')
    parser.set_defaults(write_masks=True)
    args=parser.parse_args()
    inputs=DiscoveryInputs(
        array_prefix=args.array_prefix,array_variant_include=args.array_variant_include,
        phenotype_file=args.phenotype_file,phenotype_column=args.phenotype_column,
        discovery_samples=args.discovery_samples,sample_remove=args.sample_remove,
        wgs_prefixes={str(chromosome):args.wgs_prefix_template.format(chromosome=chromosome)
                      for chromosome in args.chromosomes},
        gene_analyses=study_gene_analyses(args.annotation_root,chromosomes=args.chromosomes),
        imported_loco=args.imported_loco,
    )
    configuration=WGSConfig.paper(apply_rint=not args.no_rint)
    configuration.write_masks=args.write_masks
    configuration.execution=ExecutionConfig(args.parallel_level,args.workers,args.max_gpu_gb)
    configuration.single_variant.genotype_reader=args.genotype_reader
    configuration.gene_based.genotype_reader=args.genotype_reader
    if args.float64:
        configuration.step1=replace(configuration.step1,dtype='float64',tf32=False)
        configuration.single_variant.dtype='float64'
        configuration.single_variant.tf32=False
    output=Path(args.json_directory)
    output.mkdir(parents=True,exist_ok=True)
    (output/'discovery_inputs.json').write_text(json.dumps(asdict(inputs),indent=2)+'\n')
    (output/'analysis_config.json').write_text(json.dumps(configuration.to_dict(),indent=2)+'\n')


if __name__=='__main__':main()
