import argparse
import json
from dataclasses import replace
from .config import WGSConfig
from .pipeline import DiscoveryInputs, GeneAnalysis, run_discovery


def main():
    parser=argparse.ArgumentParser(description='PyTorch GPU discovery WGS association')
    parser.add_argument('--inputs',required=True,help='JSON DiscoveryInputs; keep private paths/covariates local')
    parser.add_argument('--config',help='JSON overrides for WGSConfig paper defaults')
    parser.add_argument('--out',required=True)
    parser.add_argument('--no-resume',action='store_true')
    parser.add_argument('--no-rint',action='store_true',help='Disable RINT in Step1 and both association analyses')
    args=parser.parse_args()
    values=json.load(open(args.inputs))
    values['gene_analyses']={str(c):[GeneAnalysis(**a) for a in analyses] for c,analyses in values.get('gene_analyses',{}).items()}
    configuration=WGSConfig.paper() if args.config is None else WGSConfig.from_dict(json.load(open(args.config)))
    if args.no_rint:
        configuration.step1=replace(configuration.step1,apply_rint=False)
        configuration.single_variant.apply_rint=False
        configuration.gene_based.apply_rint=False
        configuration.phenotype_quantile_normalize=False
    run_discovery(DiscoveryInputs(**values),config=configuration,output_dir=args.out,
                  resume=not args.no_resume)

if __name__=='__main__': main()
