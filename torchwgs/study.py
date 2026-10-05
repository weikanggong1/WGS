"""Expand the inspected study Main/Sub file layout into pipeline inputs."""
from pathlib import Path
from .pipeline import GeneAnalysis

MAIN_TYPES=('PTV','Missense','Splice','Inframe','Synonymous','Intron',
            'UTR_5','UTR_3','Upstream','Downstream','Intergenic','Pseudo','RNA','nctev')
SUB_COMBINATIONS=(
    ('Intron','GERP2'),('Intergenic','GERP2'),('UTR_5','GERP2'),
    ('UTR_3','GERP2'),('Upstream','GERP2'),('Downstream','GERP2'),
    ('Intron','Gnocchi4'),('Intergenic','Gnocchi4'),('Splice','splice05'),
    ('Missense','REVEL50'),('Upstream','JARVIS99'),('Downstream','JARVIS99'),
)


def study_gene_analyses(annotation_root, *, chromosomes=range(1,23),
                        main_types=MAIN_TYPES, sub_combinations=SUB_COMBINATIONS,
                        require_files=True):
    """Build all 14 Main + 12 Sub entries/chromosome from inspected mask files.

    Score names identify supplied whitelist files. They do not infer the numeric
    comparison from a filename, and do not replace the unpublished Table S24.
    Both lists are caller-editable. No source annotation is regenerated here.
    """
    root=Path(annotation_root);result={}
    for chrom in chromosomes:
        entries=[]
        for kind,score in [(t,None) for t in main_types]+list(sub_combinations):
            path=root/f'chr{chrom}'
            specification=GeneAnalysis(kind if score is None else kind+'_'+score,
                str(path/'Main'/f'{kind}_chr{chrom}.txt'),
                str(path/'Main'/f'chr{chrom}_{kind}.setlist'),
                str(root/'Mask'/f'Mask_{kind}.txt'),
                None if score is None else str(path/'Subset'/f'chr{chrom}_{score}.txt'))
            if require_files:
                for filename in (specification.annotation_file,specification.setlist_file,
                                  specification.mask_definition_file,specification.variant_whitelist_file):
                    if filename is not None and not Path(filename).is_file():raise FileNotFoundError(filename)
            entries.append(specification)
        result[str(chrom)]=entries
    return result
