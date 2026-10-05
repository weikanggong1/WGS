"""Discovery significance and physical-distance lead/locus definitions."""
from pathlib import Path
import math
import pandas as pd
from .config import SignificanceConfig


def read_regenie(path):
    return pd.read_csv(path, sep=r'\s+', comment='#')


def select_loci(significant, *, lead_window_bp=500000, locus_merge_bp=1000000,
                excluded_locus_regions=((6, 25000000, 34000000),)):
    """Exclude closed candidate intervals, select min-P leads and merge by distance."""
    leads = []
    for chrom, group in significant.groupby('CHROM', sort=False):
        for excluded_chromosome, start, end in excluded_locus_regions:
            if str(chrom) == str(excluded_chromosome):
                group = group.loc[~group.GENPOS.between(start, end)]
        remaining = group.sort_values(['LOG10P','GENPOS'], ascending=[False,True]).copy()
        selected = []
        while not remaining.empty:
            lead = remaining.iloc[0].to_dict()
            selected.append(lead)
            remaining = remaining[(remaining.GENPOS-int(lead['GENPOS'])).abs() > lead_window_bp]
        selected.sort(key=lambda row:row['GENPOS'])
        clusters = []
        for lead in selected:
            if not clusters or lead['GENPOS']-clusters[-1][-1]['GENPOS'] > locus_merge_bp:
                clusters.append([lead])
            else: clusters[-1].append(lead)
        for locus in clusters:
            strongest = max(locus, key=lambda row:row['LOG10P'])
            leads.append({**strongest, 'LOCUS_START': max(1,min(x['GENPOS'] for x in locus)-lead_window_bp),
                          'LOCUS_END': max(x['GENPOS'] for x in locus)+lead_window_bp,
                          'N_LEADS':len(locus)})
    return pd.DataFrame(leads)


def summarize_results(single_files, gene_files, *, output_dir, config=None):
    """Read original-format results in chunks; save paper significant tables."""
    config = SignificanceConfig() if config is None else config
    destination = Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    def significant(paths, threshold, frequency_filter=False):
        parts = []
        for path in paths:
            for frame in pd.read_csv(path, sep=r'\s+', comment='#', chunksize=200000):
                if 'LOG10P.Y1' in frame: frame.rename(columns={'LOG10P.Y1':'LOG10P'},inplace=True)
                keep = pd.to_numeric(frame.LOG10P,errors='coerce') > -math.log10(threshold)
                if frequency_filter:
                    frequency=pd.to_numeric(frame.A1FREQ,errors='coerce')
                    if config.single_frequency_field=='maf':frequency=frequency.where(frequency<=.5,1-frequency)
                    keep &= frequency > config.single_frequency_min
                if keep.any(): parts.append(frame.loc[keep])
        return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['CHROM','GENPOS','LOG10P'])
    single = significant(single_files, config.single_threshold, frequency_filter=True)
    genes = significant(gene_files, config.gene_threshold)
    loci = select_loci(single,lead_window_bp=config.lead_window_bp,locus_merge_bp=config.locus_merge_bp,
                       excluded_locus_regions=config.excluded_locus_regions)
    single.to_csv(destination/'single_significant.tsv',sep='\t',index=False)
    genes.to_csv(destination/'gene_significant.tsv',sep='\t',index=False)
    loci.to_csv(destination/'single_loci.tsv',sep='\t',index=False)
    return {'single_significant':len(single),'gene_test_rows':len(genes),'loci':len(loci)}
