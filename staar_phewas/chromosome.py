"""Complete single-trait chromosome schedules following the STAAR tutorial.

SPDX-License-Identifier: GPL-3.0-only
Coordinates, gene order and global array offsets are explicit reference inputs.
The planner never substitutes annotation candidates for the full gene catalog.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping


TUTORIAL_ANNOTATIONS = (
    'CADD', 'LINSIGHT', 'FATHMM.XF', 'aPC.EpigeneticActive',
    'aPC.EpigeneticRepressed', 'aPC.EpigeneticTranscription',
    'aPC.Conservation', 'aPC.LocalDiversity', 'aPC.Mappability',
    'aPC.TF', 'aPC.Protein',
)
CODING_MASKS = ('plof', 'plof_ds', 'missense', 'disruptive_missense',
                'synonymous', 'ptv', 'ptv_ds')
NONCODING_MASKS = ('upstream', 'downstream', 'UTR', 'promoter_CAGE',
                   'promoter_DHS', 'enhancer_CAGE', 'enhancer_DHS', 'ncRNA')


def _catalog_rows(rows, *, coding):
    if not isinstance(rows, list):
        raise ValueError('gene catalogs must be ordered lists')
    checked = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get('gene_name'), str) or not row['gene_name']:
            raise ValueError('every catalog row requires a nonempty gene_name')
        result = {'gene_name': row['gene_name']}
        if coding:
            start, end = row.get('start'), row.get('end')
            if not isinstance(start, int) or not isinstance(end, int) or start < 1 or start > end:
                raise ValueError('coding catalog requires one-based inclusive start/end')
            result.update(start=start, end=end)
        checked.append(result)
    return checked


def chromosome_configuration(config: Mapping, manifest: Mapping) -> dict:
    """Expand full catalogs and inclusive chromosome bounds to original files.

    ``manifest`` has chromosome, genes_info, ncRNA_genes, start_loc/end_loc,
    and array_offsets for coding/noncoding/ncrna/individual. The offsets count
    batches from preceding chromosomes, matching the tutorial's array IDs.
    Empty masks remain jobs; no statistical-result filtering occurs here.
    """
    if len(config.get('phenotypes', [])) != 1:
        raise ValueError('the full chromosome workflow requires one continuous phenotype')
    phenotype = config['phenotypes'][0]
    family = phenotype.get('family', phenotype.get('fit_options', {}).get('family', 'gaussian'))
    if family != 'gaussian' or phenotype.get('joint_mode') is not None or phenotype.get('fit_options', {}).get('joint_mode') is not None:
        raise ValueError('the full chromosome workflow requires one continuous phenotype')
    chromosome = str(manifest['chromosome'])
    if str(config.get('chromosome', chromosome)) != chromosome:
        raise ValueError('configuration and reference catalog chromosomes differ')
    genes = _catalog_rows(manifest['genes_info'], coding=True)
    ncrna = _catalog_rows(manifest['ncRNA_genes'], coding=False)
    for key, rows in (('number_coding_genes', genes), ('number_ncRNA_genes', ncrna)):
        if key in manifest and manifest[key] != len(rows):
            raise ValueError('reference manifest count does not match its complete gene catalog')
    start, end = manifest['start_loc'], manifest['end_loc']
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or start > end:
        raise ValueError('chromosome bounds must be one-based inclusive integers')
    offsets = manifest['array_offsets']
    kinds = ('coding', 'noncoding', 'ncrna', 'individual')
    if any(kind not in offsets or not isinstance(offsets[kind], int) or offsets[kind] < 0 for kind in kinds):
        raise ValueError('all four original global array offsets are required')
    coding_size = int(manifest.get('coding_genes_per_batch', 50))
    ncrna_size = int(manifest.get('ncrna_genes_per_batch', 100))
    width = int(manifest.get('individual_region_size', 10_000_000))
    if min(coding_size, ncrna_size, width) < 1:
        raise ValueError('batch sizes must be positive')
    output = Path(config['output_directory'])
    prefix = config.get('output_prefix', 'Imaging')
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError('output_prefix must be a nonempty file-name prefix')
    promoter_file = config['promoter_intervals_file']
    jobs = []
    for kind, label in (('coding', 'Coding'), ('noncoding', 'Noncoding')):
        for number, gene in enumerate(genes):
            array_id = offsets[kind] + number // coding_size + 1
            arguments = {'gene_name': gene['gene_name']}
            if kind == 'coding':
                arguments.update(start=gene['start'], end=gene['end'],
                                 category='all_categories_incl_ptv')
            else:
                arguments.update(category='all_categories',
                                 promoter_intervals_file=promoter_file)
            jobs.append({'name': f'{kind}:{number + 1}:{gene["gene_name"]}',
                         'kind': kind, 'arguments': arguments,
                         'output': str(output / f'{prefix}_{label}_{array_id}.Rdata'),
                         'layout': 'base'})
    for number, gene in enumerate(ncrna):
        array_id = offsets['ncrna'] + number // ncrna_size + 1
        jobs.append({'name': f'ncrna:{number + 1}:{gene["gene_name"]}',
                     'kind': 'ncrna', 'arguments': {'gene_name': gene['gene_name']},
                     'output': str(output / f'{prefix}_ncRNA_{array_id}.Rdata'),
                     'object_name': 'results_ncRNA', 'layout': 'base'})
    # Inclusive coverage also protects the exact-multiple boundary, where the
    # tutorial's ceil((end-start)/width) schedule otherwise drops the last base.
    count = (end - start) // width + 1
    for number in range(count):
        left = start + number * width
        right = min(left + width - 1, end)
        array_id = offsets['individual'] + number + 1
        jobs.append({'name': f'individual:{array_id}', 'kind': 'individual',
                     'arguments': {'start': left, 'end': right,
                                   'variant_type': 'variant',
                                   'mac_cutoff': config.get('mac_cutoff', 20),
                                   'subset_variants_num': config.get('subset_variants_num', 5000)},
                     'output': str(output / f'{prefix}_Individual_Analysis_{array_id}.Rdata'),
                     'layout': 'base'})
    selected = ('phenotypes', 'qc_path', 'annotation_catalog', 'analysis_options',
                'statistics_execution', 'debug_json', 'matmul_mode', 'precision_control',
                'stage_profile', 'resident_genotypes', 'tf32_split_k', 'individual_genotype_block_size',
                'single_batch_optimization', 'individual_effective_block_size',
                'statistics_tail_optimization', 'local_mask_reuse', 'weight_batch_optimization', 'weighted_eigensolver')
    if {'tf32_binned_tile_shape', 'tf32_binned_fused_small'} & config.keys():
        raise ValueError('Removed TF32 reconstruction parameters')
    expanded = {key: copy.deepcopy(config[key]) for key in selected if key in config}
    expanded['annotation_names'] = list(config.get('annotation_names', TUTORIAL_ANNOTATIONS))
    expanded['require_single_continuous'] = True
    from .tf32 import validate_mode, validate_split_k
    expanded.setdefault('matmul_mode', 'tf32')
    validate_mode(expanded['matmul_mode'])
    expanded.setdefault('tf32_split_k', 0)
    validate_split_k(expanded['tf32_split_k'])
    expanded.setdefault('statistics_tail_optimization', True)
    expanded.setdefault('local_mask_reuse', True)
    expanded.setdefault('weight_batch_optimization', True)
    if expanded.get('statistics_execution', 'serial') != 'serial':
        raise ValueError('complete chromosome GPU validation requires serial statistics execution')
    association_options = expanded.setdefault('analysis_options', {})
    if association_options.get('wrapper_semantics', 'base') != 'base':
        raise ValueError('the original base chromosome schedule requires wrapper_semantics=base')
    association_options['wrapper_semantics'] = 'base'
    expanded['qc_path'] = config.get('qc_path', 'annotation/info/QC_label')
    expanded['statistics_execution'] = config.get('statistics_execution', 'serial')
    if expanded['statistics_execution'] != 'serial':
        raise ValueError('statistics_execution must be serial')
    expanded['chromosomes'] = [{'name': chromosome, 'gds': config['gds'],
                               'annotation_index': {'promoter_intervals_file': promoter_file},
                               'jobs': jobs}]
    expanded['coverage'] = {
        'chromosome': chromosome, 'coding_genes': len(genes),
        'noncoding_genes': len(genes), 'ncrna_genes': len(ncrna),
        'coding_masks': list(CODING_MASKS), 'noncoding_masks': list(NONCODING_MASKS),
        'individual_start': start, 'individual_end': end,
        'individual_batches': count,
        'native_files': len({job['output'] for job in jobs}),
        'annotation_names': expanded['annotation_names'],
    }
    return expanded


def run_chromosome(config, *, device='cuda'):
    """Run every scheduled gene/mask and individual region for one trait."""
    manifest = config['manifest']
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    expanded = chromosome_configuration(config, manifest)
    from .cli import run_configuration
    report = run_configuration(expanded, device=device)
    report['coverage'] = expanded['coverage']
    return report


def main():
    parser = argparse.ArgumentParser(description='Full single-trait STAAR chromosome analysis')
    parser.add_argument('config', type=Path, help='Private JSON configuration and complete reference manifest')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--report', type=Path, required=True, help='Aggregate timing and coverage JSON')
    parser.add_argument('--plan-only', action='store_true', help='Write an expanded private job configuration')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.plan_only:
        manifest = config['manifest']
        if isinstance(manifest, str):
            manifest = json.loads(Path(manifest).read_text())
        report = chromosome_configuration(config, manifest)
    else:
        report = run_chromosome(config, device=args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
