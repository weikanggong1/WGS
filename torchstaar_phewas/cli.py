"""Private configuration entry point for shared independent association runs."""
import argparse
import importlib
import json
from pathlib import Path
from . import CacheSpec, run_configuration


def configuration_inputs(configuration):
    """Read standalone JSONs and explicit live source-proof callables."""
    analyses = []
    for value in configuration['analyses']:
        analyses.append(json.loads(Path(value).read_text()) if isinstance(value, str) else value)
    specs = {}
    for item in configuration['caches']:
        gds = Path(item['gds']).resolve()
        if gds in specs:raise ValueError('duplicate cache specification')
        binding = item['expected_binding']
        if isinstance(binding, str):binding = json.loads(Path(binding).read_text())
        reference = item['source_proof']
        if not isinstance(reference, str) or reference.count(':') != 1:
            raise ValueError('source_proof must identify a live module:callable')
        module, name = reference.split(':')
        if not module or not name:raise ValueError('source_proof must identify a live module:callable')
        proof = getattr(importlib.import_module(module), name)
        if not callable(proof):raise ValueError('source_proof must be callable')
        samples = None
        if 'expected_samples_file' in item:
            import numpy as np
            samples = np.load(item['expected_samples_file'], allow_pickle=False)
        specs[gds] = CacheSpec(Path(item['directory']), binding, proof, samples,
                              item.get('compact_cache_bytes', 64*2**20))
    return analyses, specs


def _positive_cpu_threads(value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('cpu-threads must be a positive integer') from None
    if result < 1:
        raise argparse.ArgumentTypeError('cpu-threads must be a positive integer')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent Torchstaar analyses sharing verified genotype IO')
    parser.add_argument('config', type=Path, help='Private JSON with standalone analyses and explicit cache proofs')
    parser.add_argument('--device', default='cuda:0', help='CUDA device, default cuda:0')
    parser.add_argument('--cpu-threads', type=_positive_cpu_threads, help='PyTorch CPU intra-op threads; overrides shared_options.cpu_threads, Python default 2')
    parser.add_argument('--report', type=Path, required=True, help='Private execution report JSON')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    analyses, specs = configuration_inputs(config)
    options = config.get('shared_options', {})
    if not isinstance(options, dict):raise ValueError('shared_options must be an object')
    options = dict(options)
    unknown = set(options) - {'device_cache_bytes', 'compact_cache_bytes', 'metadata_cache_bytes', 'cpu_threads'}
    if unknown:raise ValueError('unknown shared options: ' + ', '.join(sorted(unknown)))
    if args.cpu_threads is not None:options['cpu_threads'] = args.cpu_threads
    report = run_configuration(analyses, cache_specs=specs, device=args.device, **options)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':main()
