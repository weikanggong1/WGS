"""Development-only parallel runner for untouched original R functions."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,help='Host path to original package gene manifest')
    parser.add_argument('--config',required=True,help='Configuration path visible inside the R oracle')
    parser.add_argument('--runner',required=True,help='Existing Rscript or development container helper')
    parser.add_argument('--oracle',required=True,help='Oracle script path visible to the runner')
    parser.add_argument('--output-directory',type=Path,required=True,help='Host log and timing summary directory')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    manifest=json.loads(args.manifest.read_text())
    counts={'coding':(len(manifest['genes_info'])+manifest['coding_genes_per_batch']-1)//manifest['coding_genes_per_batch'],
            'noncoding':(len(manifest['genes_info'])+manifest['coding_genes_per_batch']-1)//manifest['coding_genes_per_batch'],
            'ncrna':(len(manifest['ncRNA_genes'])+manifest['ncrna_genes_per_batch']-1)//manifest['ncrna_genes_per_batch'],
            'individual':manifest['number_individual_arrays']}
    tasks=[f'{kind}:{group}'for kind,count in counts.items()for group in range(1,count+1)]
    args.output_directory.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    started=time.perf_counter();report={'workers':args.workers,'blas_threads':1,'task_counts':counts,'completed':[]}
    def run(task):
        task_started=time.perf_counter()
        with (args.output_directory/f'{task.replace(":","_")}.log').open('wb')as log:
            process=subprocess.run([args.runner,args.oracle,args.config,task],stdout=log,stderr=subprocess.STDOUT,env=env)
        return {'task':task,'returncode':process.returncode,'process_wall_seconds':time.perf_counter()-task_started}
    with ThreadPoolExecutor(max_workers=args.workers)as executor:
        futures=[executor.submit(run,task)for task in tasks]
        for future in as_completed(futures):
            result=future.result();report['completed'].append(result)
            report['elapsed_parallel_wall_seconds']=time.perf_counter()-started
            (args.output_directory/'parallel_reference_progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(result),flush=True)
    report['parallel_wall_seconds']=time.perf_counter()-started
    report['sum_process_wall_seconds']=sum(item['process_wall_seconds']for item in report['completed'])
    report['all_processes_passed']=all(item['returncode']==0 for item in report['completed'])
    (args.output_directory/'parallel_reference_report.json').write_text(json.dumps(report,indent=2))
    if not report['all_processes_passed']:raise SystemExit('An original R batch failed; inspect its separate log')


if __name__=='__main__':main()
