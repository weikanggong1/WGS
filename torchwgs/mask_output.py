"""Optional REGENIE _masks BED/BIM/FAM/snplist artifacts, streamed by gene."""
from pathlib import Path
import numpy as np


class MaskWriter:
    def __init__(self,prefix,sample_ids,sample_sex):
        self.prefix=str(prefix)+'_masks'
        Path(prefix).parent.mkdir(parents=True,exist_ok=True)
        self.n=len(sample_ids);self.stride=(self.n+3)//4
        self.n_masks=0
        self.closed=False
        self.bed=open(self.prefix+'.bed.partial','wb');self.bed.write(b'\x6c\x1b\x01')
        self.bim=open(self.prefix+'.bim.partial','w');self.snplist=open(self.prefix+'.snplist.partial','w')
        with open(self.prefix+'.fam.partial','w') as f:
            for (fid,iid),sex in zip(sample_ids,sample_sex):f.write(f'{fid}\t{iid}\t0\t0\t{sex}\t-9\n')

    def __call__(self,artifacts):
        gene=artifacts.gene
        for mask in artifacts.masks:
            frequency='all' if mask.aaf_upper==1 else mask.frequency
            identifier=f'{gene.gene}.{mask.name}.{frequency}'
            allele=f'{mask.base_name}.{frequency}'
            raw=getattr(mask,'raw_burden',None)
            g=(mask.burden if raw is None else raw).detach().cpu().numpy()
            valid=np.isfinite(g)
            calls=np.floor(np.where(valid,g,0)+.5).astype(np.int64)
            if np.any(valid&((calls<0)|(calls>2))):raise ValueError('Max mask BED calls must lie in 0..2')
            codes=np.zeros(self.stride*4,dtype=np.uint8)
            codes[:self.n]=np.array([3,2,0],dtype=np.uint8)[calls]
            codes[:self.n][~valid]=1
            packed=(codes.reshape(-1,4)<<np.array([0,2,4,6],dtype=np.uint8)).sum(1).astype(np.uint8)
            self.bed.write(packed.tobytes())
            self.bim.write(f'{gene.chrom}\t{identifier}\t0\t{gene.position}\t{allele}\tref\n')
            self.snplist.write(identifier+'\t'+','.join(mask.variant_ids)+'\n')
            self.n_masks+=1

    def close(self,commit=True):
        if self.closed:return
        self.closed=True
        self.bed.close();self.bim.close();self.snplist.close()
        for suffix in ('.bed','.bim','.fam','.snplist'):
            partial=Path(self.prefix+suffix+'.partial')
            if commit:partial.replace(self.prefix+suffix)
            else:partial.unlink(missing_ok=True)

    def __enter__(self):return self
    def __exit__(self,*exc):self.close(commit=exc[0] is None)
