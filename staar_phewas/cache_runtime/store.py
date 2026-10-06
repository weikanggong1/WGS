"""Private resumable six-state CSR frame container. No genotype/GPU cache.

Frames cover original physical columns consecutively, including all-zero and
zero-layer sites. Counts apply only to the bound complete ordered sample axis.
CSR and count frames are independently checksummed Zstd; original GDS retained.
A completed container is immutable. Incomplete recovery requires exact binding.
"""
import hashlib, json, os
from pathlib import Path
import numpy as np
from . import sparse_codec_fast as csr
from . import codec
FORMAT='private-six-state-csr-container-v1'
CHUNK=1024
INDEX=np.dtype([('start','<u8'),('m','<u4'),('offset','<u8'),('size','<u8'),('header_offset','<u8'),('header_size','<u4'),('counts_offset','<u8'),('counts_size','<u8'),('payload_sha','S64'),('counts_sha','S64')])
def sha(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,b):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def usage(p):
 files=[f for f in Path(p).iterdir() if f.is_file()]
 return dict(logical_bytes=sum(f.stat().st_size for f in files),allocated_bytes=sum(getattr(f.stat(),'st_blocks',0)*512 for f in files))
def _append(p,b):
 with p.open('ab') as f:
  pos=f.tell();f.write(b);f.flush();os.fsync(f.fileno())
 return pos
class Writer:
 def __init__(self,path,binding,samples,m,*,source_bytes,storage_fraction=.10):
  self.path=Path(path);self.samples=np.asarray(samples)
  if self.samples.dtype!=np.int64 or self.samples.ndim!=1 or len(np.unique(self.samples))!=len(self.samples):raise ValueError('Physical sample axis must be unique int64')
  if type(m)is not int or m<0 or source_bytes<=0 or not 0<storage_fraction<=.10:raise ValueError('Invalid dimension/storage budget')
  self.spec=dict(format=FORMAT,state_encoding=codec.STATE_ENCODING,m=m,n=len(samples),chunk=CHUNK,binding=binding,sample_sha256=sha(self.samples.astype('<i8').tobytes()),source_bytes=source_bytes,storage_cap_bytes=int(source_bytes*storage_fraction))
  if self.path.exists():
   if (self.path/'COMPLETE').exists():raise FileExistsError('Completed cache cannot be overwritten')
   if json.loads((self.path/'manifest.pending.json').read_text())!=self.spec:raise ValueError('Resume source/sample binding mismatch')
   actual=np.load(self.path/'samples.npy',allow_pickle=False)
   if not np.array_equal(actual,self.samples) or actual.dtype!=self.samples.dtype:raise ValueError('Resume sample axis mismatch')
   self.rows=json.loads((self.path/'journal.json').read_text())
  else:
   self.path.mkdir(parents=True,mode=0o700)
   atomic(self.path/'manifest.pending.json',canonical(self.spec));np.save(self.path/'samples.npy',self.samples,allow_pickle=False)
   for name in ('data.bin','headers.bin','counts.bin'):(self.path/name).touch()
   self.rows=[];atomic(self.path/'journal.json',canonical(self.rows))
  self._recover()
 def _recover(self):
  ends=dict(data=0,header=0,counts=0);start=0
  for row in self.rows:
   if row['start']!=start or row['m']!=min(CHUNK,self.spec['m']-start):raise ValueError('Nonconsecutive committed frame')
   for kind,key,size in [('data','offset','size'),('header','header_offset','header_size'),('counts','counts_offset','counts_size')]:
    if row[key]!=ends[kind]:raise ValueError('Frame stream gap')
    ends[kind]+=row[size]
   _read_frame(self.path,row,self.spec);start+=row['m']
  for kind,name in [('data','data.bin'),('header','headers.bin'),('counts','counts.bin')]:
   p=self.path/name
   if p.stat().st_size<ends[kind]:raise ValueError('Committed stream truncated')
   with p.open('r+b') as f:f.truncate(ends[kind])
  self.next_start=start
 def _budget(self,extra):
  u=usage(self.path);cap=self.spec['storage_cap_bytes']
  # Logical projection plus per-file alignment/headroom; no filesystem quota guarantee.
  if max(u.values())+extra+65536>cap:raise MemoryError('Additional cache storage exceeds configured fraction of retained GDS')
 def append(self,states,original_counts):
  s=codec._states(states);m,n=map(int,s.shape);start=self.next_start
  if n!=self.spec['n'] or m!=min(CHUNK,self.spec['m']-start) or m<=0:raise ValueError('Frame shape/order mismatch')
  payload,meta=csr.encode(s,level=3);arrays=csr.decode_compact(payload,meta)
  direct=csr.compact(s);counts=csr.integer_counts(*arrays,m,n)
  if not all(np.array_equal(a,b) for a,b in zip(arrays,direct)):raise ValueError('CSR roundtrip mismatch')
  keys=('reference_alleles','called_alleles','half_missing_samples')
  if not all(np.asarray(original_counts[k]).dtype==np.int64 and np.array_equal(counts[k],original_counts[k]) for k in keys):raise ValueError('Original integer summary differs before writing')
  raw=np.column_stack([counts[k] for k in keys]).astype('<i8').tobytes();cp=codec._zstd().ZstdCompressor(level=3,write_content_size=True).compress(raw)
  header=canonical(dict(csr_meta=meta,counts_raw_sha256=sha(raw),counts_raw_bytes=len(raw),counts_compressed_sha256=sha(cp),counts_compressed_bytes=len(cp)))
  row=dict(start=start,m=m,offset=(self.path/'data.bin').stat().st_size,size=len(payload),header_offset=(self.path/'headers.bin').stat().st_size,header_size=len(header),counts_offset=(self.path/'counts.bin').stat().st_size,counts_size=len(cp),payload_sha=sha(payload),counts_sha=sha(cp))
  self._budget(len(payload)+len(cp)+len(header)+len(canonical(row))*2)
  for name,b in [('data.bin',payload),('counts.bin',cp),('headers.bin',header)]:_append(self.path/name,b)
  self.rows.append(row);atomic(self.path/'journal.json',canonical(self.rows));self.next_start+=m
  if max(usage(self.path).values())>self.spec['storage_cap_bytes']:raise MemoryError('Measured cache storage cap exceeded')
 def finish(self):
  if self.next_start!=self.spec['m']:raise ValueError('Incomplete physical axis')
  a=np.empty(len(self.rows),dtype=INDEX)
  for i,row in enumerate(self.rows):a[i]=tuple(row[k] for k in INDEX.names)
  tmp=self.path/'index.npy.tmp'
  with tmp.open('wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,self.path/'index.npy')
  hashes={name:file_sha(self.path/name) for name in ('data.bin','headers.bin','counts.bin','samples.npy','index.npy')}
  manifest=dict(self.spec,frames=len(a),files_sha256=hashes,storage=usage(self.path),association_precision_accepted=False)
  b=canonical(manifest);self._budget(len(b)+128)
  atomic(self.path/'manifest.json',b);atomic(self.path/'COMPLETE',sha(b).encode())
  if max(usage(self.path).values())>self.spec['storage_cap_bytes']:(self.path/'COMPLETE').unlink();raise MemoryError('Final measured storage cap exceeded')
  return manifest

def _range(path,offset,size):
 if offset<0 or size<0 or offset+size>path.stat().st_size:raise ValueError('Frame out of bounds')
 with path.open('rb') as f:f.seek(offset);b=f.read(size)
 if len(b)!=size:raise ValueError('Short frame read')
 return b
def _read_frame(path,row,spec):
 header=json.loads(_range(path/'headers.bin',int(row['header_offset']),int(row['header_size'])))
 payload=_range(path/'data.bin',int(row['offset']),int(row['size']));cp=_range(path/'counts.bin',int(row['counts_offset']),int(row['counts_size']))
 for key,b in [('payload_sha',payload),('counts_sha',cp)]:
  expected=row[key];expected=expected.decode() if isinstance(expected,bytes) else str(expected)
  if sha(b)!=expected:raise ValueError('Frame SHA mismatch')
 meta=header['csr_meta'];m,n=meta['m'],meta['n']
 if m!=int(row['m']) or n!=spec['n'] or header['counts_raw_bytes']!=m*24 or header['counts_compressed_bytes']!=len(cp) or sha(cp)!=header['counts_compressed_sha256']:raise ValueError('Frame dimensions/count metadata mismatch')
 arrays=csr.decode_compact(payload,meta)
 if codec._zstd().frame_content_size(cp)!=m*24:raise ValueError('Counts frame content size mismatch')
 raw=codec._zstd().ZstdDecompressor().decompress(cp,max_output_size=max(1,m*24),allow_extra_data=False)
 if len(raw)!=m*24 or sha(raw)!=header['counts_raw_sha256']:raise ValueError('Counts raw SHA mismatch')
 c=np.frombuffer(raw,dtype='<i8').reshape(m,3)
 # Counts generated and proven at conversion, checksummed thereafter; no repeated CSR count reduction.
 if np.any(c<0) or np.any(c[:,0]>c[:,1]) or np.any(c[:,1]>2*n) or np.any(c[:,2]>n):raise ValueError('Impossible cached integer summary')
 return dict(offsets=arrays[0],sample_index=arrays[1],state=arrays[2],reference_alleles=c[:,0],called_alleles=c[:,1],half_missing_samples=c[:,2],m=m,n=n,start=int(row['start']))
class Container:
 def __init__(self,path,expected_source_binding=None,expected_samples=None):
  self.path=Path(path);b=(self.path/'manifest.json').read_bytes()
  if (self.path/'COMPLETE').read_text()!=sha(b):raise ValueError('Missing/invalid completed marker')
  self.manifest=json.loads(b);s=self.manifest
  if s['format']!=FORMAT or s['state_encoding']!=codec.STATE_ENCODING or s['chunk']!=CHUNK:raise ValueError('Container semantic mismatch')
  if expected_source_binding is not None and s['binding']!=expected_source_binding:raise ValueError('Source binding mismatch')
  self.samples=np.load(self.path/'samples.npy',allow_pickle=False);self.index=np.load(self.path/'index.npy',allow_pickle=False)
  if self.samples.dtype!=np.int64 or self.samples.shape!=(s['n'],) or sha(self.samples.astype('<i8').tobytes())!=s['sample_sha256']:raise ValueError('Sample binding mismatch')
  if expected_samples is not None and not np.array_equal(self.samples,expected_samples):raise ValueError('Requested sample axis differs')
  if self.index.dtype!=INDEX or len(self.index)!=s['frames'] or len(self.index)!=(s['m']+CHUNK-1)//CHUNK:raise ValueError('Invalid frame index')
  for name in ('samples.npy','index.npy'):
   if file_sha(self.path/name)!=s['files_sha256'][name]:raise ValueError('Index/sample file checksum mismatch')
  start=0;ends=[0,0,0]
  for row in self.index:
   if int(row['start'])!=start or int(row['m'])!=min(CHUNK,s['m']-start):raise ValueError('Index physical order mismatch')
   for j,(offset,size) in enumerate([('offset','size'),('header_offset','header_size'),('counts_offset','counts_size')]):
    if int(row[offset])!=ends[j]:raise ValueError('Index stream gap')
    ends[j]+=int(row[size])
   start+=int(row['m'])
  for end,name in zip(ends,('data.bin','headers.bin','counts.bin')):
   if (self.path/name).stat().st_size!=end:raise ValueError('Stream length mismatch')
  self.samples.flags.writeable=False;self.index.flags.writeable=False;self.complete=True
 def read_frame(self,frame_id):
  if type(frame_id)is not int or not 0<=frame_id<len(self.index):raise IndexError('Frame ID out of bounds')
  return _read_frame(self.path,{k:self.index[frame_id][k] for k in INDEX.names},self.manifest)
 def verify_streams(self):
  for name,h in self.manifest['files_sha256'].items():
   if file_sha(self.path/name)!=h:raise ValueError('Container file checksum mismatch')
  return True
open_container=Container
