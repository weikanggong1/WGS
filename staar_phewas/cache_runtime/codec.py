"""Private six-state cache codec; CPU packed decompression, no GDS/GPU.

Input states have shape (variants, selected samples). State 0..5 maps to
(reference allele count, called allele count): (2,2),(1,2),(0,2),(0,0),
(1,1),(0,1). States 4/5 retain half-missing evidence. This is association
state equivalence, not reconstruction of original allele codes/lane ordering.
A max_packed_bytes limit bounds serialized packed size only, not total RAM:
resident input, bit extraction, codec scratch and compressed bytes also exist.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
VERSION=1
FORMAT='private-reference-six-state-bitplanes'
STATE_ENCODING='ref-homo-first-six-v1'
MAPPING=[[2,2],[1,2],[0,2],[0,0],[1,1],[0,1]]
LAYOUTS=('plane-major','variant-major')
DEFAULT_MAX_PACKED_BYTES=256*2**20
FORMULA='3*m*ceil(n/8)'
def _sha(b):return hashlib.sha256(b).hexdigest()
def _header_sha(meta):
 return _sha(json.dumps({k:v for k,v in meta.items() if k!='header_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def _integer(v,name):
 if type(v) is not int or v<0:raise ValueError(name+' must be a nonnegative integer')
 return v

def packed_size(m,n):
 return 3*_integer(m,'m')*((_integer(n,'n')+7)//8)
def _budget(size,budget):
 if type(budget) is not int or budget<0:raise ValueError('max_packed_bytes must be nonnegative integer')
 if size>budget:raise MemoryError('Packed output exceeds explicit byte budget')

def _states(states):
 s=np.asarray(states)
 if s.ndim!=2 or s.dtype.kind not in 'iu' or s.dtype.kind=='b':raise ValueError('states must be a 2D integer array')
 if s.size and (s.min()<0 or s.max()>5):raise ValueError('Only six states 0..5 are valid')
 return s

def pack_plane(states,*,layout='plane-major',max_packed_bytes=DEFAULT_MAX_PACKED_BYTES):
 s=_states(states);m,n=map(int,s.shape)
 if layout not in LAYOUTS:raise ValueError('Unsupported bitplane layout')
 _budget(packed_size(m,n),max_packed_bytes)
 # Preallocate once; each bitplane extraction is a bounded CPU temporary.
 packed=np.empty((3,m,(n+7)//8),dtype=np.uint8)
 for bit in range(3):packed[bit]=np.packbits(((s>>bit)&1).astype(np.uint8),axis=1,bitorder='little')
 if layout=='variant-major':packed=packed.transpose(1,0,2).copy()
 return packed.tobytes(order='C')

def _validate_padding(raw,m,n,layout):
 B=(n+7)//8
 if len(raw)!=packed_size(m,n):raise ValueError('Packed byte count mismatch')
 arr=np.frombuffer(raw,dtype=np.uint8).reshape((3,m,B) if layout=='plane-major' else (m,3,B))
 if n%8 and m and np.any(arr[...,-1] & np.uint8(255 ^ ((1<<(n%8))-1))):raise ValueError('Nonzero high padding bits')
 return arr

def unpack_plane_numpy(packed,m,n,*,layout='plane-major',max_packed_bytes=DEFAULT_MAX_PACKED_BYTES):
 m=_integer(m,'m');n=_integer(n,'n')
 if layout not in LAYOUTS:raise ValueError('Unsupported bitplane layout')
 _budget(packed_size(m,n),max_packed_bytes)
 raw=bytes(packed);arr=_validate_padding(raw,m,n,layout)
 if layout=='variant-major':arr=arr.transpose(1,0,2)
 states=np.zeros((m,n),dtype=np.uint8)
 for bit in range(3):states|=np.unpackbits(arr[bit],axis=1,count=n,bitorder='little')<<bit
 if states.size and np.any(states>5):raise ValueError('Packed data contains invalid state 6/7')
 return states

def _zstd():
 try:import zstandard
 except ImportError as e:raise ImportError('Explicit optional zstandard package is required for compression') from e
 return zstandard

def encode(states,level=3,*,layout='plane-major',max_packed_bytes=DEFAULT_MAX_PACKED_BYTES):
 if type(level) is not int or level not in (1,3,6):raise ValueError('Supported probe levels are 1,3,6')
 s=_states(states);m,n=map(int,s.shape);raw=pack_plane(s,layout=layout,max_packed_bytes=max_packed_bytes)
 payload=_zstd().ZstdCompressor(level=level,write_content_size=True).compress(raw)
 meta=dict(format=FORMAT,version=VERSION,codec='zstd',level=level,layout=layout,m=m,n=n,bytes_per_variant_plane=(n+7)//8,bitorder='little',state_encoding=STATE_ENCODING,state_mapping=[row.copy() for row in MAPPING],
  packed_size_formula=FORMULA,packed_bytes=len(raw),compressed_bytes=len(payload),rawpack_sha256=_sha(raw),compressed_sha256=_sha(payload))
 meta['header_sha256']=_header_sha(meta)
 _header(meta,max_packed_bytes)
 return payload,meta

def _header(meta,max_packed_bytes):
 if not isinstance(meta,dict):raise ValueError('Header must be an object')
 keys={'format','version','codec','level','layout','m','n','bytes_per_variant_plane','bitorder','state_encoding','state_mapping','packed_size_formula','packed_bytes','compressed_bytes','rawpack_sha256','compressed_sha256','header_sha256'}
 if set(meta)!=keys:raise ValueError('Header keys mismatch')
 if meta['header_sha256']!=_header_sha(meta):raise ValueError('Header checksum mismatch')
 if meta['format']!=FORMAT or type(meta['version']) is not int or meta['version']!=VERSION or meta['codec']!='zstd' or meta['bitorder']!='little' or meta['state_encoding']!=STATE_ENCODING or meta['state_mapping']!=MAPPING or meta['packed_size_formula']!=FORMULA:raise ValueError('Unsupported header semantics')
 if meta['layout'] not in LAYOUTS or type(meta['level']) is not int or meta['level'] not in (1,3,6):raise ValueError('Unsupported layout/level')
 m=_integer(meta['m'],'m');n=_integer(meta['n'],'n');size=packed_size(m,n);_budget(size,max_packed_bytes)
 if _integer(meta['bytes_per_variant_plane'],'bytes_per_variant_plane')!=(n+7)//8 or _integer(meta['packed_bytes'],'packed_bytes')!=size:raise ValueError('Header dimension/size formula mismatch')
 if _integer(meta['compressed_bytes'],'compressed_bytes')>max_packed_bytes+2**20:raise MemoryError('Compressed payload exceeds packed budget plus fixed 1MiB allowance')
 for key in ('rawpack_sha256','compressed_sha256'):
  value=meta[key]
  if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid checksum field')
 return m,n,size

def decode_packed(payload,meta,*,max_packed_bytes=DEFAULT_MAX_PACKED_BYTES):
 m,n,size=_header(meta,max_packed_bytes);compressed=bytes(payload)
 if len(compressed)!=meta['compressed_bytes'] or _sha(compressed)!=meta['compressed_sha256']:raise ValueError('Compressed checksum/size mismatch')
 zstd=_zstd()
 # Declared content length is required, preventing an unbounded header allocation.
 if zstd.frame_content_size(compressed)!=size:raise ValueError('Zstd frame content size mismatch')
 raw=zstd.ZstdDecompressor().decompress(compressed,max_output_size=max(1,size),allow_extra_data=False)
 if len(raw)!=size or _sha(raw)!=meta['rawpack_sha256']:raise ValueError('Raw packed checksum/size mismatch')
 _validate_padding(raw,m,n,meta['layout'])
 # Check invalid six-state patterns without constructing a dense m*n array.
 B=(n+7)//8
 planes=np.frombuffer(raw,dtype=np.uint8).reshape((3,m,B) if meta['layout']=='plane-major' else (m,3,B))
 if meta['layout']=='variant-major':planes=planes.transpose(1,0,2)
 if np.any(planes[2]&planes[1]):raise ValueError('Packed data contains invalid state 6/7')
 return raw

def write_cache(directory,states,level=3,*,layout='plane-major',max_packed_bytes=DEFAULT_MAX_PACKED_BYTES):
 destination=Path(directory)
 if destination.exists():raise FileExistsError('Refuse to overwrite cache')
 payload,meta=encode(states,level,layout=layout,max_packed_bytes=max_packed_bytes)
 destination.parent.mkdir(parents=True,exist_ok=True)
 staging=Path(tempfile.mkdtemp(prefix='.six-state-',dir=destination.parent))
 try:
  for name,data in [('payload.zst',payload),('header.json',(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n').encode())]:
   with (staging/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  fd=os.open(staging,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  if destination.exists():raise FileExistsError('Refuse to overwrite cache')
  os.rename(staging,destination)
  fd=os.open(destination.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 except BaseException:
  if staging.exists():shutil.rmtree(staging)
  raise
 return meta

def read_cache(directory,*,max_packed_bytes=DEFAULT_MAX_PACKED_BYTES):
 directory=Path(directory)
 meta=json.loads((directory/'header.json').read_text())
 _header(meta,max_packed_bytes)
 if (directory/'payload.zst').stat().st_size!=meta['compressed_bytes']:raise ValueError('Payload file length mismatch')
 return decode_packed((directory/'payload.zst').read_bytes(),meta,max_packed_bytes=max_packed_bytes),meta
