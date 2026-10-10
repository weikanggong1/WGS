import copy,importlib.util,unittest
import numpy as np
from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast as c
ZSTD=importlib.util.find_spec('zstandard') is not None
class Contracts(unittest.TestCase):
 def compare(self,s):
  m,n=map(int,s.shape);o,i,v=c.compact(s);c.validate(o,i,v,m,n)
  restored=np.zeros_like(s)
  for row,(a,b) in enumerate(zip(o[:-1],o[1:])):restored[row,i[int(a):int(b)]]=v[int(a):int(b)]
  np.testing.assert_array_equal(restored,s)
  counts=c.integer_counts(o,i,v,m,n)
  np.testing.assert_array_equal(counts['reference_alleles'],c.REF[s].sum(1))
  np.testing.assert_array_equal(counts['called_alleles'],c.CALLED[s].sum(1))
  np.testing.assert_array_equal(counts['half_missing_samples'],(s>=4).sum(1))
  if ZSTD:
   for level in (1,3):
    p,h=c.encode(s,level);oo,ii,vv=c.decode_compact(p,h)
    for a,b in zip((o,i,v),(oo,ii,vv)):np.testing.assert_array_equal(a,b)
    self.assertFalse(oo.flags.writeable)
 def test_six_states_halfmissing_and_reversed_samples(self):
  s=np.array([[0,1,2,3,4,5],[5,4,3,2,1,0],[0,0,0,0,0,0]],dtype=np.uint8)
  self.compare(s);self.compare(s[:,::-1]);self.compare(s[::-1])
 def test_all_missing_default_empty_and_noncontiguous(self):
  for s in [np.full((3,9),3,dtype=np.uint8),np.zeros((3,9),dtype=np.uint8),np.empty((0,9),dtype=np.uint8),np.empty((3,0),dtype=np.uint8),(np.arange(100).reshape(10,10)%6).astype(np.uint8)[::2,::2]]:self.compare(s)
 def test_sample_width_boundary(self):
  for n,dtype in [(65535,'<u2'),(65536,'<u4')]:
   s=np.zeros((1,n),dtype=np.uint8);s[0,-1]=5;o,i,v=c.compact(s);self.assertEqual(i.dtype,np.dtype(dtype));self.assertEqual(int(i[0]),n-1);self.compare(s)
 def test_budget_and_invalid_data(self):
  with self.assertRaises(MemoryError):c.compact(np.ones((2,3),dtype=np.uint8),max_raw_bytes=10)
  with self.assertRaises(ValueError):c.compact(np.array([[6]],dtype=np.uint8))
  o,i,v=c.compact(np.array([[1,2]],dtype=np.uint8))
  for oo,ii,vv in [(o,np.array([0,0],dtype='<u2'),v),(o,np.array([0,3],dtype='<u2'),v),(np.array([1,2],dtype='<u4'),i,v),(o,i,np.array([0,6],dtype=np.uint8))]:
   with self.assertRaises(ValueError):c.validate(oo,ii,vv,1,2)
 @unittest.skipUnless(ZSTD,'optional zstandard not installed')
 def test_rehashed_invalid_header_payload(self):
  p,h=c.encode(np.array([[0,4,5]],dtype=np.uint8))
  with self.assertRaises(ValueError):c.decode_compact(p+b'x',h)
  bad=copy.deepcopy(h);bad['sample_dtype']='<u4';bad['header_sha256']=c._header_sha(bad)
  with self.assertRaises(ValueError):c.decode_compact(p,bad)
  with self.assertRaises(MemoryError):c.decode_compact(p,h,max_raw_bytes=1)
if __name__=='__main__':unittest.main()
