"""Constrained installed official CUDA-library discovery; no CUDA work on import."""
import base64
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def cuda_build_version(value):
    try:major,minor=map(int,str(value).split('.')[:2])
    except (TypeError,ValueError):raise RuntimeError('CUDA-enabled Torch build required')
    if major not in (11,12) or minor<0:raise RuntimeError('Solver supports explicit Linux CUDA11/12 ABI only')
    return major,minor,major*1000+minor*10

@dataclass(frozen=True)
class InstalledLibraries:
    solver:Path
    runtime:Path
    origin:str
    package_version:str
    cuda_build:int
    solver_sha256:str
    runtime_sha256:str
    package_hash_verified:bool


def _wheel_owned(path,distribution):
    matching=[v for v in distribution.files or () if Path(distribution.locate_file(v)).resolve()==path.resolve()]
    if not matching:raise RuntimeError('CUDA library lacks installed NVIDIA wheel file ownership')
    verified=False
    for item in matching:
        record_hash=getattr(item,'hash',None)
        if record_hash is None:continue
        if record_hash.mode!='sha256':raise RuntimeError('Unsupported wheel content hash algorithm')
        actual=base64.urlsafe_b64encode(bytes.fromhex(file_sha256(path))).decode().rstrip('=')
        if actual!=record_hash.value:raise RuntimeError('Installed NVIDIA wheel RECORD SHA mismatch')
        verified=True
    return verified


def _conda_owned(path,prefix,names):
    relative=path.relative_to(prefix).as_posix()
    for name in names:
        for record in sorted((prefix/'conda-meta').glob(name+'-*.json')):
            metadata=json.loads(record.read_text())
            if metadata.get('name')!=name:continue
            channel=str(metadata.get('channel','')).rstrip('/')
            if not (channel in ('nvidia','https://conda.anaconda.org/nvidia') or channel.startswith('https://conda.anaconda.org/nvidia/')):
                continue
            owned=[v for v in metadata.get('files',[]) if (prefix/v).resolve()==path.resolve()]
            if relative not in metadata.get('files',[]) and not owned:continue
            verified=False
            for entry in metadata.get('paths_data',{}).get('paths',[]):
                if entry.get('path_type')!='softlink' and (prefix/entry.get('_path','')).resolve()==path.resolve() and entry.get('sha256'):
                    if file_sha256(path)!=entry['sha256']:raise RuntimeError('Official Conda installed file SHA mismatch')
                    verified=True
            return str(metadata['version']),verified
    raise RuntimeError('Library lacks active official NVIDIA Conda package ownership')


def resolve_libraries(torch_module,*,environ=None,distribution_getter=None,
                      expected_solver_sha256=None,expected_runtime_sha256=None):
    """Resolve a matched provider pair; never search system loader paths.

    SHA is checked against package RECORD/paths metadata when available, and
    against caller's explicit expected hash when supplied. Always record SHA.
    """
    if sys.platform!='linux':raise RuntimeError('Linux .so resolver only')
    major,minor,build=cuda_build_version(torch_module.version.cuda)
    env=os.environ if environ is None else environ
    get=importlib.metadata.distribution if distribution_getter is None else distribution_getter
    site=Path(torch_module.__file__).resolve().parent.parent
    solver=site/'nvidia/cusolver/lib/libcusolver.so.11'
    runtime_name='libcudart.so.11.0' if major==11 else 'libcudart.so.12'
    runtime=site/'nvidia/cuda_runtime/lib'/runtime_name
    if solver.is_file() and runtime.is_file():
        solver_dist=get('nvidia-cusolver-cu'+str(major));runtime_dist=get('nvidia-cuda-runtime-cu'+str(major))
        verified=_wheel_owned(solver,solver_dist)&_wheel_owned(runtime,runtime_dist)
        origin='installed NVIDIA Torch wheel pair';version=str(solver_dist.version)
    else:
        prefix=Path(env.get('CONDA_PREFIX',''))
        if not env.get('CONDA_PREFIX') or not prefix.is_absolute():raise RuntimeError('No matched wheel pair or active Conda prefix')
        prefix=prefix.resolve()
        if not Path(torch_module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError('Torch must belong to active CONDA_PREFIX')
        solver=prefix/'lib/libcusolver.so.11'
        runtime=prefix/'lib'/runtime_name
        # Official Conda may expose only the canonical libcudart.so alias.
        if not runtime.is_file():runtime=prefix/'lib/libcudart.so'
        if not solver.is_file() or not runtime.is_file():raise RuntimeError('No matched official active Conda CUDA pair')
        version,solver_verified=_conda_owned(solver,prefix,('libcusolver','cusolver'))
        _,runtime_verified=_conda_owned(runtime,prefix,('cuda-cudart','libcudart'))
        verified=solver_verified&runtime_verified;origin='active official NVIDIA Conda pair'
    root=site if origin.startswith('installed') else prefix
    if not solver.resolve().is_relative_to(root) or not runtime.resolve().is_relative_to(root):
        raise RuntimeError('Official CUDA library links must remain within provider installation')
    solver=solver.resolve();runtime=runtime.resolve()
    sol_sha=file_sha256(solver);rt_sha=file_sha256(runtime)
    if expected_solver_sha256 is not None and sol_sha!=expected_solver_sha256:raise RuntimeError('Expected cuSOLVER SHA mismatch')
    if expected_runtime_sha256 is not None and rt_sha!=expected_runtime_sha256:raise RuntimeError('Expected CUDA runtime SHA mismatch')
    return InstalledLibraries(solver,runtime,origin,version,build,sol_sha,rt_sha,bool(verified))


def query_versions(libraries,*,loader=ctypes.CDLL):
    """Version ABI queries without explicit handle, tensor, or kernel calls.

    CUDA 11 runtime queries may initialize internal CUDA state. Resolution is
    lazy until an actual selected CUDA tensor call; no no-initialization claim
    is made for these queries.
    """
    integer=ctypes.c_int
    solver=loader(str(libraries.solver));runtime=loader(str(libraries.runtime))
    for lib,name in ((solver,'cusolverGetVersion'),(runtime,'cudaRuntimeGetVersion')):
        fn=getattr(lib,name);fn.argtypes=[ctypes.POINTER(integer)];fn.restype=integer
    a=integer();b=integer()
    if solver.cusolverGetVersion(ctypes.byref(a))!=0 or runtime.cudaRuntimeGetVersion(ctypes.byref(b))!=0:
        raise RuntimeError('Installed CUDA library version query failed')
    if b.value!=libraries.cuda_build:raise RuntimeError('Runtime CUDA major/minor must match Torch build in this solver')
    components=libraries.package_version.split('.')
    try:expected=int(components[0])*1000+int(components[1])*100+int(components[2])
    except (IndexError,ValueError):raise RuntimeError('Official cuSOLVER package version cannot be verified')
    if a.value!=expected or a.value<11000:raise RuntimeError('cuSOLVER ABI version differs from installed package metadata')
    if file_sha256(libraries.solver)!=libraries.solver_sha256 or file_sha256(libraries.runtime)!=libraries.runtime_sha256:
        raise RuntimeError('Installed CUDA library changed during version/load proof')
    return dict(cusolver_version=a.value,cuda_runtime_version=b.value,cuda_build_version=libraries.cuda_build,
                library_origin=libraries.origin,library_sha256=libraries.solver_sha256,
                runtime_library_sha256=libraries.runtime_sha256,package_hash_verified=libraries.package_hash_verified)
