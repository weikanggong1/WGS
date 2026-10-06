"""Explicit optional packed Bit2 reader bound to one existing SDK source build.

SPDX-License-Identifier: GPL-3.0-only
No build, download or installation occurs while opening or reading data.
CoreArray internal C++ allocator layout is not a stable public capsule API.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
import sysconfig


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _platform_binding():
    return dict(soabi=sysconfig.get_config_var('SOABI'),
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),
                machine=platform.machine(),system=platform.system(),byteorder=sys.byteorder,
                pointer_bytes=ctypes.sizeof(ctypes.c_void_p),position_bytes=8)


def _headers(source):
    source=Path(source)
    files=sorted([*(source/'pygds/include').glob('*.h'),*(source/'src/CoreArray').glob('*.h')])
    if not (source/'src/CoreArray/CoreArray.h').is_file() or not files:
        raise FileNotFoundError('Corresponding official SDK source headers are required')
    return {str(p.relative_to(source)):_sha(p) for p in files}


def _source_files(source):
    source=Path(source)
    files=set()
    for base in ('pygds','src'):
        files.update(p for p in (source/base).rglob('*')
                     if p.is_file() and p.suffix in ('.h','.hpp','.c','.cpp','.cxx','.py'))
    if (source/'setup.py').is_file():files.add(source/'setup.py')
    return {str(p.relative_to(source)):_sha(p) for p in sorted(files)}


def _header_digest(headers):
    return hashlib.sha256(json.dumps(headers,sort_keys=True).encode()).hexdigest()


def _import_binary(binary):
    spec=importlib.util.spec_from_file_location('staar_gds_packed',binary)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the explicitly configured packed adapter')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=Path(binary).resolve():
        raise ImportError('Packed adapter was loaded from an unexpected path')
    return module


def _validate_binding(binding,directory):
    if binding.get('schema')!=1 or binding['platform']!=_platform_binding():
        raise RuntimeError('Packed adapter platform/SOABI binding mismatch')
    if (binding['platform']['system']!='Linux' or binding['platform']['pointer_bytes']!=8
            or binding['platform']['byteorder']!='little'):
        raise RuntimeError('Packed adapter requires the validated 64-bit little-endian platform')
    filename=binding['extension_filename']
    if Path(filename).name!=filename or filename!='staar_gds_packed'+binding['platform']['extension_suffix']:
        raise RuntimeError('Invalid packed adapter filename')
    binary=Path(directory)/filename
    if _sha(binary)!=binding['extension_sha256']:
        raise RuntimeError('Packed adapter binary hash mismatch')
    sdk=importlib.import_module('pygds.ccall')
    if _sha(sdk.__file__)!=binding['sdk_binary_sha256'] or _sha(binding['source_built_sdk'])!=binding['sdk_binary_sha256']:
        raise RuntimeError('Installed SDK differs from the corresponding source-build binary')
    if _source_files(binding['source_directory'])!=binding['source_files']:
        raise RuntimeError('Packed adapter corresponding source binding mismatch')
    actual=_headers(binding['source_directory'])
    if actual!=binding['headers'] or _header_digest(actual)!=binding['official_headers_sha256']:
        raise RuntimeError('Packed adapter official header binding mismatch')
    import pygds
    for name in ('PyGDS.h','PyGDS2.h','dType.h','CoreDEF.h'):
        if _sha(Path(pygds.get_include())/name)!=actual['pygds/include/'+name]:
            raise RuntimeError('Installed public SDK headers differ from source build')
    if _sha(Path(__file__).with_name('_gds_packed.cpp'))!=binding['adapter_source_sha256']:
        raise RuntimeError('Packed adapter source binding mismatch')
    return binary


def load_packed_reader(directory=None):
    """None retains the existing reader; any explicit invalid config raises.

    directory contains a locally built extension and packed_binding.json.
    Its private source/build paths must remain available for validation.
    Read failures are never suppressed or redirected to another backend.
    """
    if directory is None:
        return None
    directory=Path(directory).expanduser().resolve()
    binding=json.loads((directory/'packed_binding.json').read_text())
    binary=_validate_binding(binding,directory)
    module=_import_binary(binary)
    if module.binding()!=dict(sdk_binary_sha256=binding['sdk_binary_sha256'],
                              official_headers_sha256=binding['official_headers_sha256'],layout=binding['layout']):
        raise RuntimeError('Packed adapter compiled layout binding mismatch')
    if binding['layout']['pointer']!=8 or binding['layout']['position']!=8:
        raise RuntimeError('Unsupported packed adapter layout')
    module._packed_metadata=dict(reader_backend='pygds_packed_pinned',
        packed_native_binary_sha256=binding['extension_sha256'],
        packed_sdk_binary_sha256=binding['sdk_binary_sha256'],
        packed_headers_sha256=binding['official_headers_sha256'],
        packed_binding_verified=True,packed_layout=dict(binding['layout']))
    return module


def build_packed_reader(*,source_dir,source_built_sdk,output_dir):
    """Build only this adapter against an existing matching official build.

    source_dir is the original PyGDS source tree with CoreArray headers;
    source_built_sdk is its existing compiled pygds.ccall library; output_dir
    is an empty local build directory. No upstream files are copied.
    Requires an existing C++ compiler, Python development headers and the
    same CoreArray build dependencies. This is a pinned optional interface.
    """
    import pygds
    sdk=importlib.import_module('pygds.ccall')
    source=Path(source_dir).expanduser().resolve();built=Path(source_built_sdk).expanduser().resolve();destination=Path(output_dir).expanduser().resolve()
    if _sha(built)!=_sha(sdk.__file__):
        raise RuntimeError('Source-build and installed SDK binaries differ')
    headers=_headers(source);source_files=_source_files(source)
    for name in ('PyGDS.h','PyGDS2.h','dType.h','CoreDEF.h'):
        if _sha(Path(pygds.get_include())/name)!=headers['pygds/include/'+name]:
            raise RuntimeError('Installed SDK header mismatch')
    identity=_platform_binding()
    if identity['system']!='Linux' or identity['pointer_bytes']!=8 or identity['byteorder']!='little':
        raise RuntimeError('Only the validated 64-bit little-endian Linux build is supported')
    destination.mkdir(parents=True,exist_ok=False)
    digest=_header_digest(headers);sdksha=_sha(built);cpp=Path(__file__).with_name('_gds_packed.cpp')
    (destination/'packed_binding.hpp').write_text('#define BOUND_BINARY_SHA256 "'+sdksha+'"\n#define BOUND_HEADERS_SHA256 "'+digest+'"\n')
    binary=destination/('staar_gds_packed'+identity['extension_suffix'])
    include=Path(sysconfig.get_paths()['include'])
    flags=['-std=c++11','-O3','-shared','-fPIC','-DUSING_PYTHON','-D_FILE_OFFSET_BITS=64','-DCOREARRAY_USE_LZMA_EXT']
    command=['c++',*flags,'-I'+str(include),'-I'+str(include.parent),'-I'+str(source/'pygds/include'),'-I'+str(source/'src/CoreArray'),'-I'+str(destination),str(cpp),'-o',str(binary)]
    subprocess.run(command,check=True)
    if (_headers(source)!=headers or _source_files(source)!=source_files
            or _sha(built)!=sdksha or _sha(sdk.__file__)!=sdksha):
        raise RuntimeError('SDK source/build changed during compilation')
    module=_import_binary(binary);compiled=module.binding()
    if compiled['sdk_binary_sha256']!=sdksha or compiled['official_headers_sha256']!=digest:
        raise RuntimeError('Compiled SDK identity mismatch')
    binding=dict(schema=1,platform=identity,extension_filename=binary.name,extension_sha256=_sha(binary),source_directory=str(source),source_built_sdk=str(built),sdk_binary_sha256=sdksha,source_files=source_files,headers=headers,official_headers_sha256=digest,layout=compiled['layout'],adapter_source_sha256=_sha(cpp),compiler_flags=flags)
    (destination/'packed_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    return load_packed_reader(destination)._packed_metadata


def main(argv=None):
    p=argparse.ArgumentParser(description='Build optional pinned packed Bit2 adapter; no downloads or installation')
    p.add_argument('--source-dir',required=True);p.add_argument('--source-built-sdk',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args(argv);print(json.dumps(build_packed_reader(source_dir=a.source_dir,source_built_sdk=a.source_built_sdk,output_dir=a.output_dir),sort_keys=True))


if __name__=='__main__':main()
