"""Optional continuous Bit2 I/O using the official installed PyGDS capsule SDK.

SPDX-License-Identifier: GPL-3.0-only
Build explicitly with ``python -m staar_phewas.gds_flat --output-dir DIR``.
Add DIR to PYTHONPATH before launching an analysis. This module never builds
or installs anything during a data read and does not modify an environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path


def load_flat_reader():
    """Return the installed optional module, or None when it is absent.

    An installed but incompatible SDK or broken module must raise its import
    error. Read, allocation and file-lifetime errors are never suppressed.
    """
    try:
        return importlib.import_module("staar_gds_flat")
    except ModuleNotFoundError as error:
        if error.name != "staar_gds_flat":
            raise
        return None


def flat_reader_metadata(module=None):
    """Report the actual reader and, when installed, its binary SHA-256."""
    if module is None:
        return {"reader_backend": "pygds_generic", "native_binary_sha256": None}
    binary = Path(module.__file__)
    return {"reader_backend": "pygds_flat_sdk",
            "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def build_flat_reader(output_dir):
    """Compile only this adapter into output_dir using installed SDK headers.

    Requires setuptools, Python development headers and a C++ compiler.
    The official pygds package must already be installed; this function does
    not download a similarly named package or install into site-packages.
    """
    from setuptools import Distribution, Extension
    import pygds

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("_gds_flat.cpp")
    if not source.is_file():
        raise FileNotFoundError("the package does not contain _gds_flat.cpp")
    include = Path(pygds.get_include())
    headers = [include / name for name in ("PyGDS.h", "PyGDS2.h", "dType.h", "CoreDEF.h")]
    if not all(path.is_file() for path in headers):
        raise FileNotFoundError("the installed official PyGDS SDK headers are incomplete")
    extension = Extension(
        "staar_gds_flat", [str(source)], include_dirs=[str(include)],
        depends=[str(path) for path in headers], language="c++",
        extra_compile_args=["/O2"] if os.name == "nt" else ["-O3", "-std=c++11"],
    )
    distribution = Distribution({"name": "staar-gds-flat", "ext_modules": [extension]})
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(destination)
    command.build_temp = str(destination / "build")
    command.inplace = False
    command.force = True
    command.ensure_finalized()
    command.run()
    binary = Path(command.get_ext_fullpath("staar_gds_flat"))
    if not binary.is_file():
        raise RuntimeError("the compiler did not produce the optional GDS module")
    return {"reader_backend": "pygds_flat_sdk", "native_binary": str(binary),
            "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "adapter_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sdk_headers_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in headers},
            "pygds_version": pygds.__version__}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build optional flat Bit2 I/O against the installed official PyGDS SDK")
    parser.add_argument("--output-dir", required=True, help="directory for the compiled module and its build objects")
    args = parser.parse_args(argv)
    print(json.dumps(build_flat_reader(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
