"""转换输入与软件的完整 hash/stat 绑定，不包含固定数据标识。"""
import hashlib
import json
from pathlib import Path
from ..gds_packed import load_packed_reader

def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def source_stat(path):
    value = Path(path).stat()
    return dict(device=value.st_dev, inode=value.st_ino, size=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)

def make_source_binding(gds_path, *, packed_directory, input_files=None,
                        source_manifest=None):
    """返回源 GDS 身份、全部包源码、packed 构建文件及额外输入 SHA。

    input_files 为语义名称到文件路径的映射，例如 config/model/samples。
    source_manifest 可选，格式为包根目录相对 .py 路径到 SHA256 的 JSON。
    调用者必须在转换/分析前后重新调用，并拒绝任何差异。
    """
    package = Path(__file__).resolve().parent.parent
    sources = {str(path.relative_to(package)): file_sha(path)
               for path in sorted(package.rglob('*'))
               if path.is_file() and path.suffix in ('.py', '.cpp', '.h', '.hpp')}
    if source_manifest is not None:
        expected = json.loads(Path(source_manifest).read_text())
        if not isinstance(expected, dict) or any(sources.get(k) != v for k, v in expected.items()):
            raise ValueError('Source manifest does not match installed package')
    packed_directory = Path(packed_directory)
    # Includes SDK/header digests recorded by the build, its binding, extension
    # binary and loader. No external SDK is copied into the analysis cache.
    packed = {str(path.relative_to(packed_directory)): file_sha(path)
              for path in sorted(packed_directory.rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts}
    if not packed or 'packed_binding.json' not in packed:
        raise ValueError('Bound packed reader directory is required')
    # Re-run the existing SDK, official-source/header and compiled binary
    # validation each proof, not merely hash stale digests in a binding JSON.
    sdk_metadata = dict(load_packed_reader(packed_directory)._packed_metadata)
    inputs = {str(key): file_sha(path) for key, path in sorted((input_files or {}).items())}
    if source_manifest is not None:
        inputs['source_manifest'] = file_sha(source_manifest)
    return dict(gds_stat=source_stat(gds_path), source_files_sha256=sources,
                packed_files_sha256=packed, packed_sdk_binding=sdk_metadata,
                input_files_sha256=inputs)
