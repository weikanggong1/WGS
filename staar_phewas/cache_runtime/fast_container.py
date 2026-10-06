"""Private immutable-container reader with three instance-owned stream handles.

All initialization and per-frame scientific validations run in unchanged store.
Unbuffered handles deliberately observe mutations even following repeated reads.
Stream identity checks happen at open/close, not per frame; callers must preserve
source-wide pre/post checks. No writes, global patching, GPU or shared FD state.
"""
import os
import threading
from . import store

_NAMES = ('data.bin', 'headers.bin', 'counts.bin')

def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

class _BorrowedHandle:
    def __init__(self, handle):
        self.handle = handle
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def seek(self, offset):
        self.offset = offset
        return offset
    def read(self, size):
        return os.pread(self.handle.fileno(), size, self.offset)

class _StreamPath:
    def __init__(self, path, handle, initial_stat):
        self.path, self.handle, self.initial_stat = path, handle, initial_stat
    def stat(self):
        return self.initial_stat
    def open(self, mode):
        if mode != 'rb':
            raise ValueError('Private stream proxy is read-only')
        return _BorrowedHandle(self.handle)

class _FramePath:
    def __init__(self, streams):
        self.streams = streams
    def __truediv__(self, name):
        return self.streams[name]

class Container(store.Container):
    def __init__(self, path, expected_source_binding=None, expected_samples=None):
        super().__init__(path, expected_source_binding, expected_samples)
        self._lock = threading.RLock()
        self._closed = False
        self._streams = {}
        try:
            for name in _NAMES:
                stream_path = self.path / name
                initial = stream_path.stat()
                offset_key, size_key = {
                    'data.bin': ('offset', 'size'),
                    'headers.bin': ('header_offset', 'header_size'),
                    'counts.bin': ('counts_offset', 'counts_size'),
                }[name]
                expected_size = (int(self.index[-1][offset_key]) + int(self.index[-1][size_key])
                                 if len(self.index) else 0)
                if initial.st_size != expected_size:
                    raise ValueError('Stream length changed while opening')
                handle = stream_path.open('rb', buffering=0)
                self._streams[name] = _StreamPath(stream_path, handle, initial)
                if _identity(os.fstat(handle.fileno())) != _identity(initial):
                    raise ValueError('Stream identity changed while opening')
            self._frame_path = _FramePath(self._streams)
        except BaseException:
            for stream in self._streams.values():
                stream.handle.close()
            self._closed = True
            raise

    def read_frame(self, frame_id):
        if type(frame_id) is not int or not 0 <= frame_id < len(self.index):
            raise IndexError('Frame ID out of bounds')
        with self._lock:
            if self._closed:
                raise ValueError('Container is closed')
            return store._read_frame(self._frame_path,
                {key: self.index[frame_id][key] for key in store.INDEX.names}, self.manifest)

    def check_stream_identity(self):
        with self._lock:
            if self._closed:
                raise ValueError('Container is closed')
            for stream in self._streams.values():
                expected = _identity(stream.initial_stat)
                if (_identity(os.fstat(stream.handle.fileno())) != expected or
                        _identity(stream.path.stat()) != expected):
                    raise ValueError('Immutable stream changed after opening')
        return True

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self.check_stream_identity()
            finally:
                for stream in self._streams.values():
                    stream.handle.close()
                self._closed = True

    def __enter__(self):
        if self._closed:
            raise ValueError('Container is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        # Preserve an existing validation failure while still closing every FD.
        try:
            self.close()
        except (OSError, ValueError):
            if exc_type is None:
                raise
        return False

    def __del__(self):
        # Best effort only; explicit close/context exit supplies identity proof.
        for stream in getattr(self, '_streams', {}).values():
            stream.handle.close()

open_container = Container
