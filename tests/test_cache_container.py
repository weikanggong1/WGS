"""CPU-only contract tests; generated states validate mechanics, not benchmark."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from staar_phewas.cache_runtime import store, fast_container
from staar_phewas.cache_runtime import sparse_codec_fast as csr

def exact(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            assert left[key].dtype == right[key].dtype
            assert np.array_equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key

def rejected(call):
    try:
        call()
    except (ValueError, IndexError, OSError):
        return
    raise AssertionError('Invalid input accepted')

def run():
    source_hash = store.file_sha(store.__file__)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        cache = root / 'complete'
        samples = np.arange(31, dtype=np.int64)
        states = np.random.default_rng(44).integers(0, 6, (2067, len(samples)), dtype=np.uint8)
        states[0] = 0
        states[1] = 3
        def append(writer, block):
            arrays = csr.compact(block)
            writer.append(block, csr.integer_counts(*arrays, *block.shape))
        writer = store.Writer(cache, {'test': 'six-state'}, samples, len(states), source_bytes=100_000_000)
        append(writer, states[:1024])
        # Recovery must discard uncommitted tails and retain the committed frame.
        with (cache / 'data.bin').open('ab') as tail:
            tail.write(b'uncommitted')
        writer = store.Writer(cache, {'test': 'six-state'}, samples, len(states), source_bytes=100_000_000)
        assert writer.next_start == 1024
        append(writer, states[1024:2048])
        append(writer, states[2048:])
        writer.finish()
        original = store.Container(cache)
        with fast_container.Container(cache, {'test': 'six-state'}, samples) as fast:
            assert fast.verify_streams()
            for frame in range(len(fast.index)):
                exact(original.read_frame(frame), fast.read_frame(frame))
            for bad in (-1, 3, True, 1.0):
                rejected(lambda bad=bad: fast.read_frame(bad))
            fd_numbers = [stream.handle.fileno() for stream in fast._streams.values()]
        rejected(lambda: fast.read_frame(0))
        for fd in fd_numbers:
            rejected(lambda fd=fd: os.fstat(fd))
        fast.close()
        with fast_container.Container(cache) as a, fast_container.Container(cache) as b:
            assert all(a._streams[n].handle.fileno() != b._streams[n].handle.fileno() for n in a._streams)
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                jobs = [pool.submit(reader.read_frame, frame) for reader in (a, b) for frame in (2, 0, 1, 0, 2)]
                for job, frame in zip(jobs, (2, 0, 1, 0, 2) * 2):
                    exact(original.read_frame(frame), job.result())
        for change in ('corrupt', 'truncate', 'replace', 'bounds'):
            mutated = root / change
            shutil.copytree(cache, mutated)
            fast = fast_container.Container(mutated)
            fast.read_frame(0)  # Warm reader: no stale buffered bytes permitted.
            data = mutated / 'data.bin'
            if change == 'corrupt':
                with data.open('r+b') as stream:
                    byte = stream.read(1)
                    stream.seek(0)
                    stream.write(bytes([byte[0] ^ 1]))
                rejected(lambda: fast.read_frame(0))
            elif change == 'truncate':
                with data.open('r+b') as stream:
                    stream.truncate(0)
                rejected(lambda: fast.read_frame(0))
            elif change == 'replace':
                replacement = mutated / 'replacement'
                shutil.copyfile(data, replacement)
                os.replace(replacement, data)
                # Owned old FD remains a consistent snapshot until final identity check.
                exact(original.read_frame(0), fast.read_frame(0))
            else:
                row = {k: fast.index[0][k] for k in store.INDEX.names}
                row['size'] = data.stat().st_size + 1
                rejected(lambda: store._read_frame(fast._frame_path, row, fast.manifest))
            if change != 'bounds':
                rejected(fast.check_stream_identity)
                rejected(fast.close)
            else:
                fast.close()
            assert all(stream.handle.closed for stream in fast._streams.values())
        rejected(lambda: fast_container.Container(cache, {'bad': True}))
        rejected(lambda: fast_container.Container(cache, expected_samples=samples[::-1]))
    assert store.file_sha(store.__file__) == source_hash
    return {'passed': True, 'generated_frames': 3, 'all_six_states': True, 'resumed_complete': True,
            'independent_threaded_readers': True, 'mutation_and_bounds_checks': True,
            'closed_fd_cleanup': True, 'original_store_sha256': source_hash,
            'fast_container_sha256': hashlib.sha256(Path(fast_container.__file__).read_bytes()).hexdigest()}

def test_generated_container_contract():
    assert run()["passed"]
