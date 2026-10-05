"""Bounded analysis concurrency using independent CUDA streams."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from numbers import Integral
import threading
import torch


@dataclass
class ExecutionConfig:
    parallel_level: str = 'serial'
    workers: int = 1
    max_gpu_gb: float = 20.

    def __post_init__(self):
        if self.parallel_level not in ('serial', 'mask', 'chromosome'):
            raise ValueError('parallel_level must be serial, mask or chromosome')
        if isinstance(self.workers,bool) or not isinstance(self.workers,Integral) or self.workers < 1:
            raise ValueError('Worker count must be a positive integer')
        try:
            valid_budget=(not isinstance(self.max_gpu_gb,bool)
                          and math.isfinite(self.max_gpu_gb) and self.max_gpu_gb > 0)
        except TypeError:
            valid_budget=False
        if not valid_budget:
            raise ValueError('GPU memory budget must be finite and positive')

    @property
    def concurrency(self):
        return 1 if self.parallel_level == 'serial' else self.workers


class GpuExecutor:
    """Run independent files on separate streams in one CUDA context.

    The caller keeps input tensors alive and does not change TF32 flags while
    workers execute. Each worker synchronizes its stream before reporting a
    completed output. Results retain submission order regardless of completion.
    """
    def __init__(self, configuration, device='cuda'):
        self.configuration = configuration
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.local = threading.local()
        self.pool = None
        self.parent_stream = None

    def __enter__(self):
        if self.device.type == 'cuda':
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1., self.configuration.max_gpu_gb * 1024**3 / total), self.device)
            self.parent_stream = torch.cuda.current_stream(self.device)
        if self.configuration.concurrency > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.configuration.concurrency)
        return self

    def _run(self, function, item):
        if self.device.type != 'cuda' or self.pool is None:
            return function(item)
        if not hasattr(self.local, 'stream'):
            self.local.stream = torch.cuda.Stream(device=self.device)
        stream = self.local.stream
        stream.wait_stream(self.parent_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            try:
                result = function(item)
            finally:
                stream.synchronize()
        return result

    def map(self, function, items):
        if self.pool is None:
            return [self._run(function, item) for item in items]
        futures = [self.pool.submit(self._run, function, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    def __exit__(self, *exception):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=exception[0] is not None)
