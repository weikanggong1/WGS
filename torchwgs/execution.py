"""GPU-only serial execution with a bounded CUDA allocator."""
from dataclasses import dataclass
import math
from numbers import Integral
import torch


@dataclass
class ExecutionConfig:
    parallel_level: str = 'serial'
    workers: int = 1
    max_gpu_gb: float = 20.

    def __post_init__(self):
        if self.parallel_level != 'serial':
            raise ValueError('Only GPU serial execution is supported (parallel_level="serial")')
        if isinstance(self.workers,bool) or not isinstance(self.workers,Integral) or self.workers != 1:
            raise ValueError('GPU serial execution requires workers=1')
        try:
            valid_budget=(not isinstance(self.max_gpu_gb,bool)
                          and math.isfinite(self.max_gpu_gb) and self.max_gpu_gb > 0)
        except TypeError:
            valid_budget=False
        if not valid_budget:
            raise ValueError('GPU memory budget must be finite and positive')

    @property
    def concurrency(self):
        return 1


class GpuExecutor:
    """Run files in input order on the requested CUDA device."""
    def __init__(self, configuration, device='cuda'):
        self.configuration = configuration
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Discovery association requires a CUDA GPU')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; CPU association fallback is disabled')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())

    def __enter__(self):
        total = torch.cuda.get_device_properties(self.device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1., self.configuration.max_gpu_gb * 1024**3 / total), self.device)
        return self

    def _run(self, function, item):
        with torch.cuda.device(self.device):
            return function(item)

    def map(self, function, items):
        return [self._run(function, item) for item in items]

    def __exit__(self, *exception):
        return False
