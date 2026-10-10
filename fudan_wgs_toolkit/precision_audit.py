"""Reject hidden dense FP64 products in forced TF32 pipeline executions.

The Torch dispatcher observes explicit ATen products (including products from
einsum). Solver-internal eigendecompositions are not GEMM fallback routes and
are reported separately. Triton products are verified by tf32.py's PTX checks.
"""
import torch
from contextlib import contextmanager
from contextvars import ContextVar

_spectral_refinement = ContextVar("explicit_fp64_spectral_refinement", default=False)

@contextmanager
def explicit_fp64_spectral_refinement():
    """Declare and count only the long-mask spectral FP64 calculation."""
    token = _spectral_refinement.set(True)
    try:
        yield
    finally:
        _spectral_refinement.reset(token)

from torch.utils._python_dispatch import TorchDispatchMode


class DenseProductAudit(TorchDispatchMode):
    _products = {"aten.mm.default", "aten.mv.default", "aten.bmm.default",
                 "aten.addmm.default", "aten.addmv.default", "aten.baddbmm.default",
                 "aten.dot.default", "aten._scaled_mm.default"}

    def __init__(self, *, forced):
        super().__init__()
        self.forced = forced
        self.fp64_products = 0
        self.explicit_spectral_fp64_products = 0
        self.observed_products = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in self._products:
            self.observed_products += 1
            if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64 for value in args):
                if _spectral_refinement.get():
                    self.explicit_spectral_fp64_products += 1
                else:
                    self.fp64_products += 1
                if self.forced and not _spectral_refinement.get():
                    raise RuntimeError(f"Forced TF32 execution rejected hidden FP64 product: {func}")
        return func(*args, **(kwargs or {}))

    def report(self):
        return {"enabled": self.forced, "observed_aten_dense_products": self.observed_products,
                "hidden_fp64_dense_products": self.fp64_products,
                "explicit_fastskat_spectral_fp64_products": self.explicit_spectral_fp64_products,
                "scope": "explicit ATen dense mm/mv/bmm/addmm/addmv/baddbmm/dot; solver internals excluded"}
