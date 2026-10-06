"""Anonymous stage timing and evidence-bounded runtime estimates.

This module uses only the Python standard library.  CUDA synchronization imports
PyTorch lazily, only for a CUDA span without an injected synchronization function.
Names must be anonymous category labels, never sample, gene, variant, or trait IDs.
Observed ranges describe measured rates; they are not confidence intervals.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import math
from numbers import Integral, Real
import random
import re
import time


_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def _label(value):
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise ValueError("Use an anonymous category label of at most 64 characters")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + " must be an integer within its allowed range")
    return int(value)


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite number within its allowed range")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(name + " must be a finite number within its allowed range")
    return result


def _sum(values):
    try:
        result = math.fsum(values)
    except OverflowError:
        raise ValueError("The aggregate exceeds the finite numeric range") from None
    if not math.isfinite(result):
        raise ValueError("The aggregate exceeds the finite numeric range")
    return result


def _finite_product(left, right):
    try:
        result = left * right
    except OverflowError:
        raise ValueError("The estimate exceeds the finite numeric range") from None
    if not math.isfinite(result):
        raise ValueError("The estimate exceeds the finite numeric range")
    return result


def _record(value, required):
    if not isinstance(value, Mapping) or not all(key in value for key in required):
        raise ValueError("An observation is missing required fields")
    return value


class StageTimers:
    """Measure nested spans, separating inclusive wall time from self time.

    ``synchronize`` is a zero-argument callback.  For ``cuda=True``, it is called
    before and after the body, including a body that fails.  These waits are part
    of the measured wall time.  A pre-existing body exception is preserved if the
    final synchronization also fails.  CPU spans never import or access CUDA.

    An instance is intended for one serial worker.  Each active invocation has
    its own frame, so repeated or nested uses of the same name remain correct.
    ``summary()`` returns detached primitive values, including failed spans.
    """

    def __init__(self, device=None, synchronize=None, clock=time.perf_counter):
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be a zero-argument callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.device = device
        self._synchronize_callback = synchronize
        self._clock = clock
        self._frames = []
        self._stages = {}

    def _synchronize(self):
        if self._synchronize_callback is not None:
            self._synchronize_callback()
        else:
            import torch

            torch.cuda.synchronize(self.device)

    @contextmanager
    def span(self, name, cuda=False):
        """Time a stage; nested children are subtracted from exclusive time."""
        name = _label(name)
        if not isinstance(cuda, bool):
            raise TypeError("cuda must be a boolean")
        record = self._stages.setdefault(name, {
            "calls": 0,
            "completed_calls": 0,
            "failed_calls": 0,
            "seconds_inclusive": 0.0,
            "seconds_exclusive": 0.0,
            "first_call_seconds": None,
            "maximum_call_seconds": 0.0,
            "cuda_calls": 0,
        })
        frame = {"start": self._clock(), "children": 0.0,
                 "first": record["calls"] == 0}
        record["calls"] += 1
        record["cuda_calls"] += int(cuda)
        self._frames.append(frame)
        failed = False
        synchronized_before = False
        try:
            if cuda:
                self._synchronize()
                synchronized_before = True
            yield
        except BaseException:
            failed = True
            raise
        finally:
            try:
                if cuda and synchronized_before:
                    try:
                        self._synchronize()
                    except BaseException:
                        if not failed:
                            failed = True
                            raise
            finally:
                elapsed = max(0.0, self._clock() - frame["start"])
                self._frames.pop()
                exclusive = max(0.0, elapsed - frame["children"])
                record["failed_calls" if failed else "completed_calls"] += 1
                record["seconds_inclusive"] += elapsed
                record["seconds_exclusive"] += exclusive
                record["maximum_call_seconds"] = max(
                    record["maximum_call_seconds"], elapsed)
                if frame["first"]:
                    record["first_call_seconds"] = elapsed
                if self._frames:
                    self._frames[-1]["children"] += elapsed

    def summary(self):
        """Return stage labels and numeric counters, without device or errors."""
        return {name: dict(record) for name, record in self._stages.items()}


def plan_windows(total_sites, block_size, sample_blocks, seed=0):
    """Select spread-out, nonoverlapping site-ordinal windows ``[start, stop)``.

    Block ordinals are partitioned into equally sized contiguous strata, and one
    block is selected from each using a local seeded generator.  At most one
    window is returned per existing block; the last block can be shorter.
    Requesting at least the number of blocks returns the entire source once.
    """
    total_sites = _integer(total_sites, "total_sites")
    block_size = _integer(block_size, "block_size", minimum=1)
    sample_blocks = _integer(sample_blocks, "sample_blocks")
    seed = _integer(seed, "seed", minimum=-(2 ** 63))
    blocks = (total_sites + block_size - 1) // block_size
    count = min(blocks, sample_blocks)
    if count == 0:
        return []
    rng = random.Random(seed)
    selected = [rng.randrange(index * blocks // count,
                              (index + 1) * blocks // count)
                for index in range(count)]
    return [(index * block_size, min(total_sites, (index + 1) * block_size))
            for index in selected]


def estimate_linear(records, total_units):
    """Extrapolate a weighted measured rate, with a descriptive observed range.

    Records contain positive ``units`` and nonnegative ``seconds``.  The weighted
    rate is sum(seconds)/sum(units), rather than the unweighted mean of rates.
    Different workload distributions or cache states must be estimated in
    separate calls; this helper does not assume that their rates are equivalent.
    """
    total_units = _number(total_units, "total_units")
    units, durations, rates = [], [], []
    for observation in records:
        observation = _record(observation, ("units", "seconds"))
        if "complete" in observation and observation["complete"] is not True:
            raise ValueError("Linear estimation requires completed observations")
        size = _number(observation["units"], "units", positive=True)
        seconds = _number(observation["seconds"], "seconds")
        rate = seconds / size
        if not math.isfinite(rate):
            raise ValueError("An observed rate exceeds the finite numeric range")
        units.append(size)
        durations.append(seconds)
        rates.append(rate)
    measured_units, measured_seconds = _sum(units), _sum(durations)
    weighted_rate = measured_seconds / measured_units if units else None
    if weighted_rate is not None and not math.isfinite(weighted_rate):
        raise ValueError("The aggregate rate exceeds the finite numeric range")
    total_seconds = _finite_product(total_units, weighted_rate) if rates else None
    bounds = ({"min": _finite_product(total_units, min(rates)),
               "max": _finite_product(total_units, max(rates))}
              if rates else None)
    return {
        "total_units": total_units,
        "observations": len(rates),
        "measured_units": measured_units,
        "measured_seconds": measured_seconds,
        "seconds_per_unit": weighted_rate,
        "total_seconds": total_seconds,
        "range_seconds": bounds,
        "range_kind": "observed_rates_not_confidence_interval",
    }


def estimate_gene_strata(census, measurements):
    """Estimate separate anonymous workload strata without discarding timeouts.

    Each measurement must describe one distinct gene attempt in its stratum;
    repeated cold/warm measurements belong in separate estimates.  A stratum
    must distinguish analysis group as well as workload size.  An unsampled
    positive-count stratum, or any censored observation, prevents a whole-run
    estimate.  A timeout contributes its elapsed time to the strict measured
    lower bound, never to a completed-duration mean.

    The observed range uses the measured completed time plus the minimum or
    maximum completed duration for each unmeasured gene.  Thus a fully measured
    census has an exact total, and neither end is below already observed time.
    """
    groups = {}
    for item in census:
        item = _record(item, ("stratum", "count"))
        name = _label(item["stratum"])
        count = _integer(item["count"], "count")
        if name in groups:
            raise ValueError("Census strata must be unique")
        groups[name] = {"count": count, "complete": [], "censored": []}
    for item in measurements:
        item = _record(item, ("stratum", "seconds", "complete"))
        name = _label(item["stratum"])
        if name not in groups:
            raise ValueError("A measurement has no matching census stratum")
        if not isinstance(item["complete"], bool):
            raise ValueError("complete must be a boolean")
        seconds = _number(item["seconds"], "seconds")
        group = groups[name]
        group["complete" if item["complete"] else "censored"].append(seconds)
        if len(group["complete"]) + len(group["censored"]) > group["count"]:
            raise ValueError("Distinct measured genes exceed the census count")
    strata, missing, censored_strata = [], [], []
    for name, group in groups.items():
        completed, censored = group["complete"], group["censored"]
        count = group["count"]
        complete_seconds = _sum(completed)
        observed_seconds = _sum(completed + censored)
        mean = complete_seconds / len(completed) if completed else None
        ready = count == 0 or (bool(completed) and not censored)
        if count > 0 and not completed:
            missing.append(name)
        if censored:
            censored_strata.append(name)
        if ready:
            remaining = count - len(completed)
            point = _sum((complete_seconds, _finite_product(remaining, mean or 0.0)))
            bounds = {
                "min": _sum((complete_seconds, _finite_product(
                    remaining, min(completed) if completed else 0.0))),
                "max": _sum((complete_seconds, _finite_product(
                    remaining, max(completed) if completed else 0.0))),
            }
        else:
            point, bounds = None, None
        strata.append({
            "stratum": name,
            "count": count,
            "completed": len(completed),
            "censored": len(censored),
            "observed_seconds": observed_seconds,
            "mean_complete_seconds": mean,
            "total_seconds": point,
            "range_seconds": bounds,
            "lower_bound_seconds": observed_seconds,
        })
    ready = not missing and not censored_strata
    return {
        "total_seconds": _sum(item["total_seconds"] for item in strata) if ready else None,
        "range_seconds": {
            "min": _sum(item["range_seconds"]["min"] for item in strata),
            "max": _sum(item["range_seconds"]["max"] for item in strata),
        } if ready else None,
        "range_kind": "observed_complete_durations_not_confidence_interval",
        "lower_bound_seconds": _sum(item["lower_bound_seconds"] for item in strata),
        "missing_strata": missing,
        "censored_strata": censored_strata,
        "strata": strata,
    }


def _suggestion(name):
    name = name.lower()
    tokens = set(re.split(r"[_-]", name))
    categories = (
        (("jit", "compile"), "Measure cold compilation separately from warm execution; inspect shape specialization without changing numerical precision."),
        (("h2d", "transfer"), "Inspect bounded packed transfers and reusable sample mappings; compare transfer time with storage time."),
        (("decode", "filter", "mac"), "Inspect packed counting and retained-only decoding; retain exact allele counts and filtering thresholds."),
        (("bim", "read", "index", "metadata", "io"), "Inspect input reuse and bounded metadata caches; include first preparation and distinguish storage-cache conditions."),
        (("eigh", "eigen"), "Inspect eigensolver dimensions and reusable covariance state; preserve the statistical kernel and verify numerical output."),
        (("quadrature", "integrat", "davies", "qags", "skato", "tail"), "Inspect quadrature nodes and controller cost separately from tail kernels; preserve accuracy thresholds and stopping rules."),
        (("sbat", "nnls", "orthant", "weight"), "Inspect active-set and orthant work separately; preserve column selection, weighting, and random-seed protocol."),
        (("score", "product", "covariance", "gram", "projection"), "Inspect matrix dimensions and valid product reuse; use bounded workspaces and verify all masks and output values."),
        (("write", "output", "pack", "d2h", "format"), "Inspect bounded packing and result transfers; preserve missing values, rounding, attachment order, and file bytes."),
        (("step1", "ridge"), "Separate input, matrix construction, solves, and prediction writes; keep the fitted model and paper parameters unchanged."),
    )
    for fragments, suggestion in categories:
        if any((fragment in tokens if fragment == "io" else fragment in name)
               for fragment in fragments):
            return suggestion
    return "Subdivide this measured stage before choosing an optimization; preserve output and statistical parameters."


def optimization_summary(stages):
    """Rank exclusive measured costs, without claiming an unmeasured speedup."""
    if not isinstance(stages, Mapping):
        raise ValueError("stages must be an anonymous stage summary")
    ranked = []
    for name, stage in stages.items():
        name = _label(name)
        stage = _record(stage, ("seconds_exclusive",))
        seconds = _number(stage["seconds_exclusive"], "seconds_exclusive")
        calls = _integer(stage.get("calls", 0), "calls")
        completed = _integer(stage.get("completed_calls", 0), "completed_calls")
        failed = _integer(stage.get("failed_calls", 0), "failed_calls")
        if completed + failed > calls:
            raise ValueError("Stage completion counters exceed the call count")
        ranked.append({"stage": name, "seconds_exclusive": seconds,
                       "calls": calls, "completed_calls": completed,
                       "failed_calls": failed})
    ranked.sort(key=lambda item: (-item["seconds_exclusive"], item["stage"]))
    measured = _sum(item["seconds_exclusive"] for item in ranked)
    recommendations = []
    for item in ranked:
        item["fraction_of_measured_exclusive_time"] = (
            item["seconds_exclusive"] / measured if measured else 0.0)
        if item["seconds_exclusive"] > 0:
            recommendations.append({"stage": item["stage"],
                                    "seconds_exclusive": item["seconds_exclusive"],
                                    "suggestion": _suggestion(item["stage"]),
                                    "contains_failed_calls": item["failed_calls"] > 0})
    return {"ranked_stages": ranked,
            "measured_exclusive_seconds": measured,
            "recommendations": recommendations,
            "basis": "measured_exclusive_time",
            "speedup_claimed": False}
