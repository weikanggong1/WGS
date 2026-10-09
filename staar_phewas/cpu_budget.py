"""Detect CPU constraints and share a bounded preparation budget across workers.

This module uses only the standard library. Detection reports contain counts and
quota values rather than machine names, cgroup paths, or process identifiers.
"""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path


_UNSET = object()


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mount_field(value):
    return re.sub(r"\\(040|011|012|134)",
                  lambda match: chr(int(match.group(1), 8)), value)


def _read_text(path):
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError):
        return None, "unreadable"


def _memberships(raw):
    result = {"v1": [], "v2": []}
    for line in (raw or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = parts[1].split(",")
        if parts[0] == "0" and parts[1] == "":
            result["v2"].append(parts[2])
        elif "cpu" in controllers:
            result["v1"].append(parts[2])
    return result


def _mounts(raw):
    result = []
    for line in (raw or "").splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) < 3:
            continue
        filesystem = right[0]
        if filesystem == "cgroup2":
            version = "v2"
        elif filesystem == "cgroup" and "cpu" in set(
                ",".join([right[1], right[2], left[5]]).split(",")):
            version = "v1"
        else:
            continue
        result.append((version, _mount_field(left[3]),
                       Path(_mount_field(left[4]))))
    return result


def _leaf(mount_root, mount_point, membership):
    group = posixpath.normpath("/" + membership.lstrip("/"))
    root = posixpath.normpath("/" + mount_root.lstrip("/"))
    if root == "/":
        relative = group.lstrip("/")
    elif group == root:
        relative = ""
    elif group.startswith(root + "/"):
        relative = group[len(root) + 1:]
    else:
        # A cgroup namespace may expose paths relative to a non-root mount.
        relative = group.lstrip("/")
    return mount_point / relative


def _ancestors(leaf, mount_point):
    current, depth = leaf, 0
    while True:
        yield current, depth
        if current == mount_point:
            break
        parent = current.parent
        if parent == current or (mount_point not in parent.parents and parent != mount_point):
            break
        current, depth = parent, depth + 1


def _quota(path, version):
    if version == "v2":
        raw, error = _read_text(path / "cpu.max")
        if error:
            return None, error
        parts = raw.split()
        if len(parts) != 2:
            return None, "invalid"
        quota_raw, period_raw = parts
    else:
        quota_raw, quota_error = _read_text(path / "cpu.cfs_quota_us")
        period_raw, period_error = _read_text(path / "cpu.cfs_period_us")
        if quota_error or period_error:
            return None, "unreadable" if "unreadable" in (quota_error, period_error) else "missing"
    try:
        period = int(period_raw)
        quota = None if quota_raw.strip() == "max" else int(quota_raw)
        if period <= 0 or quota is not None and quota != -1 and quota <= 0:
            return None, "invalid"
        if version == "v2" and quota == -1:
            return None, "invalid"
        if quota in (None, -1):
            return {"version": version, "limited": False, "period_us": period}, None
        return {"version": version, "limited": True, "quota_us": quota,
                "period_us": period, "cores": quota / period,
                "floor_cores": quota // period}, None
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"


def detect_cpu_capacity(*, cpu_count=_UNSET, affinity=_UNSET,
                        proc_root="/proc", cgroup_root="/sys/fs/cgroup"):
    """Return the tightest visible CPU limit, rounding fractional quotas down.

    ``cpu_count`` and ``affinity`` can be supplied for deterministic inspection
    and tests. ``None`` means that constraint is unavailable. By default they
    come from :func:`os.cpu_count` and :func:`os.sched_getaffinity`. The Linux
    controller is located through self/cgroup and self/mountinfo, then every
    visible ancestor quota is checked. Conventional mount locations provide a
    fallback when procfs membership or mount information is unavailable.
    """
    warnings = []
    if cpu_count is _UNSET:
        cpu_count = os.cpu_count()
    if cpu_count is not None:
        _integer(cpu_count, "cpu_count", minimum=1)
    if affinity is _UNSET:
        try:
            affinity = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            affinity = None
    if affinity is None:
        affinity_cores = None
    else:
        try:
            cpu_ids = set(affinity)
        except TypeError as error:
            raise ValueError("affinity must be an iterable of CPU indices or None") from error
        if not cpu_ids or any(type(item) is not int or item < 0 for item in cpu_ids):
            raise ValueError("affinity must contain nonnegative integer CPU indices")
        affinity_cores = len(cpu_ids)

    proc = Path(proc_root)
    cgroups_raw, cgroups_error = _read_text(proc / "self" / "cgroup")
    mounts_raw, mounts_error = _read_text(proc / "self" / "mountinfo")
    if cgroups_error:
        warnings.append("cgroup_membership_unavailable")
    if mounts_error:
        warnings.append("cgroup_mounts_unavailable")
    memberships, mounts = _memberships(cgroups_raw), _mounts(mounts_raw)
    locations = []
    for version, root, point in mounts:
        groups = memberships[version]
        if not groups:
            locations.append((version, point, point, "mount_root"))
        for group in groups:
            locations.append((version, _leaf(root, point, group), point, "membership"))
        if root != "/":
            warnings.append("ancestors_outside_visible_mount_not_inspected")

    fallback = Path(cgroup_root)
    for version, point in (("v2", fallback), ("v1", fallback),
                           ("v1", fallback / "cpu"), ("v1", fallback / "cpu,cpuacct")):
        for group in memberships[version]:
            locations.append((version, _leaf("/", point, group), point, "filesystem_fallback"))
        locations.append((version, point, point, "filesystem_fallback"))
    seen, quotas, errors = set(), [], {"invalid": 0, "unreadable": 0}
    for version, leaf, point, source in locations:
        for directory, depth in _ancestors(leaf, point):
            key = (version, directory)
            if key in seen:
                continue
            seen.add(key)
            quota, error = _quota(directory, version)
            if error:
                if error != "missing":
                    errors[error] += 1
                continue
            quota.update(ancestor_depth=depth, source=source)
            quotas.append(quota)
    if errors["invalid"]:
        warnings.append("invalid_quota_files_ignored")
    if errors["unreadable"]:
        warnings.append("quota_files_unreadable")
    limited = [item for item in quotas if item["limited"]]
    quota_floor = min((item["floor_cores"] for item in limited), default=None)
    quota_cores = min((item["cores"] for item in limited), default=None)
    limits = [value for value in (cpu_count, affinity_cores, quota_floor) if value is not None]
    if limits:
        effective = min(limits)
    else:
        effective = 1
        warnings.append("cpu_constraints_unavailable_using_one_core")
    return {"effective_cores": effective, "host_cpu_count": cpu_count,
            "affinity_cores": affinity_cores, "quota_cores": quota_cores,
            "quota_floor_cores": quota_floor, "quota_constraints": quotas,
            "quota_read_errors": errors, "warnings": sorted(set(warnings)),
            "rounding": "floor", "scope": "visible_affinity_and_cgroup_ancestors"}


def allocate_cpu_budget(num_gpu_workers, *, requested="auto", reserve_cores=4,
                        background_cores=4, main_threads_per_worker=1,
                        cache_enabled=True, prefetch_enabled=True, detection=None):
    """Allocate a *global* preparation-process count without oversubscription.

    Active prefetch reserves one parent loader/prepare thread per GPU worker.
    That thread performs verified cache reads while process-pool children build
    compact cache entries. With no pool it performs the original preparation.
    Child processes are enabled only when caching and prefetch are both enabled;
    each child uses one CPU thread. Explicit counts are capped to available
    cores and distributed by quotient and remainder in worker order.
    """
    _integer(num_gpu_workers, "num_gpu_workers", minimum=1)
    _integer(reserve_cores, "reserve_cores")
    _integer(background_cores, "background_cores")
    _integer(main_threads_per_worker, "main_threads_per_worker", minimum=1)
    if requested != "auto":
        _integer(requested, "requested")
    if type(cache_enabled) is not bool or type(prefetch_enabled) is not bool:
        raise ValueError("cache_enabled and prefetch_enabled must be booleans")
    detection = detect_cpu_capacity() if detection is None else dict(detection)
    effective = _integer(detection.get("effective_cores"), "effective_cores")
    main_total = num_gpu_workers * main_threads_per_worker
    loader_total = num_gpu_workers if prefetch_enabled else 0
    fixed = reserve_cores + background_cores + main_total + loader_total
    if fixed > effective:
        raise ValueError(
            f"CPU budget requires {fixed} cores for {num_gpu_workers} workers "
            f"({main_total} main, {loader_total} loader, {background_cores} background, "
            f"{reserve_cores} reserve), but only {effective} effective cores are available; "
            "reduce workers, thread counts, or explicit reserves")
    available = effective - fixed
    pool_enabled = cache_enabled and prefetch_enabled
    process_total = (available if requested == "auto" else min(requested, available)) if pool_enabled else 0
    quotient, remainder = divmod(process_total, num_gpu_workers)
    allocations = [{"main_threads": main_threads_per_worker,
                    "prepare_loader_threads": int(prefetch_enabled),
                    "prepare_processes": quotient + int(index < remainder)}
                   for index in range(num_gpu_workers)]
    allocated = main_total + loader_total + process_total
    return {"effective_cores": effective, "reserve_cores": reserve_cores,
            "background_cores": background_cores, "main_thread_total": main_total,
            "prepare_loader_threads": loader_total, "prepare_process_total": process_total,
            "allocated_cores": allocated, "reserved_cores": reserve_cores + background_cores,
            "total_budgeted_cores": allocated + reserve_cores + background_cores,
            "unallocated_cores": effective - allocated - reserve_cores - background_cores,
            "requested_prepare_processes": requested,
            "prepare_process_capacity": available if pool_enabled else 0,
            "requested_count_capped": requested != "auto" and process_total < requested and pool_enabled,
            "cache_enabled": cache_enabled, "prefetch_enabled": prefetch_enabled,
            "worker_allocations": allocations, "detection": detection}
