"""CPU quota and admission tests; these fixtures are not timing benchmarks."""

import json
from pathlib import Path

import pytest

from staar_phewas.cpu_budget import allocate_cpu_budget, detect_cpu_capacity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def mount_field(value):
    return str(value).replace("\\", "\\134").replace(" ", "\\040")


def filesystem(tmp_path, *, version="v2", membership="/jobs/worker", mount_root="/",
               point_name="controller", mount=True):
    proc, fallback, point = tmp_path / "proc", tmp_path / "fallback", tmp_path / point_name
    point.mkdir()
    if version == "v2":
        write(proc / "self/cgroup", "0::" + membership + "\n")
        mountinfo = f"30 1 0:20 {mount_field(mount_root)} {mount_field(point)} rw - cgroup2 cgroup rw\n"
    else:
        write(proc / "self/cgroup", "2:cpu,cpuacct:" + membership + "\n")
        mountinfo = f"30 1 0:20 {mount_field(mount_root)} {mount_field(point)} rw - cgroup cgroup rw,cpu,cpuacct\n"
    if mount:
        write(proc / "self/mountinfo", mountinfo)
    return proc, fallback, point


def detect(proc, fallback, *, affinity=range(96), cpu_count=128):
    return detect_cpu_capacity(cpu_count=cpu_count, affinity=affinity,
                               proc_root=proc, cgroup_root=fallback)


def v1quota(directory, quota, period=100000):
    write(directory / "cpu.cfs_quota_us", str(quota))
    write(directory / "cpu.cfs_period_us", str(period))


def test_v2_reads_tightest_ancestor_and_floors_fractional_quota(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    write(point / "cpu.max", "4000000 100000")
    write(point / "jobs/cpu.max", "250000 100000")
    write(point / "jobs/worker/cpu.max", "800000 100000")
    result = detect(proc, fallback)
    assert result["effective_cores"] == 2
    assert result["quota_cores"] == 2.5
    assert result["quota_floor_cores"] == 2
    assert sorted(item["ancestor_depth"] for item in result["quota_constraints"]) == [0, 1, 2]


def test_v1_reads_ancestor_quota_not_just_leaf(tmp_path):
    proc, fallback, point = filesystem(tmp_path, version="v1")
    v1quota(point, 4000000)
    v1quota(point / "jobs", 399999)
    v1quota(point / "jobs/worker", -1)
    result = detect(proc, fallback)
    assert result["effective_cores"] == 3
    assert result["quota_cores"] == 3.99999
    assert len(result["quota_constraints"]) == 3
    assert result["quota_constraints"][0]["limited"] is False


def test_affinity_is_not_replaced_by_full_host_cpu_count(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    write(point / "cpu.max", "800000 100000")
    result = detect(proc, fallback, affinity={1, 8, 63}, cpu_count=128)
    assert result["host_cpu_count"] == 128
    assert result["affinity_cores"] == 3
    assert result["effective_cores"] == 3


def test_host_count_is_also_a_constraint(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    write(point / "cpu.max", "max 100000")
    assert detect(proc, fallback, cpu_count=4)["effective_cores"] == 4


def test_non_root_mount_maps_matching_membership_and_checks_visible_root(tmp_path):
    proc, fallback, point = filesystem(tmp_path, membership="/private-namespace/jobs/worker",
                                       mount_root="/private-namespace")
    write(point / "cpu.max", "120000 100000")
    write(point / "jobs/worker/cpu.max", "400000 100000")
    result = detect(proc, fallback)
    assert result["effective_cores"] == 1
    assert "ancestors_outside_visible_mount_not_inspected" in result["warnings"]


def test_namespace_relative_membership_and_escaped_mountpoint(tmp_path):
    proc, fallback, point = filesystem(tmp_path, membership="/jobs/worker",
                                       mount_root="/private-namespace", point_name="cpu controller")
    write(point / "cpu.max", "max 100000")
    write(point / "jobs/worker/cpu.max", "600000 100000")
    assert detect(proc, fallback)["effective_cores"] == 6


def test_namespace_root_membership_reads_mount_root(tmp_path):
    proc, fallback, point = filesystem(tmp_path, membership="/", mount_root="/private-namespace")
    write(point / "cpu.max", "700000 100000")
    assert detect(proc, fallback)["effective_cores"] == 7


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_filesystem_fallback_when_procfs_is_missing(tmp_path, version):
    fallback = tmp_path / "fallback"
    if version == "v2":
        write(fallback / "cpu.max", "900000 100000")
    else:
        v1quota(fallback / "cpu,cpuacct", 900000)
    result = detect(tmp_path / "missing-proc", fallback)
    assert result["effective_cores"] == 9
    assert result["quota_constraints"][0]["source"] == "filesystem_fallback"
    assert "cgroup_membership_unavailable" in result["warnings"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_filesystem_fallback_still_checks_nested_membership_without_mountinfo(tmp_path, version):
    proc, fallback, _ = filesystem(tmp_path, version=version, mount=False)
    point = fallback if version == "v2" else fallback / "cpu"
    if version == "v2":
        write(point / "cpu.max", "4000000 100000")
        write(point / "jobs/cpu.max", "350000 100000")
    else:
        v1quota(point, 4000000)
        v1quota(point / "jobs", 350000)
    result = detect(proc, fallback)
    assert result["effective_cores"] == 3
    assert "cgroup_mounts_unavailable" in result["warnings"]


def test_mount_root_quota_without_membership(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    (proc / "self/cgroup").unlink()
    write(point / "cpu.max", "500000 100000")
    assert detect(proc, fallback)["effective_cores"] == 5


def test_hybrid_controller_limits_are_combined(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    v1point = tmp_path / "cpu-v1"
    write(proc / "self/cgroup", "0::/jobs/worker\n2:cpu,cpuacct:/jobs/worker\n")
    with (proc / "self/mountinfo").open("a", encoding="utf-8") as stream:
        stream.write(f"31 1 0:21 / {mount_field(v1point)} rw - cgroup cgroup rw,cpu,cpuacct\n")
    write(point / "cpu.max", "600000 100000")
    v1quota(v1point / "jobs", 450000)
    assert detect(proc, fallback)["effective_cores"] == 4


def test_malformed_quota_is_reported_while_valid_ancestor_still_applies(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    write(point / "cpu.max", "200000 100000")
    write(point / "jobs/worker/cpu.max", "400000 0")
    result = detect(proc, fallback)
    assert result["effective_cores"] == 2
    assert result["quota_read_errors"]["invalid"] == 1
    assert "invalid_quota_files_ignored" in result["warnings"]


def test_sub_one_core_quota_does_not_round_up(tmp_path):
    proc, fallback, point = filesystem(tmp_path)
    write(point / "cpu.max", "99999 100000")
    result = detect(proc, fallback)
    assert result["effective_cores"] == 0
    with pytest.raises(ValueError, match="only 0 effective cores"):
        allocate_cpu_budget(1, reserve_cores=0, background_cores=0, detection=result)


def test_unknown_constraints_use_conservative_one_core(tmp_path):
    result = detect(tmp_path / "proc", tmp_path / "fallback", cpu_count=None, affinity=None)
    assert result["effective_cores"] == 1
    assert "cpu_constraints_unavailable_using_one_core" in result["warnings"]


def test_report_does_not_include_mount_paths_or_membership_names(tmp_path):
    proc, fallback, point = filesystem(tmp_path, membership="/private-namespace/worker")
    write(point / "private-namespace/worker/cpu.max", "4000000 100000")
    result = detect(proc, fallback)
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded
    assert "private-namespace" not in encoded
    assert "worker" not in encoded


def test_membership_parent_traversal_cannot_read_above_mount(tmp_path):
    proc, fallback, point = filesystem(tmp_path, membership="/../../")
    write(tmp_path / "cpu.max", "100000 100000")
    write(point / "cpu.max", "4000000 100000")
    assert detect(proc, fallback)["effective_cores"] == 40


def capacity(cores):
    return {"effective_cores": cores, "source": "anonymous_test_fixture"}


def test_default_40_core_budget_counts_loaders_and_16_processes():
    result = allocate_cpu_budget(8, detection=capacity(40))
    assert result["main_thread_total"] == 8
    assert result["prepare_loader_threads"] == 8
    assert result["prepare_process_total"] == 16
    assert result["reserve_cores"] == result["background_cores"] == 4
    assert result["allocated_cores"] == 32
    assert result["total_budgeted_cores"] == 40
    assert result["unallocated_cores"] == 0
    assert result["worker_allocations"] == [dict(main_threads=1, prepare_loader_threads=1,
                                                prepare_processes=2)] * 8


def test_explicit_count_uses_global_quotient_and_remainder():
    result = allocate_cpu_budget(8, requested=13, detection=capacity(40))
    assert [item["prepare_processes"] for item in result["worker_allocations"]] == [2] * 5 + [1] * 3
    assert result["prepare_process_total"] == 13
    assert result["unallocated_cores"] == 3
    assert result["requested_count_capped"] is False


def test_small_pool_keeps_all_fallback_thread_costs_reserved():
    result = allocate_cpu_budget(8, requested=3, detection=capacity(40))
    assert [item["prepare_processes"] for item in result["worker_allocations"]] == [1] * 3 + [0] * 5
    assert result["prepare_loader_threads"] == 8
    assert result["allocated_cores"] == 19


def test_explicit_oversized_count_is_capped_and_recorded():
    result = allocate_cpu_budget(8, requested=100, detection=capacity(40))
    assert result["requested_prepare_processes"] == 100
    assert result["requested_count_capped"] is True
    assert result["prepare_process_total"] == 16
    assert result["total_budgeted_cores"] == 40


@pytest.mark.parametrize("options", [{"requested": 0}, {"cache_enabled": False}])
def test_thread_route_has_no_pool_and_counts_one_prepare_thread_per_worker(options):
    result = allocate_cpu_budget(8, detection=capacity(40), **options)
    assert result["prepare_process_total"] == 0
    assert result["prepare_loader_threads"] == 8
    assert result["allocated_cores"] == 16
    assert all(item["prepare_processes"] == 0 for item in result["worker_allocations"])


def test_disabled_prefetch_has_neither_loader_nor_pool():
    result = allocate_cpu_budget(8, requested=100, prefetch_enabled=False, detection=capacity(40))
    assert result["prepare_process_total"] == result["prepare_loader_threads"] == 0
    assert result["allocated_cores"] == 8
    assert result["total_budgeted_cores"] == 16


def test_explicit_single_worker_preflight_budget():
    result = allocate_cpu_budget(1, requested=4, background_cores=0, detection=capacity(40))
    assert result["worker_allocations"] == [dict(main_threads=1, prepare_loader_threads=1,
                                               prepare_processes=4)]
    assert result["allocated_cores"] == 6
    assert result["total_budgeted_cores"] == 10


def test_main_thread_count_is_multiplied_by_all_gpu_workers():
    result = allocate_cpu_budget(8, main_threads_per_worker=2, detection=capacity(40))
    assert result["main_thread_total"] == 16
    assert result["prepare_process_total"] == 8


def test_exact_fixed_budget_allows_original_thread_route():
    result = allocate_cpu_budget(8, detection=capacity(24))
    assert result["prepare_process_total"] == 0
    assert result["total_budgeted_cores"] == 24


def test_insufficient_cpu_budget_fails_before_worker_launch():
    with pytest.raises(ValueError, match="requires 24 cores.*only 23 effective cores"):
        allocate_cpu_budget(8, detection=capacity(23))


def test_capacity_is_refreshed_instead_of_cached_between_launches(monkeypatch):
    capacities = iter([capacity(40), capacity(32)])
    monkeypatch.setattr("staar_phewas.cpu_budget.detect_cpu_capacity", lambda: next(capacities))
    assert allocate_cpu_budget(8)["prepare_process_total"] == 16
    assert allocate_cpu_budget(8)["prepare_process_total"] == 8


@pytest.mark.parametrize("name,value", [
    ("num_gpu_workers", 0), ("num_gpu_workers", True), ("requested", -1),
    ("requested", 1.5), ("requested", "0"), ("requested", True),
    ("reserve_cores", -1), ("reserve_cores", False), ("background_cores", 1.5),
    ("main_threads_per_worker", 0), ("cache_enabled", 1), ("prefetch_enabled", 0)])
def test_invalid_allocation_parameters_are_rejected(name, value):
    options = {"num_gpu_workers": 8, "detection": capacity(40), name: value}
    with pytest.raises(ValueError):
        allocate_cpu_budget(**options)


@pytest.mark.parametrize("options", [
    {"cpu_count": 0}, {"cpu_count": True}, {"affinity": []},
    {"affinity": [-1]}, {"affinity": [True]}, {"affinity": 3}])
def test_invalid_detection_inputs_are_rejected(tmp_path, options):
    arguments = dict(cpu_count=8, affinity=range(8), proc_root=tmp_path / "proc",
                     cgroup_root=tmp_path / "cgroup")
    arguments.update(options)
    with pytest.raises(ValueError):
        detect_cpu_capacity(**arguments)
