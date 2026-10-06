"""Independent timing/work-count oracles; never performance benchmarks."""
import copy
import json
import math
import unittest

from torchwgs.runtime_estimate import (StageTimers, plan_windows, estimate_linear,
                                      estimate_gene_strata, optimization_summary)


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class StageTimerTests(unittest.TestCase):
    def test_nested_times_use_exclusive_arithmetic_without_double_counting(self):
        clock = ManualClock()
        timers = StageTimers(clock=clock)
        with timers.span("outer"):
            clock.advance(2.)
            with timers.span("child"):
                clock.advance(3.)
            clock.advance(7.)
        summary = timers.summary()
        self.assertEqual(summary["outer"]["seconds_inclusive"], 12.)
        self.assertEqual(summary["outer"]["seconds_exclusive"], 9.)
        self.assertEqual(summary["child"]["seconds_inclusive"], 3.)
        self.assertEqual(sum(s["seconds_exclusive"] for s in summary.values()), 12.)
        self.assertEqual(sum(s["seconds_inclusive"] for s in summary.values()), 15.)

    def test_recursive_same_stage_still_partitions_wall_time(self):
        clock = ManualClock()
        timers = StageTimers(clock=clock)
        with timers.span("same"):
            clock.advance(1.)
            with timers.span("same"):
                clock.advance(4.)
            clock.advance(2.)
        entry = timers.summary()["same"]
        self.assertEqual(entry["calls"], 2)
        self.assertEqual(entry["seconds_inclusive"], 11.)
        self.assertEqual(entry["seconds_exclusive"], 7.)

    def test_repeated_calls_record_first_maximum_and_success_counts(self):
        clock = ManualClock()
        timers = StageTimers(clock=clock)
        for seconds in (2., 5., 3.):
            with timers.span("read"):
                clock.advance(seconds)
        entry = timers.summary()["read"]
        self.assertEqual(entry["calls"], 3)
        self.assertEqual(entry["completed_calls"], 3)
        self.assertEqual(entry["failed_calls"], 0)
        self.assertEqual(entry["first_call_seconds"], 2.)
        self.assertEqual(entry["maximum_call_seconds"], 5.)
        self.assertEqual(entry["seconds_exclusive"], 10.)
        self.assertEqual(entry["cuda_calls"], 0)

    def test_exception_is_recorded_and_original_type_survives(self):
        clock = ManualClock()
        timers = StageTimers(clock=clock)
        with self.assertRaisesRegex(ArithmeticError, "independent sentinel"):
            with timers.span("failed"):
                clock.advance(4.)
                raise ArithmeticError("independent sentinel")
        with timers.span("next"):
            clock.advance(2.)
        summary = timers.summary()
        self.assertEqual(summary["failed"]["failed_calls"], 1)
        self.assertEqual(summary["failed"]["completed_calls"], 0)
        self.assertEqual(summary["failed"]["seconds_exclusive"], 4.)
        self.assertEqual(summary["next"]["seconds_exclusive"], 2.)

    def test_caught_child_failure_counts_time_once_and_parent_completes(self):
        clock = ManualClock()
        timers = StageTimers(clock=clock)
        with timers.span("parent"):
            clock.advance(1.)
            try:
                with timers.span("child"):
                    clock.advance(3.)
                    raise LookupError("synthetic failure")
            except LookupError:
                pass
            clock.advance(2.)
        summary = timers.summary()
        self.assertEqual(summary["parent"]["completed_calls"], 1)
        self.assertEqual(summary["parent"]["seconds_exclusive"], 3.)
        self.assertEqual(summary["child"]["failed_calls"], 1)

    def test_cuda_sync_waits_are_both_included_in_instrumented_wall_time(self):
        clock = ManualClock()
        sync_calls = []

        def synchronize():
            sync_calls.append(clock.value)
            clock.advance(.25)

        timers = StageTimers(synchronize=synchronize, clock=clock)
        with timers.span("cuda_work", cuda=True):
            clock.advance(1.)
        entry = timers.summary()["cuda_work"]
        self.assertEqual(len(sync_calls), 2)
        self.assertEqual(entry["seconds_inclusive"], 1.5)
        self.assertEqual(entry["cuda_calls"], 1)

    def test_cuda_async_failure_does_not_mark_call_successful(self):
        clock = ManualClock()
        calls = []

        def synchronize():
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("completion sentinel")

        timers = StageTimers(synchronize=synchronize, clock=clock)
        with self.assertRaisesRegex(RuntimeError, "completion sentinel"):
            with timers.span("cuda_failed", cuda=True):
                clock.advance(3.)
        with timers.span("subsequent"):
            clock.advance(1.)
        entry = timers.summary()["cuda_failed"]
        self.assertEqual(entry["completed_calls"], 0)
        self.assertEqual(entry["failed_calls"], 1)
        self.assertEqual(entry["seconds_inclusive"], 3.)
        self.assertEqual(timers.summary()["subsequent"]["seconds_exclusive"], 1.)

    def test_cpu_stage_does_not_call_cuda_synchronizer(self):
        clock = ManualClock()

        def forbidden():
            self.fail("CPU span synchronized CUDA")

        timers = StageTimers(synchronize=forbidden, clock=clock)
        with timers.span("cpu_stage"):
            clock.advance(1.)
        self.assertEqual(timers.summary()["cpu_stage"]["cuda_calls"], 0)

    def test_summary_cannot_mutate_timer_ledger(self):
        clock = ManualClock()
        timers = StageTimers(clock=clock)
        with timers.span("stage"):
            clock.advance(2.)
        summary = timers.summary()
        summary["stage"]["seconds_exclusive"] = 123456.
        self.assertEqual(timers.summary()["stage"]["seconds_exclusive"], 2.)

    def test_private_labels_are_rejected_without_echoing(self):
        for name in ("/private/sensitive_trait", "123-4.5", "name\nsecret", "", "a" * 65):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as error:
                    with StageTimers().span(name):
                        pass
                if name:
                    self.assertNotIn(name, str(error.exception))


class WindowPlanningTests(unittest.TestCase):
    def test_all_source_blocks_cover_exactly_once_including_partial_tail(self):
        windows = plan_windows(23, 4, 6, seed=91)
        self.assertEqual(windows, [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23)])
        covered = [i for start, stop in windows for i in range(start, stop)]
        self.assertEqual(covered, list(range(23)))

    def test_sampling_is_seed_reproducible_aligned_and_nonoverlapping(self):
        windows = plan_windows(13733596, 1000, 7, seed=719)
        self.assertEqual(windows, plan_windows(13733596, 1000, 7, seed=719))
        self.assertEqual(len(windows), 7)
        self.assertEqual(windows, sorted(windows))
        self.assertEqual(len({start for start, _ in windows}), 7)
        for index, (start, stop) in enumerate(windows):
            self.assertEqual(start % 1000, 0)
            self.assertGreater(stop, start)
            self.assertLessEqual(stop - start, 1000)
            self.assertLessEqual(stop, 13733596)
            if index:
                self.assertLessEqual(windows[index - 1][1], start)

    def test_one_window_is_selected_per_spatial_stratum(self):
        blocks, count = 103, 9
        windows = plan_windows(blocks * 4, 4, count, seed=-17)
        for ordinal, (start, _) in enumerate(windows):
            self.assertGreaterEqual(start // 4, ordinal * blocks // count)
            self.assertLess(start // 4, (ordinal + 1) * blocks // count)

    def test_empty_sources_and_oversampling_never_duplicate_blocks(self):
        self.assertEqual(plan_windows(0, 4, 10), [])
        self.assertEqual(plan_windows(17, 4, 0), [])
        self.assertEqual(plan_windows(3, 4, 100), [(0, 3)])

    def test_invalid_geometry_rejects_boolean_noninteger_and_negative(self):
        for args in ((True, 4, 2), (10, False, 2), (10, 4, True),
                     (10., 4, 2), (-1, 4, 2), (10, 0, 2), (10, 4, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan_windows(*args)


class LinearEstimateTests(unittest.TestCase):
    def test_independent_weighted_rate_and_observed_range(self):
        result = estimate_linear([{"units": 1, "seconds": 1},
                                  {"units": 9, "seconds": 27}], 100)
        self.assertEqual(result["measured_units"], 10)
        self.assertEqual(result["measured_seconds"], 28)
        self.assertEqual(result["seconds_per_unit"], 2.8)
        self.assertEqual(result["total_seconds"], 280)
        self.assertEqual(result["range_seconds"], {"min": 100, "max": 300})
        self.assertIn("not_confidence_interval", result["range_kind"])

    def test_empty_observation_never_implies_zero_measured_runtime(self):
        for size in (0, 100):
            result = estimate_linear([], size)
            self.assertIsNone(result["total_seconds"])
            self.assertIsNone(result["range_seconds"])
            self.assertIsNone(result["seconds_per_unit"])
            self.assertEqual(result["observations"], 0)

    def test_censored_observations_are_rejected_instead_of_averaged(self):
        for complete in (False, 0, 1, "true", None):
            with self.subTest(complete=complete), self.assertRaises(ValueError):
                estimate_linear([{"units": 3, "seconds": 9, "complete": complete}], 30)
        self.assertEqual(estimate_linear(
            [{"units": 3, "seconds": 9, "complete": True}], 30)["total_seconds"], 90)

    def test_invalid_numeric_records_and_nonfinite_overflow_fail(self):
        for units, seconds in ((0, 1), (-1, 1), (True, 1), (1, True),
                               (1, -1), (1, math.nan), (1, math.inf)):
            with self.subTest(units=units, seconds=seconds), self.assertRaises(ValueError):
                estimate_linear([{"units": units, "seconds": seconds}], 100)
        with self.assertRaises(ValueError):
            estimate_linear([{"units": 1, "seconds": 1e308}], 1e308)


class GeneStratumEstimateTests(unittest.TestCase):
    def test_stratified_point_range_and_exact_fully_observed_stratum(self):
        census = [{"stratum": "small", "count": 10}, {"stratum": "large", "count": 2}]
        records = [{"stratum": name, "seconds": sec, "complete": True}
                   for name, sec in (("small", 2), ("small", 4), ("large", 10), ("large", 20))]
        result = estimate_gene_strata(census, records)
        self.assertEqual(result["total_seconds"], 60)
        self.assertEqual(result["range_seconds"], {"min": 52, "max": 68})
        self.assertEqual(result["lower_bound_seconds"], 36)
        strata = {item["stratum"]: item for item in result["strata"]}
        self.assertEqual(strata["small"]["mean_complete_seconds"], 3)
        self.assertEqual(strata["small"]["range_seconds"], {"min": 22, "max": 38})
        self.assertEqual(strata["large"]["range_seconds"], {"min": 30, "max": 30})
        self.assertIn("not_confidence_interval", result["range_kind"])

    def test_one_censored_attempt_blocks_whole_estimate_even_with_complete_peer(self):
        result = estimate_gene_strata(
            [{"stratum": "group", "count": 4}],
            [{"stratum": "group", "seconds": 2, "complete": True},
             {"stratum": "group", "seconds": 9, "complete": False}])
        self.assertIsNone(result["total_seconds"])
        self.assertIsNone(result["range_seconds"])
        self.assertEqual(result["lower_bound_seconds"], 11)
        self.assertEqual(result["censored_strata"], ["group"])
        self.assertEqual(result["missing_strata"], [])
        self.assertEqual(result["strata"][0]["mean_complete_seconds"], 2)
        self.assertIsNone(result["strata"][0]["total_seconds"])

    def test_unsampled_positive_stratum_blocks_total_but_zero_census_does_not(self):
        result = estimate_gene_strata(
            [{"stratum": "empty", "count": 0}, {"stratum": "unseen", "count": 1}], [])
        self.assertEqual(result["missing_strata"], ["unseen"])
        self.assertIsNone(result["total_seconds"])
        self.assertEqual(result["lower_bound_seconds"], 0)
        empty = estimate_gene_strata([{"stratum": "empty", "count": 0}], [])
        self.assertEqual(empty["total_seconds"], 0)
        self.assertEqual(empty["range_seconds"], {"min": 0, "max": 0})

    def test_analysis_categories_are_not_pooled(self):
        result = estimate_gene_strata(
            [{"stratum": "a0_small", "count": 4}, {"stratum": "a1_small", "count": 4}],
            [{"stratum": "a0_small", "seconds": 3, "complete": True}])
        self.assertEqual(result["missing_strata"], ["a1_small"])
        self.assertIsNone(result["total_seconds"])

    def test_invalid_counts_unmatched_categories_and_duplicate_census_fail(self):
        cases = [([{"stratum": "a", "count": True}], []),
                 ([{"stratum": "a", "count": 1}, {"stratum": "a", "count": 1}], []),
                 ([{"stratum": "a", "count": 1}], [{"stratum": "b", "seconds": 1, "complete": True}]),
                 ([{"stratum": "a", "count": 0}], [{"stratum": "a", "seconds": 1, "complete": True}]),
                 ([{"stratum": "a", "count": 1}], [{"stratum": "a", "seconds": 1, "complete": 1}])]
        for census, records in cases:
            with self.subTest(census=census, records=records), self.assertRaises(ValueError):
                estimate_gene_strata(census, records)

    def test_observation_extra_private_data_does_not_enter_aggregate(self):
        records = [{"stratum": "a0_small", "seconds": 2, "complete": True,
                    "private_path": "/private/synthetic", "gene": "synthetic_gene"}]
        result = estimate_gene_strata([{"stratum": "a0_small", "count": 1}], records)
        serialized = json.dumps(result)
        self.assertNotIn("/private/", serialized)
        self.assertNotIn("synthetic_gene", serialized)


class OptimizationSummaryTests(unittest.TestCase):
    def test_exclusive_ranking_is_detached_whitelisted_and_no_speedup_claim(self):
        stages = {
            "projection": {"calls": 2, "completed_calls": 1, "failed_calls": 1,
                           "seconds_exclusive": 3, "seconds_inclusive": 300,
                           "private_path": "/private/synthetic"},
            "read": {"calls": 1, "completed_calls": 1, "seconds_exclusive": 1}}
        untouched = copy.deepcopy(stages)
        result = optimization_summary(stages)
        self.assertEqual(result["measured_exclusive_seconds"], 4)
        self.assertEqual([item["stage"] for item in result["ranked_stages"]], ["projection", "read"])
        self.assertEqual(result["ranked_stages"][0]["fraction_of_measured_exclusive_time"], .75)
        self.assertTrue(result["recommendations"][0]["contains_failed_calls"])
        self.assertIn("matrix dimensions", result["recommendations"][0]["suggestion"])
        self.assertFalse(result["speedup_claimed"])
        self.assertEqual(stages, untouched)
        self.assertNotIn("/private/", json.dumps(result))
        self.assertNotIn("seconds_inclusive", json.dumps(result))

    def test_ties_have_deterministic_order_and_zero_has_no_recommendation(self):
        result = optimization_summary({"z": {"seconds_exclusive": 0}, "a": {"seconds_exclusive": 0}})
        self.assertEqual([s["stage"] for s in result["ranked_stages"]], ["a", "z"])
        self.assertEqual(result["recommendations"], [])
        self.assertTrue(all(s["fraction_of_measured_exclusive_time"] == 0 for s in result["ranked_stages"]))

    def test_impossible_counts_and_nonfinite_costs_fail(self):
        for record in ({"seconds_exclusive": -1}, {"seconds_exclusive": math.nan},
                       {"seconds_exclusive": 1, "calls": True},
                       {"seconds_exclusive": 1, "calls": 1, "completed_calls": 1, "failed_calls": 1}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                optimization_summary({"stage": record})


if __name__ == "__main__":
    unittest.main()
