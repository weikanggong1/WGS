"""Three-input alignment and durable execution units; these are not benchmarks."""
import importlib
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

workflow = importlib.import_module("staar_phewas.run")


class ThreeInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.phenotype = self.root / "phenotypes.csv"
        self.covariate = self.root / "covariates.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def inputs(self):
        self.phenotype.write_text("eid,trait_a,trait_b\n1002,1,9\n99,3,NA\n101,4,6\n103,8,2\n105,7,7\n107,6,4\n109,5,5\n111,2,8\n")
        self.covariate.write_text("eid,adjustment\n111,7\n109,6\n107,5\n105,4\n103,3\n101,2\n99,1\n1002,0\n")
        return workflow.read_csv_inputs(self.phenotype,self.covariate)

    def dataset(self):
        root = self.root / "cache"
        root.mkdir()
        ids = np.asarray([1002,99,101,103,105,107,109,111],dtype=np.int64)
        np.save(root / "sample_ids.npy",ids)
        (root / "cache_dataset.json").write_text(json.dumps(dict(schema_version=1,sample_ids="sample_ids.npy",
            chromosomes=[dict(name="1",container_directory="chr01",metadata_directory="chr01/metadata")],
            annotation_catalog={},annotation_names=[])))
        return root

    def test_alignment_is_numeric_ordered_and_complete_cases_are_trait_specific(self):
        phenotypes,covariates = self.inputs()
        ids,y,x,names,counts = workflow.aligned_phenotype(phenotypes,covariates,[1002,99,101,103,105,107,109,111],"trait_a")
        self.assertEqual(ids.tolist(),["99","101","103","105","107","109","111","1002"])
        np.testing.assert_array_equal(x[:,0],np.ones(8))
        self.assertEqual(names,("Intercept","adjustment"))
        self.assertEqual(counts["analysis_samples"],8)
        second = workflow.aligned_phenotype(phenotypes,covariates,ids,"trait_b")
        self.assertEqual(second[0].tolist(),["101","103","105","107","109","111","1002"])
        self.assertEqual(second[-1]["excluded_missing_rows"],1)

    def test_covariate_missing_and_cache_absence_are_distinct_counts(self):
        phenotypes,covariates = self.inputs()
        covariates.values[4,0] = np.nan
        selected = workflow.aligned_phenotype(phenotypes,covariates,[99,101,103,105,107,109,111],"trait_a")
        self.assertEqual(selected[-1]["phenotype_rows_absent_from_cache"],1)
        self.assertEqual(selected[-1]["excluded_missing_rows"],1)
        self.assertEqual(selected[-1]["analysis_samples"],6)

    def test_existing_intercept_is_not_duplicated_and_no_covariate_file_is_valid(self):
        self.inputs()
        self.covariate.write_text("eid,constant\n" + "".join(f"{eid},1\n" for eid in [1002,99,101,103,105,107,109,111]))
        phenotypes,covariates = workflow.read_csv_inputs(self.phenotype,self.covariate)
        selected = workflow.aligned_phenotype(phenotypes,covariates,phenotypes.sample_ids,"trait_a")
        self.assertFalse(selected[-1]["intercept_added"])
        self.assertEqual(selected[2].shape,(8,1))
        self.covariate.write_text("eid\n" + "".join(f"{eid}\n" for eid in [1002,99,101,103,105,107,109,111]))
        phenotypes,covariates = workflow.read_csv_inputs(self.phenotype,self.covariate)
        self.assertEqual(workflow.aligned_phenotype(phenotypes,covariates,phenotypes.sample_ids,"trait_a")[2].shape,(8,1))

    def test_duplicate_and_nonstandard_ids_are_rejected_without_float_roundtrip(self):
        self.covariate.write_text("eid,x\n1,2\n")
        for content in ("eid,y\n1,2\n1,3\n", "eid,y\n1000012.0,2\n", "eid,y\n001002,2\n", "eid,y\n9223372036854775808,2\n"):
            with self.subTest(content=content):
                self.phenotype.write_text(content)
                with self.assertRaises(ValueError):
                    workflow.read_csv_inputs(self.phenotype,self.covariate)

    def test_invalid_headers_covariates_and_infinity_are_rejected(self):
        self.covariate.write_text("eid,x\n1,2\n")
        for content in ("id,y\n1,2\n", "eid,y,y\n1,2,3\n", "eid,y\n1,infinity\n", "eid,y\n1,category\n", "eid,y\n1,2,3\n"):
            self.phenotype.write_text(content)
            with self.assertRaises(ValueError):
                workflow.read_csv_inputs(self.phenotype,self.covariate)

    def test_rank_deficiency_fails_before_genotype_reading(self):
        phenotypes,covariates = self.inputs()
        duplicate = workflow.NumericCSV(covariates.sample_ids,("first","second"),np.column_stack((covariates.values,covariates.values)))
        with self.assertRaisesRegex(ValueError,"rank deficient"):
            workflow.aligned_phenotype(phenotypes,duplicate,phenotypes.sample_ids,"trait_a")

    def test_preparation_fits_actual_ols_and_resume_is_bound_to_inputs(self):
        phenotypes,covariates = self.inputs()
        cache = self.dataset()
        output = self.root / "output"
        plan_path = workflow.prepare_run(self.phenotype,self.covariate,cache,output_directory=output,analyses=["individual"])
        plan = json.loads(plan_path.read_text())
        self.assertEqual(plan["task_count"],2)
        from staar_phewas.io import load_null_model
        model = load_null_model(plan["phenotypes"][0]["path"])
        ids,y,x,*_ = workflow.aligned_phenotype(phenotypes,covariates,phenotypes.sample_ids,"trait_a")
        expected = np.linalg.lstsq(x,y,rcond=None)[0]
        np.testing.assert_allclose(model.coefficients.cpu().numpy(),expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(model.scaled_residuals.cpu().numpy(),(y-x@expected)/np.sum((y-x@expected)**2)*(len(y)-x.shape[1]),rtol=1e-12,atol=1e-12)
        self.assertEqual(workflow.prepare_run(self.phenotype,self.covariate,cache,output_directory=output,analyses=["individual"]),plan_path)
        self.phenotype.write_text(self.phenotype.read_text().replace("1002,1,9","1002,2,9"))
        with self.assertRaisesRegex(ValueError,"different plan"):
            workflow.prepare_run(self.phenotype,self.covariate,cache,output_directory=output,analyses=["individual"])

    def test_dataset_cannot_escape_cache_directory(self):
        self.inputs()
        cache = self.dataset()
        document = json.loads((cache / "cache_dataset.json").read_text())
        document["chromosomes"][0]["container_directory"] = "../outside"
        (cache / "cache_dataset.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,"within"):
            workflow.prepare_run(self.phenotype,self.covariate,cache,analyses=["individual"])

    def test_probability_gate_rejects_invalid_outputs(self):
        workflow._probability_gate([{"STAAR-O":0.,"pvalue_log10":1000.,"Score":0.}])
        for row in ({"STAAR-O":float("nan")},{"pvalue":1.2},{"Score_se":float("nan")}):
            with self.assertRaises(ArithmeticError):
                workflow._probability_gate([row])

    def test_host_admission_defers_and_releases_instead_of_deadlocking(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE leases(job_id INTEGER PRIMARY KEY,gib REAL)")
        connection.executemany("INSERT INTO leases VALUES(?,?)",[(1,8),(2,24)])
        connection.commit()
        plan = {"input_identity":{"host_memory_limit_gib":80.,"host_memory_reserve_gib":20.}}
        with mock.patch.object(workflow,"_read_host_memory",return_value=None), mock.patch.object(workflow,"_rss_bytes",return_value=0):
            with self.assertRaises(workflow._HostLeaseDeferred):
                workflow._host_memory_guard(connection,plan,1,n=200_000,m=24_000)
            self.assertEqual(connection.execute("SELECT gib FROM leases WHERE job_id=1").fetchone()[0],8)
            connection.execute("DELETE FROM leases WHERE job_id=2")
            connection.commit()
            workflow._host_memory_guard(connection,plan,1,n=200_000,m=24_000)
        self.assertGreater(connection.execute("SELECT gib FROM leases WHERE job_id=1").fetchone()[0],8)
        connection.close()

    def test_live_cgroup_pressure_is_checked_even_when_other_jobs_have_no_lease(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE leases(job_id INTEGER PRIMARY KEY,gib REAL)")
        plan = {"input_identity":{"host_memory_limit_gib":100.,"host_memory_reserve_gib":20.}}
        live = dict(limit_bytes=100*1024**3,effective_current_bytes=70*1024**3)
        with mock.patch.object(workflow,"_read_host_memory",return_value=live):
            self.assertFalse(workflow._host_budget(connection,plan,desired_gib=20.,extra_gib=20.,excluding_job=1)["admissible"])
        live["effective_current_bytes"] = 10*1024**3
        with mock.patch.object(workflow,"_read_host_memory",return_value=live):
            self.assertTrue(workflow._host_budget(connection,plan,desired_gib=20.,extra_gib=20.,excluding_job=1)["admissible"])
        connection.close()

    def test_pending_metadata_does_not_prevent_claiming_a_ready_later_chromosome(self):
        self.inputs()
        cache = self.dataset()
        document = json.loads((cache/"cache_dataset.json").read_text())
        document["chromosomes"].append(dict(name="21",container_directory="chr21",metadata_directory="chr21/metadata"))
        (cache/"cache_dataset.json").write_text(json.dumps(document))
        ready = cache/"chr21"/"metadata"
        ready.mkdir(parents=True)
        (ready/"manifest.json").write_text("{}")
        (ready/"COMPLETE").write_text(workflow._sha256(ready/"manifest.json"))
        (ready.parent/"COMPLETE").write_text("completed")
        path = workflow.prepare_run(self.phenotype,self.covariate,cache,output_directory=self.root/"pending",analyses=["individual"])
        plan = json.loads(path.read_text())
        connection = sqlite3.connect(path.parent/"jobs.sqlite")
        with mock.patch.object(workflow,"_read_host_memory",return_value=None):
            claimed = workflow._claim(connection,plan)
        self.assertEqual(claimed[1]["chromosome"],"21")
        connection.close()

    def test_late_metadata_binding_rejects_a_second_valid_but_different_manifest(self):
        self.inputs()
        cache = self.dataset()
        path = workflow.prepare_run(self.phenotype,self.covariate,cache,output_directory=self.root/"binding",analyses=["individual"])
        plan = json.loads(path.read_text())
        directory = cache/"chr01"/"metadata"
        directory.mkdir(parents=True)
        connection = sqlite3.connect(path.parent/"jobs.sqlite")
        for content in ("{\"first\":1}","{\"second\":2}"):
            (directory/"manifest.json").write_text(content)
            (directory/"COMPLETE").write_text(workflow._sha256(directory/"manifest.json"))
            if "first" in content:
                workflow._bind_chromosome(connection,plan,plan["chromosomes"][0])
            else:
                with self.assertRaisesRegex(ValueError,"changed since"):
                    workflow._bind_chromosome(connection,plan,plan["chromosomes"][0])
        connection.close()

    def test_single_stream_flushes_complete_groups_and_preserves_global_row_metadata(self):
        from staar_phewas.pipeline import PheWASPipeline
        from staar_phewas.results import single_variant_record
        import rdata
        rows = []
        for position,chunk,common,ref,alt in ((10,0,False,"A","T"),(20,0,True,"C","G"),
                                            (30,1,True,"G","A"),(40,1,False,"T","C"),(50,2,False,"A","G")):
            row = single_variant_record("1",position,ref,alt,.02,.02,8,
                {"pvalue_log":1.,"Score":.2,"Score_se":.3,"Est":.1,"Est_se":.4})
            row.update(_chunk=chunk,_common=common)
            rows.append(row)
        baseline = PheWASPipeline.individual_tables([copy.deepcopy(rows)])[0]
        class Stream:
            individual_tables = staticmethod(PheWASPipeline.individual_tables)
            def iter_individual_records(self,*args,**kwargs):
                # A reader batch crosses an original output-group boundary.
                for batch in (rows[:1],rows[1:3],rows[3:]):
                    yield 0,copy.deepcopy(batch)
        identity = dict(single_group_variants=5000,single_output_groups=1,single_mac_cutoff=0)
        output = self.root/"stream"
        parts,count = workflow._single_outputs(Stream(),"1",output,identity,{"start":1,"end":100})
        self.assertEqual(count,5)
        self.assertEqual([item["rows"] for item in parts],[2,2,1])
        tables = [rdata.read_rda(item["path"])["results_individual_analysis"] for item in parts]
        self.assertEqual([str(value) for table in tables for value in table.index],
                         [str(value) for value in baseline.row_names])
        self.assertEqual([int(value) for table in tables for value in table["POS"]],[row["POS"] for row in baseline])
        index = json.loads((output/"index.private.json").read_text())
        self.assertEqual(index["final_factor_levels"],baseline.factor_levels)
        self.assertEqual(index["requested_region"],{"start":1,"end":100})
        self.assertEqual(index["scope"],"selected interval")
        self.assertTrue(all(item["native_validation"]["computed_values_exactly_preserved"] for item in parts))


if __name__ == "__main__":
    unittest.main()
