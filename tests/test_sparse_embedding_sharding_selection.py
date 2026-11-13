from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.sparse_embedding_sharding_selection.run import (
    BAGS, TABLE, Budget, compare, dense_pool, estimate, global_root, load_latency,
    outputs, sharded_pool, simulate, table_gradient,
)


class SparseEmbeddingShardingTests(unittest.TestCase):
    def test_table_row_column_pooling_matches_dense_for_sum_mean_weighted_sum(self):
        weights=[[(j+1)/3+i/10 for j in range(len(bag))] for i,bag in enumerate(BAGS)]
        for mode,values in (("sum",None),("mean",None),("sum",weights)):
            expected=dense_pool(TABLE,BAGS,mode,values)
            for axis in ("table","row","column"):
                actual=sharded_pool(TABLE,BAGS,mode,values,axis=axis)
                self.assertEqual(len(actual),len(BAGS))
                self.assertEqual(actual[0],[0.0]*6)
                for row,reference in zip(actual,expected):
                    for value,target in zip(row,reference):
                        self.assertAlmostEqual(value,target,places=12)

    def test_all_table_gradient_coordinates_against_sharded_finite_differences(self):
        upstream=[[(i+1)*(c+1)%5/7-0.2 for c in range(6)] for i in range(len(BAGS))]
        weights=[[(j+1)/3+i/10 for j in range(len(bag))] for i,bag in enumerate(BAGS)]
        def objective(table,mode,values):
            output=sharded_pool(table,BAGS,mode,values)
            return sum(a*b for row,u in zip(output,upstream) for a,b in zip(row,u))
        epsilon=1e-5
        for mode,values in (("sum",None),("mean",None),("sum",weights)):
            gradient=table_gradient(TABLE,BAGS,upstream,mode,values)
            for i in range(8):
                for c in range(6):
                    plus,minus=[row[:] for row in TABLE],[row[:] for row in TABLE]
                    plus[i][c]+=epsilon
                    minus[i][c]-=epsilon
                    numerical=(objective(plus,mode,values)-objective(minus,mode,values))/(2*epsilon)
                    self.assertAlmostEqual(gradient[i][c],numerical,places=8)

    def test_local_means_dedup_column_order_and_explicit_epsilon_policy(self):
        report=simulate()
        cases={row["id"]:row for row in report["cases"]}
        for axis in ("table","row","column"):
            self.assertTrue(cases[axis+"-ideal-mean"]["matches_ideal"])
        for name in ("sum-of-local-means","dedup-without-multiplicity","reversed-column-pieces"):
            self.assertGreater(cases[name]["maximum_absolute_error"],1)
        epsilon=cases["global-count-plus-epsilon"]
        self.assertFalse(epsilon["matches_ideal"])
        self.assertIn("policy",epsilon["interpretation"])
        for row,bag,ideal in zip(epsilon["output"],BAGS,dense_pool(TABLE,BAGS,"mean")):
            for value,reference in zip(row,ideal):
                expected=reference*len(bag)/(len(bag)+1e-6) if bag else 0
                self.assertAlmostEqual(value,expected,places=12)
        self.assertTrue(all(row["matches_ideal"] for row in report["perturbations"]))

    def test_subgroup_root_is_global_member_not_literal_zero(self):
        self.assertEqual(global_root([1,3,5,7],0),1)
        self.assertEqual(global_root([1,3,5,7],3),7)
        self.assertEqual(global_root([0,2,4],0),0)
        for members,root in (([1,3],2),([1,1],0),([],0)):
            with self.assertRaises(ValueError):
                global_root(members,root)

    def test_storage_and_logical_delivery_budgets_are_distinct(self):
        result=estimate(Budget())
        self.assertEqual(result["table_wise_peak_weight_bytes"],25600000000)
        self.assertEqual(result["row_wise_peak_weight_bytes"],3200000000)
        self.assertEqual(result["column_wise_logical_id_delivery_bytes"],8*result["row_wise_logical_id_delivery_bytes"])
        self.assertEqual(result["row_wise_dense_partial_bytes_all_ranks"],8*result["column_wise_output_piece_bytes_all_ranks"])
        uneven=estimate(replace(Budget(),vocabulary=11,dimension=7,shards=3))
        self.assertEqual(uneven["row_wise_peak_weight_bytes"],4*7*2)
        self.assertEqual(uneven["column_wise_peak_weight_bytes"],11*3*2)

    def test_invalid_inputs_fail_instead_of_silent_drops(self):
        with self.assertRaises(ValueError):
            dense_pool(TABLE,[[8]])
        with self.assertRaises(ValueError):
            dense_pool(TABLE,[[0]],"mean",[[1.0]])
        with self.assertRaises(ValueError):
            dense_pool(TABLE,[[0]],"sum",[[math.nan]])
        with self.assertRaises(ValueError):
            compare("shape",[[1]],[[1,2]])
        for profile in (replace(Budget(),shards=129),replace(Budget(),vocabulary=2**32,id_bytes=4)):
            with self.assertRaises(ValueError):
                estimate(profile)

    def test_sealed_cpu_samples_generate_true_medians(self):
        data=load_latency()
        self.assertEqual(data["provenance"],"measured")
        for row in data["rows"]:
            for path in ("dense","row_sharded"):
                samples=row["samples_ns"][path]
                self.assertEqual(row[path+"_median_ms"],sorted(samples)[3]/1e6)
                self.assertEqual(len(samples),7)
                self.assertTrue(all(n>0 for n in samples))

    def test_artifacts_csp_seven_cards_and_geometry(self):
        for path,content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'),7)
                self.assertNotIn("<script",content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'",content)
            if path.suffix==".svg":
                root=ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"],"0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    for point in polygon.attrib["points"].split():
                        self.assertTrue(200<=float(point.split(",")[1])<=620)

    @unittest.skipUnless(shutil.which("node"),"Node.js optional")
    def test_javascript_budget_parity(self):
        from experiments.sparse_embedding_sharding_selection.render import CALCULATOR
        profiles=[Budget(),replace(Budget(),vocabulary=11,dimension=7,shards=3)]
        source=CALCULATOR.split("const form=")[0]
        script=source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        result=json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,actual in zip(profiles,result):
            for key,value in actual.items():
                self.assertEqual(value,estimate(profile)[key])


if __name__=="__main__":
    unittest.main()
