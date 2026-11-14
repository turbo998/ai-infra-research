import copy
import json
import random
import subprocess
import unittest

from experiments.draft_metadata_layout_selection import run as m
from experiments.draft_metadata_layout_selection.render import JS,cards


class DraftMetadataTests(unittest.TestCase):
    def test_common_layout_keeps_metadata_free_constraints(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual((r["largest_common_kernel"],r["allocated_kernel_used"],r["stride"]),(64,64,4))
        self.assertEqual(r["naive_remove_metadata_free_groups_kernel"],256)
        self.assertFalse(r["naive_kernel_valid_for_all_layout_groups"])

    def test_builders_workspace_and_attention_only_graph_scope(self):
        r=m.evaluate(m.DEFAULT);a,b=r["after"],r["before"]
        self.assertEqual((b["builder_count"],a["builder_count"]),(4,2))
        self.assertEqual(a["metadata_layer_names"],["target"])
        self.assertEqual((b["selected_workspace_bytes"],a["selected_workspace_bytes"]),(4096,0))
        self.assertEqual((b["attention_metadata_cg_support"],a["attention_metadata_cg_support"]),("UNIFORM_BATCH","ALWAYS"))
        self.assertIsNone(r["native_global_cudagraph_readiness"])

    def test_payload_offsets_and_independent_address_oracle(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual(r["original_kv_payload_bytes"],524288)
        self.assertEqual(r["mapped_address"],{"kernel_block_id":15,"position":58})
        self.assertEqual((r["before"]["original_unpadded_table_bytes"],r["after"]["original_unpadded_table_bytes"]),(512,256))
        self.assertEqual(r["after"]["original_offset_bytes"],32)
        for block in range(4):
            for position in range(256):
                p={**m.DEFAULT,"address":{"block":block,"position":position}}
                mapped=m.evaluate(p)["mapped_address"]
                self.assertEqual(mapped["kernel_block_id"]*64+mapped["position"],block*256+position)

    def test_mixed_and_all_skipped_groups(self):
        mixed={**m.DEFAULT,"groups":[{**m.DEFAULT["groups"][1],"name":"mixed","layers":["target","draft"]}]}
        r=m.evaluate(mixed)["after"]
        self.assertEqual(r["builder_count"],2)
        self.assertEqual(r["metadata_layer_names"],["target","draft"])
        empty=m.evaluate({**m.DEFAULT,"no_metadata_layers":["target","draft"]})
        self.assertEqual(empty["after"]["builder_count"],0)
        self.assertEqual(empty["after"]["original_offset_bytes"],32)
        self.assertEqual(empty["original_kv_payload_bytes"],524288)

    def test_owned_zero_workspace_and_existing_smaller_layout(self):
        p=copy.deepcopy(m.DEFAULT);p["no_metadata_layers"]=[];p["groups"][1]["workspace_bytes"]=0
        r=m.evaluate(p)["after"]
        self.assertEqual(r["selected_workspace_owner"],"draft")
        self.assertEqual(r["selected_workspace_bytes"],0)
        p=copy.deepcopy(m.DEFAULT);p["allocated_kernel"]=32;p["groups"][0].update(fixed=[32],multiples=[])
        r=m.evaluate(p)
        self.assertEqual(r["allocated_kernel_used"],32)
        self.assertEqual(m.choose(256,[p["groups"][1]]),64)

    def test_invalid_geometry_skip_and_incompatible_support(self):
        for change in ({"allocated_kernel":128},{"no_metadata_layers":["missing"]},{"ubatches":True},{"address":{"block":4,"position":0}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.evaluate({**m.DEFAULT,**change})
        p=copy.deepcopy(m.DEFAULT);p["groups"][0].update(fixed=[7],multiples=[])
        with self.assertRaises(ValueError):
            m.evaluate(p)

    def test_random_skip_address_node_parity_and_constructor_paths(self):
        rng=random.Random(155);inputs=[m.DEFAULT,{**m.DEFAULT,"allocated_kernel":None}]
        for _ in range(32):
            p=copy.deepcopy(m.DEFAULT);p["no_metadata_layers"]=[v for v in ("target","draft") if rng.randrange(2)]
            p["address"]={"block":rng.randrange(4),"position":rng.randrange(256)}
            p["ubatches"]=rng.randrange(1,5);inputs.append(p)
        script=JS+"\nconsole.log(JSON.stringify("+json.dumps(inputs)+".map(evaluate)));"
        result=subprocess.run(["node"],input=script,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[m.evaluate(p) for p in inputs])
        for n in (16,64,256):
            funcs=m.construction_paths(n);self.assertEqual(funcs[0](),funcs[1]())

    def test_sealed_cases_and_report_contract(self):
        latency=json.loads((m.HERE/"latency.json").read_text());r=json.loads((m.HERE/"report.json").read_text())
        self.assertEqual(r,m.build_report(latency))
        self.assertEqual(len(r["matrix"]),4);self.assertIsNone(r["native_vllm_verified"])
        text=cards(r);self.assertEqual(text.count('class="card"'),7);self.assertNotIn("<script",text)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",text)
