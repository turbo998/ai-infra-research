import copy
import json
import random
import subprocess
import unittest
from experiments.image_metadata_placeholder_selection.run import DEFAULT,geometry,geometry_rows,plan
from experiments.image_metadata_placeholder_selection.render import JS


class ImageMetadataTests(unittest.TestCase):
    def test_ordered_original_dimensions_and_flat_ranges(self):
        out=plan(DEFAULT)
        self.assertEqual([x["tokens"] for x in out["resize_rows"]],[15,1,15])
        self.assertEqual([x["original_size"] for x in out["headers"]],DEFAULT["sizes"])
        self.assertEqual(out["planned_embedding_ranges"],[[0,15],[15,16],[16,31]])
        self.assertEqual(out["mode"],"metadata_only_deferred")
        self.assertIsNone(out["native_embedding_execution"])

    def test_missing_values_gate_and_flat_row_sum(self):
        with self.assertRaisesRegex(ValueError,"required"):
            plan({**DEFAULT,"allow_missing":False})
        self.assertEqual(plan({**DEFAULT,"allow_missing":False,"embedding_shape":[31,2]})["mode"],"declared_embedding_rows")
        with self.assertRaisesRegex(ValueError,"sum"):
            plan({**DEFAULT,"embedding_shape":[30,2]})
        largest={**DEFAULT,"sizes":[[8192,8192]]*64,"patch":128,"merge":1,
                 "side_patches":64,"patch_budget":4096,"embedding_shape":[262144,2]}
        self.assertEqual(plan(largest)["total_pad_tokens"],262144)

    def test_padding_limits_and_fixed_count(self):
        self.assertEqual(geometry(8192,1,DEFAULT),{"resized":[128,1],"padded":[128,16],"tokens":8})
        self.assertEqual(geometry(9,7,{**DEFAULT,"patch_budget":1})["tokens"],1)
        self.assertEqual(geometry(9,7,{**DEFAULT,"patch_budget":1})["padded"],[16,16])
        self.assertEqual(plan({**DEFAULT,"fixed_tokens":4})["total_pad_tokens"],12)

    def test_explicit_counterfactuals_do_not_share_the_correct_answer(self):
        self.assertEqual(plan(DEFAULT,proposal="padded_headers")["placeholder_misses"],3)
        collapsed=plan(DEFAULT,proposal="collapse_occurrences")
        self.assertEqual((collapsed["retained_occurrences"],collapsed["retained_pad_tokens"]),(2,16))
        self.assertEqual(collapsed["placeholder_misses"],1)

    def test_geometry_memo_keeps_occurrences_and_configuration_scope(self):
        p=copy.deepcopy(DEFAULT);p["sizes"]=[[65,33]]*64
        self.assertEqual(plan(p,True),plan(p,False))
        self.assertEqual(plan(p)["retained_occurrences"],64)
        changed={**DEFAULT,"patch":16}
        self.assertNotEqual(plan(DEFAULT)["total_pad_tokens"],plan(changed)["total_pad_tokens"])
        self.assertEqual(plan(changed,True),plan(changed,False))

    def test_reject_invalid_metadata_without_coercion(self):
        for change in ({"sizes":[[True,4]]},{"sizes":[[1.5,4]]},{"sizes":[]},
                       {"sizes":[[0,4]]},{"merge":3},{"allow_missing":1},
                       {"fixed_tokens":0},{"embedding_shape":[31,0]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                plan({**DEFAULT,**change})

    def test_budget_is_explicit_descriptor_not_network_or_tokenizer(self):
        out=plan(DEFAULT)
        packet=[{"image_sizes":s} for s in DEFAULT["sizes"]]
        self.assertEqual(out["compact_descriptor_bytes"],len(json.dumps(packet,separators=(",",":")).encode()))
        self.assertEqual(out["declared_uncompressed_rgb_bytes"],13059)
        self.assertIsNone(out["native_transport_bytes"])
        self.assertIsNone(out["native_tokenizer_token_ids"])

    def test_node_parity_and_random_memo(self):
        rng=random.Random(166);profiles=[copy.deepcopy(DEFAULT)]
        for i in range(64):
            p={**DEFAULT,"sizes":[[rng.randint(1,8192),rng.randint(1,8192)] for _ in range(rng.randint(1,6))],
               "patch":rng.choice([4,8,16]),"merge":rng.choice([1,2,4]),"patch_budget":rng.randint(1,4096),
               "side_patches":rng.choice([8,16,32,64]),"fixed_tokens":rng.choice([None,1,7,64])}
            if i%2:p["embedding_shape"]=[plan(p)["total_pad_tokens"],2]
            profiles.append(p)
        expected=[]
        for p in profiles:
            self.assertEqual(geometry_rows(p["sizes"],p,True),geometry_rows(p["sizes"],p,False))
            expected.append([plan(p,True,mode) for mode in ("original","padded_headers","collapse_occurrences")])
        script=JS+"\nconst ps="+json.dumps(profiles)+";console.log(JSON.stringify(ps.map(p=>['original','padded_headers','collapse_occurrences'].map(m=>plan(p,true,m)))));"
        result=subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),expected)
