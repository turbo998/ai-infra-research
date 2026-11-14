import json
import random
import shutil
import subprocess
import unittest
from experiments.cumulative_graph_buffer_selection import run as m
from experiments.cumulative_graph_buffer_selection.render import JS


class CumulativeBufferTests(unittest.TestCase):
    def test_default_terminal_repeat_and_zero_length_padding(self):
        r=m.evaluate(m.DEFAULT)
        self.assertEqual(r["buffer"],[0,3,8,8,8])
        self.assertEqual([b-a for a,b in zip(r["buffer"],r["buffer"][1:])],[3,5,0,0])
        self.assertEqual(m.evaluate(m.DEFAULT,True)["word_misses"],2)
        self.assertEqual(r["logical_int32_buffer_bytes"],20)

    def test_packed_sections_not_flat_repeat(self):
        p={**m.DEFAULT,"layout":"packed","src":[0,3,8,0,4,9],"capacity":10,
           "previous":[0]*10,"override":"sections"}
        self.assertEqual(m.evaluate(p)["buffer"],[0,3,8,8,8,0,4,9,9,9])
        self.assertEqual(m.evaluate({**p,"override":"repeat"})["word_misses"],4)
        self.assertEqual(m.evaluate({**p,"override":None})["status"],"invalid_padding_policy")

    def test_override_priority_and_scalar_bypass(self):
        self.assertEqual(m.evaluate({**m.DEFAULT,"override":"zeros"})["status"],"invalid_padding_policy")
        p={**m.DEFAULT,"layout":"scalar","src":42,"capacity":1,"previous":[9],"override":"prefix_only"}
        self.assertEqual((m.evaluate(p)["rule"],m.evaluate(p)["buffer"]),("scalar_copy",[42]))
        self.assertEqual(m.evaluate({**m.DEFAULT,"key":"position_ids","layout":"generic"})["buffer"],[0,3,8,0,0])

    def test_inplace_full_write_and_partial_callback(self):
        dst=[0,9,12,17,21];identity=id(dst)
        self.assertTrue(all(m.copy_buffer(dst,[0,3,8],"repeat")))
        self.assertEqual((id(dst),dst),(identity,[0,3,8,8,8]))
        r=m.evaluate({**m.DEFAULT,"previous":[0,9,12,17,21],"override":"prefix_only"})
        self.assertFalse(r["full_buffer_written"])
        self.assertEqual(r["status"],"invalid_padding_policy")
        self.assertEqual(m.DEFAULT["previous"],[0,1,2,3,4])

    def test_missing_shared_and_oversize_are_not_success(self):
        with self.assertRaisesRegex(ValueError,"shared"):
            m.evaluate({**m.DEFAULT,"src":None})
        self.assertEqual(m.evaluate({**m.DEFAULT,"src":None,"shared":False})["status"],"captured_constant_retained")
        p={**m.DEFAULT,"src":[0,1,2,3,4,5]}
        self.assertEqual(m.evaluate(p)["status"],"eager_required")
        self.assertEqual(m.evaluate(p)["buffer"],p["previous"])

    def test_invalid_offsets_and_empty_sequence_representation(self):
        for change in ({"src":[]},{"src":[1,2]},{"src":[0,3,2]},{"capacity":True},
                       {"layout":"packed","src":[0,1,2]},{"previous":[0]},{"src":[False,1]}):
            with self.assertRaises(ValueError):m.evaluate({**m.DEFAULT,**change})
        self.assertEqual(m.evaluate({**m.DEFAULT,"src":[0]})["buffer"],[0]*5)

    def test_redundant_zero_paths_have_identical_results(self):
        for n in (64,256,1024,4096):
            a,b=m.paths(n);self.assertEqual(a(),b())

    @unittest.skipUnless(shutil.which("node"),"Node required for original H5metadata parity")
    def test_random_sections_and_node_parity(self):
        rng=random.Random(160);cases=[m.DEFAULT]
        for _ in range(64):
            offsets=[0]
            for _ in range(rng.randrange(1,6)):offsets.append(offsets[-1]+rng.randrange(6))
            cap=len(offsets)+rng.randrange(1,5)
            p={**m.DEFAULT,"src":offsets,"capacity":cap,"previous":[99]*cap}
            r=m.evaluate(p)
            self.assertEqual(r["buffer"][:len(offsets)],offsets)
            self.assertTrue(all(v==offsets[-1] for v in r["buffer"][len(offsets):]))
            cases.append(p)
        cases.append({**m.DEFAULT,"layout":"packed","src":[0,3,8,0,4,9],"capacity":10,
                      "previous":[0]*10,"override":"sections"})
        script=JS+"\nconst p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(p.map(v=>[evaluate(v,true),evaluate(v)])));"
        result=subprocess.run(["node","-e",script],input=json.dumps(cases),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[m.evaluate(v,True),m.evaluate(v)] for v in cases])


if __name__=="__main__":unittest.main()
