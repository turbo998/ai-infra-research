import copy
import json
import random
import subprocess
import unittest
from experiments.hicache_page_rotation_selection.run import DEFAULT, buckets, counts, plan, validate
from experiments.hicache_page_rotation_selection.render import JS


class HiCachePageRotationTests(unittest.TestCase):
    def test_chain_and_counterexample(self):
        a=plan(DEFAULT);b=plan(DEFAULT,True)
        self.assertEqual(a["owners"],[3,0,1,2,3,0,1])
        self.assertEqual((a["owner_misses"],b["owner_misses"]),(0,7))
        self.assertEqual(a["owned_local_host_pages"],[1,2])
        self.assertEqual((a["recovered_rotation_base"],a["published_rotation_base"]),(1,1))
        self.assertFalse(a["actual_legacy_startup_admitted"])

    def test_registry_both_entry_points(self):
        r=plan({**DEFAULT,"registered":[]})
        self.assertFalse(r["startup_admitted"]);self.assertFalse(r["runtime_attachment_admitted"])
        p={**DEFAULT,"shards":1,"rank":0,"free_pages":[8],"base_at_alloc":0,"base_at_admit":0,"registered":[]}
        self.assertEqual(plan(p)["action"],"publish_rotation")
        self.assertEqual(plan({**DEFAULT,"hicache":False})["action"],"not_initialized")

    def test_staged_revalidation_and_cover_precedence(self):
        r=plan({**DEFAULT,"base_at_admit":2})
        self.assertEqual(r["action"],"refetch_rotation");self.assertIsNone(r["published_rotation_base"])
        self.assertEqual(plan({**DEFAULT,"found_at_admit":0,"base_at_admit":None})["published_rotation_base"],1)
        self.assertEqual(plan({**DEFAULT,"joint_tokens":144,"base_at_admit":2})["action"],"device_covered")

    def test_new_chain_and_least_full_tie(self):
        p={**DEFAULT,"prefix_tokens":0,"found_at_alloc":0,"base_at_alloc":None,"found_at_admit":0,"base_at_admit":None,
           "free_pages":[5,8,8,6]}
        self.assertEqual(plan(p)["raw_start_owner"],1)
        self.assertEqual(plan(p)["published_rotation_base"],1)
        self.assertEqual(plan({**p,"prefix_tokens":64})["raw_start_owner"],5)

    def test_empty_and_capacity_attempt(self):
        r=plan({**DEFAULT,"fetch_pages":0})
        self.assertIsNone(r["recovered_rotation_base"]);self.assertEqual(r["owners"],[])
        r=plan({**DEFAULT,"free_pages":[0,0,0,0]})
        self.assertEqual(r["action"],"attempt_capacity_miss");self.assertEqual(r["owners"],[])

    def test_random_closed_form_and_local_partition(self):
        rng=random.Random(162)
        for _ in range(64):
            n=rng.randrange(1,17);start=rng.randrange(128);size=rng.randrange(129)
            owners=[(start+j)%n for j in range(size)]
            self.assertEqual(counts(start,size,n),[owners.count(c) for c in range(n)])
            pages=list(range(n+start,n+start+size))
            out=buckets(pages,n)
            self.assertEqual(sum(map(len,out)),size)
            for c,vals in enumerate(out):
                self.assertEqual(vals,[h//n for h in pages if h%n==c])

    def test_invalid_contract(self):
        for k,v in (("page_size",0),("shards",0),("rank",4),("prefix_tokens",1),
                    ("registered",["a","a"]),("free_pages",[1]),("fetch_pages",129),("base_at_alloc",4)):
            with self.assertRaises(ValueError):validate({**DEFAULT,k:v})
        with self.assertRaises(ValueError):validate({**DEFAULT,"surprise":1})
        with self.assertRaises(ValueError):validate({**DEFAULT,"found_at_admit":160})

    def test_node_parity(self):
        rng=random.Random(162)
        cases=[copy.deepcopy(DEFAULT)]
        for _ in range(64):
            n=rng.randrange(1,9);page=16;prefix=rng.randrange(8)*page;fetch=rng.randrange(9);base=rng.randrange(n)
            cases.append({**DEFAULT,"shards":n,"rank":rng.randrange(n),"prefix_tokens":prefix,"fetch_pages":fetch,
                "found_at_alloc":prefix,"base_at_alloc":base,"found_at_admit":prefix,
                "base_at_admit":rng.choice([base,rng.randrange(n),None]),
                "joint_tokens":rng.choice([0,prefix+fetch*page]),"free_pages":[rng.randrange(10) for _ in range(n)],
                "registered":rng.choice([[],["rank-local-demo"]]),"hicache":rng.choice([False,True])})
        script=JS+'const fs=require("fs");const ps=JSON.parse(fs.readFileSync(0,"utf8"));console.log(JSON.stringify(ps.map(p=>[plan(p),plan(p,true)])));'
        result=subprocess.run(["node","-e",script],input=json.dumps(cases),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),[[plan(p),plan(p,True)] for p in cases])


if __name__ == "__main__":
    unittest.main()
