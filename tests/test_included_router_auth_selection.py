import copy
import json
import random
import subprocess
import unittest
from experiments.included_router_auth_selection.run import DEFAULT,compile_routes,decision,leaf,plan,resolve
from experiments.included_router_auth_selection.render import JS


class IncludedRouteTests(unittest.TestCase):
    def test_nested_prefix_and_policy_not_top_only(self):
        a=plan(DEFAULT);b=plan(DEFAULT,"top_only")
        self.assertEqual((a["registered_leaves"],a["resolved_level"],a["matched_path"]),(5,"admin_optional","/api/control/pause"))
        self.assertEqual(a["decision"]["status"],401)
        self.assertTrue(b["decision"]["allowed"])
        self.assertEqual(b["registered_leaves"],2)

    def test_key_configuration_truth_table(self):
        for api in (False,True):
            for admin in (False,True):
                for level in ("normal","admin_optional","admin_force"):
                    for witness in ("none","api","admin","both","other"):
                        p={**DEFAULT,"api_configured":api,"admin_configured":admin,"credential":witness}
                        required="api" if level=="normal" and api else "admin" if level!="normal" and admin else "api" if level=="admin_optional" and api else None
                        expected=not (level=="admin_force" and not admin) and (required is None or witness==required or witness=="both")
                        self.assertEqual(decision(p,level)["allowed"],expected)
        self.assertEqual(decision({**DEFAULT,"admin_configured":False},"admin_force")["status"],403)

    def test_first_full_not_specificity_and_partial(self):
        p={**DEFAULT,"tree":[leaf("/x","admin_force",methods=["GET"]),leaf("/x","admin_optional")],
           "scope":{"path":"/x","method":"POST","kind":"http"}}
        a=plan(p)
        self.assertEqual([x["result"] for x in a["resolution_trace"]],["PARTIAL","FULL"])
        self.assertEqual(a["resolved_level"],"admin_optional")
        p["tree"]=[leaf("/x"),leaf("/x","admin_optional")]
        self.assertEqual(plan(p)["resolved_level"],"normal")
        p["tree"].reverse()
        self.assertEqual(plan(p)["resolved_level"],"admin_optional")

    def test_metadata_enum_witness_and_explicit_match_error(self):
        p=copy.deepcopy(DEFAULT);target=p["tree"][2]["routes"][0]["routes"][0]
        target["typed"]=False
        self.assertEqual(plan(p)["resolved_level"],"normal")
        target["typed"]=True;target["matcher_error"]=True
        out=plan(p)
        self.assertIsNone(out["matched_path"])
        self.assertIn("MATCHER_ERROR",[x["result"] for x in out["resolution_trace"]])

    def test_protocol_and_raw_probe_prefix(self):
        p={**DEFAULT,"scope":{"path":"/api/control/socket","method":"GET","kind":"websocket"}}
        self.assertEqual(plan(p)["matched_path"],"/api/control/socket")
        p["scope"]={**p["scope"],"kind":"http"}
        self.assertIsNone(plan(p)["matched_path"])
        for path in ("/health_generate","/ready","/metrics"):
            self.assertTrue(decision({**DEFAULT,"scope":{"path":path,"method":"GET","kind":"http"},"credential":"none"},"admin_force")["allowed"])
        self.assertFalse(decision({**DEFAULT,"scope":{"path":"/api/health","method":"GET","kind":"http"},"credential":"none"},"normal")["allowed"])

    def test_compiled_snapshot_needs_rebuild_on_metadata_change(self):
        tree=[leaf("/x","admin_optional")];scope={"path":"/x","method":"POST","kind":"http"}
        old=compile_routes(tree);tree[0]["level"]="normal"
        self.assertEqual(resolve(old,scope)[0],"admin_optional")
        self.assertEqual(resolve(compile_routes(tree),scope)[0],"normal")

    def test_invalid_inputs_not_success_shaped_fallbacks(self):
        for change in ({"tree":[]},{"credential":"secret-text"},{"api_configured":1},
                       {"scope":{"path":"/x?query","method":"GET","kind":"http"}}):
            with self.subTest(change=change),self.assertRaises(ValueError):plan({**DEFAULT,**change})

    def test_node_parity_across_seeded_profiles(self):
        rng=random.Random(167);profiles=[copy.deepcopy(DEFAULT)]
        for _ in range(64):
            nodes=[leaf("/x",rng.choice(["normal","admin_optional","admin_force"])) for _ in range(rng.randint(1,4))]
            for x in nodes:
                x["typed"]=rng.choice([True,False]);x["methods"]=[rng.choice(["GET","POST"])]
                x["matcher_error"]=rng.randrange(8)==0
            profiles.append({**DEFAULT,"tree":[{"prefix":"/api","routes":nodes}],
                             "scope":{"path":rng.choice(["/api/x","/apix","/health"]),"method":rng.choice(["GET","POST","OPTIONS"]),"kind":rng.choice(["http","websocket"])},
                             "api_configured":rng.choice([True,False]),"admin_configured":rng.choice([True,False]),
                             "credential":rng.choice(["none","api","admin","both","other"])})
        expected=[[plan(p,s) for s in ("included","top_only")] for p in profiles]
        script=JS+"\nconst ps="+json.dumps(profiles)+";console.log(JSON.stringify(ps.map(p=>['included','top_only'].map(s=>plan(p,s)))));"
        result=subprocess.run(["node"],input=script,check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),expected)
