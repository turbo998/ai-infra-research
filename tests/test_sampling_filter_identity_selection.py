from fractions import Fraction
import itertools
import json
import subprocess
import unittest

from experiments.sampling_filter_identity_selection import run,render


class SamplingContractTests(unittest.TestCase):
    def test_composition_changes_conditional_mass(self):
        first=run.filtered([4,3,2,1],2,Fraction(1,2),"top_k_first")
        joint=run.filtered([4,3,2,1],2,Fraction(1,2),"joint")
        self.assertEqual(first,(1,0,0,0))
        self.assertEqual(joint,(Fraction(4,7),Fraction(3,7),0,0))
        self.assertEqual(run.inverse_cdf(first,Fraction(3,5)),0)
        self.assertEqual(run.inverse_cdf(joint,Fraction(3,5)),1)

    def test_nucleus_minimal_prefix_matches_independent_subset_reference(self):
        for weights in itertools.product(range(1,4),repeat=3):
            ranking=sorted(range(3),key=lambda i:(-weights[i],i))
            for k,p in itertools.product((1,2,3),(Fraction(1,4),Fraction(1,2),Fraction(1))):
                top=ranking[:k]
                mass=sum(weights[i] for i in top)
                prefixes=[top[:n] for n in range(1,len(top)+1)]
                reference=next(ids for ids in prefixes if sum(weights[i] for i in ids)>=p*mass)
                actual=run.filtered(weights,k,p,"top_k_first")
                self.assertEqual({i for i,v in enumerate(actual) if v},set(reference))
                self.assertEqual(sum(actual),1)

    def test_exact_cdf_intervals_and_zero_mass_tokens(self):
        weights=[0,2,0,3]
        self.assertEqual(run.inverse_cdf(weights,0),1)
        self.assertEqual(run.inverse_cdf(weights,Fraction(2,5)),3)
        counts=[0]*4
        for numerator in range(50):
            counts[run.inverse_cdf(weights,Fraction(2*numerator+1,100))]+=1
        self.assertEqual(counts,[0,20,0,30])

    def test_tie_policy_scale_and_threshold_endpoints(self):
        self.assertEqual(run.filtered([3,3,1],1,1,"top_k_first"),(1,0,0))
        for order in ("joint","top_k_first"):
            self.assertEqual(run.filtered([4,3,2,1],4,1,order),run.normalize([4,3,2,1]))
            self.assertEqual(run.filtered([4,3,2,1],2,Fraction(1,2),order),
                             run.filtered([40,30,20,10],2,Fraction(1,2),order))

    def test_request_uniform_identity_and_bad_bound(self):
        report=run.build_report({"rows":[]})
        self.assertEqual(report["identity"]["request_key_matches"],2)
        self.assertEqual(report["identity"]["slot_key_matches"],0)
        self.assertNotEqual(report["bound_control"]["requested_distribution"],
                            report["bound_control"]["clipped_distribution"])

    def test_original_invalid_inputs_never_return_success_tokens(self):
        for weights in ([],[0,0],[-1,2],[True,2],[float("nan"),1]):
            with self.assertRaises(ValueError):
                run.filtered(weights,1,Fraction(1,2),"joint")
        for k,p,order in ((0,1,"joint"),(5,1,"joint"),(2,0,"joint"),(2,2,"joint"),(2,1,"unknown")):
            with self.assertRaises(ValueError):
                run.filtered([4,3,2,1],k,p,order)
        for uniform in (-1,1,Fraction(11,10)):
            with self.assertRaises(ValueError):
                run.inverse_cdf([1,1],uniform)

    def test_bigint_workbench_and_byte_budget_parity(self):
        cases=[([4,3,2,1],2,1,2,3,5),([0,2,0,3],4,1,1,2,5),([3,3,1],1,1,1,0,1)]
        expected=[]
        for weights,k,pn,pd,un,ud in cases:
            row={}
            for order in ("top_k_first","joint"):
                distribution=run.filtered(weights,k,Fraction(pn,pd),order)
                row[order]={"distribution":run.render_fractions(distribution),
                            "token":run.inverse_cdf(distribution,Fraction(un,ud)),
                            "support":sum(v>0 for v in distribution)}
            expected.append(row)
        script="const DEFAULT="+json.dumps(run.DEFAULT)+";\n"+render.JS
        script+="console.log(JSON.stringify("+json.dumps(cases)+".map(v=>calculate(...v))));"
        script+="console.log(JSON.stringify(estimate(DEFAULT)));"
        script+="for(const patch of [{unique_rows:0},{index_bytes:2},{vocabulary:2**53-1}]){let bad=false;try{estimate({...DEFAULT,...patch})}catch(e){bad=true}if(!bad)throw Error('invalid accepted')}"
        result=subprocess.run(["node","-e",script],check=True,capture_output=True,text=True)
        lines=result.stdout.splitlines()
        self.assertEqual(json.loads(lines[0]),expected)
        self.assertEqual(json.loads(lines[1]),run.estimate_budget(run.DEFAULT))
        for patch in ({"unique_rows":0},{"index_bytes":2},{"vocabulary":2**53-1}):
            with self.assertRaises(ValueError):
                run.estimate_budget({**run.DEFAULT,**patch})

    def test_integer_cdf_search_and_sealed_report(self):
        cdf=list(itertools.accumulate([0,1,0,2,3]))
        for draw in range(cdf[-1]):
            import bisect
            self.assertEqual(run.linear_lookup(cdf,draw),bisect.bisect_right(cdf,draw))
        report=json.loads((run.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,run.build_report(report["latency"]))
        cards=render.cards(report)
        self.assertEqual(cards.count('<section class="card">'),7)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'",cards)
        self.assertNotIn("<script",cards)
        self.assertEqual(cards.count("<table>"),cards.count("</table>"))


if __name__=="__main__":
    unittest.main()
