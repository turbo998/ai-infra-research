from itertools import combinations
import json
import statistics
import subprocess
import unittest

from experiments.erasure_stripe_contract_selection import run as m
from experiments.erasure_stripe_contract_selection.render import JS


class ErasureStripeTests(unittest.TestCase):
    def test_matrix_encoder_matches_independent_lagrange(self):
        for k,parity in ((1,1),(2,2),(3,2),(4,2)):
            for length in (0,1,7,17):
                data=bytes((i*37+19)%256 for i in range(length))
                shards=m.encode(data,k,parity)
                self.assertEqual(shards,m.independent_lagrange_encode(data,k,parity))
                self.assertEqual(m.generator(k,parity)[:k],[[int(i==j) for j in range(k)] for i in range(k)])

    def test_all_two_erasure_survivor_matrices(self):
        data=bytes(range(1,11));shards=m.encode(data,3,2)
        for missing in combinations(range(5),2):
            survivors={i:s for i,s in enumerate(shards) if i not in missing}
            self.assertEqual(m.recover(survivors,len(data),3,2),data)

    def test_field_inverse_and_nontrivial_byte_recovery(self):
        for value in (1,2,7,128,255):
            self.assertEqual(m.mul(value,m.inverse(value)),1)
        data=bytes(range(256));shards=m.encode(data,4,3)
        for chosen in combinations(range(7),4):
            self.assertEqual(m.recover({i:shards[i] for i in chosen},len(data),4,3),data)

    def test_insufficient_survivors_bounds_and_original_length(self):
        data=bytes(range(10));shards=m.encode(data,3,2)
        with self.assertRaisesRegex(ValueError,"at least k"):
            m.recover({0:shards[0],1:shards[1]},len(data),3,2)
        with self.assertRaises(ValueError):
            m.encode([256],3,2)
        with self.assertRaises(ValueError):
            m.recover({0:shards[0],1:shards[1],5:shards[2]},len(data),3,2)
        self.assertEqual(len(m.recover({i:s for i,s in enumerate(shards)},len(data),3,2)),10)

    def test_domains_count_lost_chunks_not_failed_hosts(self):
        bad=["A","A","A","B","C"];good=["A","B","C","D","E"]
        self.assertEqual(m.lost_in_domains(bad,["A"]),[0,1,2])
        self.assertEqual(m.lost_in_domains(good,["A"]),[0])

    def test_corruption_is_not_erasured_by_this_decoder(self):
        data=bytes(range(10));shards=m.encode(data,3,2)
        survivors={i:list(shards[i]) for i in (0,1,2)}
        survivors[0][0]^=1
        self.assertNotEqual(m.recover(survivors,len(data),3,2),data)
        with self.assertRaises(ValueError):
            m.budget({**m.DEFAULT,"object_bytes":2**53})

    def test_node_codec_padding_and_budget_parity(self):
        data=list(range(1,11))
        for missing in ([0,1],[0,4],[3,4]):
            source=JS+"\nconsole.log(JSON.stringify({result:simulate("+json.dumps(data)+",3,2,"+json.dumps(missing)+"),budget:budget("+json.dumps(m.DEFAULT)+")}))"
            out=json.loads(subprocess.check_output(["node","-e",source],text=True))
            self.assertEqual(out["result"]["encoded_chunks"],m.encode(data,3,2))
            self.assertEqual(out["result"]["recovered_bytes"],data)
            self.assertEqual(out["budget"],m.budget(m.DEFAULT))

    def test_sealed_report_and_timing_paths(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency))
        for row in latency["rows"]:
            for prefix in ("lookup","bitwise"):
                samples=row[prefix+"_samples_ms"]
                self.assertEqual(len(samples),7)
                self.assertEqual(row[prefix+"_median_ms"],statistics.median(samples))
        a,b=m.timing_paths(64);self.assertEqual(a(),b())
        self.assertIsNone(report["durability_probability"])


if __name__=="__main__":
    unittest.main()
