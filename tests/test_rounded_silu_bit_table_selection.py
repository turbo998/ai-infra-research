import json
import math
import random
import shutil
import subprocess
import unittest

from experiments.rounded_silu_bit_table_selection import run as m
from experiments.rounded_silu_bit_table_selection.render import JS


class RoundedSiluTests(unittest.TestCase):
    def test_intermediate_rounding_is_not_final_rounding(self):
        r = m.lane(0x0401, 0x3E00, "float16")
        self.assertEqual((r["activation_word"], r["rounded_output_word"], r["single_round_output_word"]),
                         (0x0201, 0x0302, 0x0301))
        self.assertEqual(m.direct_output(0x0401, 0x3E00, "float16"), 0x0302)

    def test_signed_zero_keys_and_missing_sparse_word(self):
        table = m.sparse_table([0, 0x8000], "float16")
        self.assertEqual(table, {0: 0, 0x8000: 0x8000})
        self.assertEqual(m.lookup(0x8000, 0x3E00, "float16", table), 0x8000)
        lossy = {m.decode(k, "float16"): v for k, v in table.items()}
        self.assertEqual(len(lossy), 1)
        self.assertNotEqual(lossy[0.0], table[0])
        with self.assertRaisesRegex(ValueError, "absent"):
            m.lookup(1, 0x3E00, "float16", table)

    def test_ieee_rounding_boundaries_and_exp_overflow_scope(self):
        self.assertEqual(m.encode(2**-25, "float16"), 0)
        self.assertEqual(m.encode(3 * 2**-25, "float16"), 2)
        self.assertEqual(m.encode(1 + 2**-8, "bfloat16"), 0x3F80)
        self.assertEqual(m.encode(1 + 3 * 2**-8, "bfloat16"), 0x3F82)
        self.assertEqual(m.encode(65520, "float16"), 0x7C00)
        self.assertEqual(m.encode(m.activation(-89), "bfloat16"), 0x8000)
        stable = -89 * math.exp(-89) / (1 + math.exp(-89))
        self.assertEqual(m.encode(stable, "bfloat16"), 0x8287)
        self.assertNotEqual(m.encode(stable, "bfloat16"), m.encode(m.activation(-89), "bfloat16"))

    def test_cache_phase_exact_key_and_nonmutation(self):
        cold = m.route(m.DEFAULT)
        self.assertEqual((cold["action"], cold["new_logical_table_bytes"]), ("build", 131072))
        capture = {**m.DEFAULT, "capturing": True}
        self.assertEqual(m.route(capture)["action"], "fallback")
        self.assertEqual(m.route({**capture, "cached_keys": [["float16", 0]]})["action"], "reuse")
        for key in (["bfloat16", 0], ["float16", 1]):
            self.assertEqual(m.route({**capture, "cached_keys": [key]})["action"], "fallback")
        self.assertEqual(m.DEFAULT["cached_keys"], [])
        self.assertEqual(m.route({**m.DEFAULT, "tokens": 0})["new_logical_table_bytes"], 131072)
        self.assertFalse(m.route({**m.DEFAULT, "tokens": 0})["activation_launch"])
        self.assertEqual(m.route({**m.DEFAULT, "tokens": 0})["shared_bytes_per_table_cta"], 0)

    def test_uncached_dispatch_gates_and_cached_order(self):
        for changes in ({"hip": True}, {"major": 9}, {"major": 12}, {"shared_optin_bytes": 131071},
                        {"cuda": False}, {"dtype": "float32"}, {"clamp": True}, {"operation": "gelu"}):
            self.assertEqual(m.route({**m.DEFAULT, **changes})["action"], "fallback")
        # The source cache lookup precedes hardware/capture probes, not eligibility.
        self.assertEqual(m.route({**m.DEFAULT, "major": 9, "capturing": True,
                                  "cached_keys": [["float16", 0]]})["action"], "reuse")
        self.assertEqual(m.route({**m.DEFAULT, "cuda": False,
                                  "cached_keys": [["float16", 0]]})["action"], "fallback")

    def test_explicit_invalid_inputs(self):
        for gate in (True, -1, 65536, 0x7C00, 0x7E00):
            with self.assertRaises(ValueError):
                m.lane(gate, 0x3E00, "float16")
        for changes in ({"tokens": True}, {"device": -1}, {"clamp": 1},
                        {"cached_keys": [["float16", 0], ["float16", 0]]},
                        {"cached_keys": [["float32", 0]]}, {"dtype": "fp16"}):
            with self.assertRaises(ValueError):
                m.route({**m.DEFAULT, **changes})
        with self.assertRaises(ValueError):
            m.route({**m.DEFAULT, "extra": True})

    def test_random_finite_lookup_matches_explicit_reference(self):
        rng = random.Random(158)
        for dtype in m.DTYPES:
            gates = [m.encode(rng.uniform(-20, 20), dtype) for _ in range(64)]
            ups = [m.encode(rng.uniform(-3, 3), dtype) for _ in gates]
            table = m.sparse_table(gates, dtype)
            for g, u in zip(gates, ups):
                a = m.encode(m.activation(m.decode(g, dtype)), dtype)
                expected = m.encode(m.f32(m.decode(a, dtype) * m.decode(u, dtype)), dtype)
                self.assertEqual(m.lookup(g, u, dtype, table), expected)
                self.assertEqual(m.direct_output(g, u, dtype), expected)
        for n in (64, 256, 1024, 4096):
            a, b = m.paths(n)
            self.assertEqual(a(), b())

    @unittest.skipUnless(shutil.which("node"), "Node required for original H5 arithmetic parity")
    def test_javascript_word_and_dispatch_parity(self):
        rng = random.Random(1580)
        cases = []
        for dtype in m.DTYPES:
            cases += [[g, u, dtype] for g, u in m.pairs(dtype)]
            cases += [[m.encode(rng.uniform(-20, 20), dtype), m.encode(rng.uniform(-3, 3), dtype), dtype]
                      for _ in range(64)]
        routes = [{**m.DEFAULT, **v} for v in ({}, {"capturing": True}, {"tokens": 0}, {"major": 9},
                  {"shared_optin_bytes": 131071}, {"hip": True}, {"dtype": "float32"}, {"clamp": True},
                  {"capturing": True, "cached_keys": [["float16", 0]]},
                  {"capturing": True, "cached_keys": [["bfloat16", 0]]},
                  {"capturing": True, "cached_keys": [["float16", 1]]})]
        payload = {"cases": cases, "routes": routes}
        script = JS + "\nconst p=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify({lanes:p.cases.map(v=>lane(...v)),routes:p.routes.map(route)}));"
        result = subprocess.run(["node", "-e", script], input=json.dumps(payload), text=True,
                                capture_output=True, check=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["lanes"], [m.lane(*v) for v in cases])
        self.assertEqual(observed["routes"], [m.route(v) for v in routes])


if __name__ == "__main__":
    unittest.main()
