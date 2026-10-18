import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.structured_decoding_xgrammar import run as xg_mod


class TestStructuredDecodingXgrammar(unittest.TestCase):
    def test_compact_int32_bitmask_achieves_32x_compression(self) -> None:
        prof = xg_mod.compute_mask_memory_profile(vocab_size=128256, batch_size=64)
        self.assertEqual(prof["bitmask_int32_words_per_row"], 4008)
        self.assertEqual(prof["bitmask_bytes_per_row"], 16032)
        self.assertEqual(prof["fp32_bytes_per_row"], 513024)
        self.assertEqual(prof["compression_ratio"], 32.0)

    def test_async_xgrammar_keeps_tpot_overhead_tiny(self) -> None:
        res32 = xg_mod.simulate_structured_decode_step(batch_size=32, backend="xgrammar_async_overlap")
        res64 = xg_mod.simulate_structured_decode_step(batch_size=64, backend="xgrammar_async_overlap")
        naive64 = xg_mod.simulate_structured_decode_step(batch_size=64, backend="naive_dynamic_fsm_fp32")
        self.assertLess(res32["exposed_overhead_us"], 40.0)
        self.assertLess(res32["tpot_degradation_pct"], 2.0)
        self.assertGreater(res64["batch_tokens_per_sec"] / naive64["batch_tokens_per_sec"], 3.5)

    def test_cli_check_passes(self) -> None:
        script = Path(xg_mod.__file__).resolve()
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        data = json.loads(xg_mod.RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "structured_decoding_xgrammar")


if __name__ == "__main__":
    unittest.main()
