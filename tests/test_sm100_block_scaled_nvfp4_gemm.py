import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.sm100_block_scaled_nvfp4_gemm.run import (
    BASIC_CHUNK_BYTES,
    BLK_MN,
    BLK_SF,
    build_results,
    sf_k_major_atom_offset,
)


class TestSm100BlockScaledNvfp4Gemm(unittest.TestCase):
    def test_sf_k_major_atom_bijection_and_strides(self) -> None:
        # 128 rows x 4 scale-factor columns must map bijectively onto [0, 511]
        offsets = [sf_k_major_atom_offset(r, k * 16, 16) for r in range(BLK_MN) for k in range(BLK_SF)]
        self.assertEqual(len(offsets), BASIC_CHUNK_BYTES)
        self.assertEqual(len(set(offsets)), BASIC_CHUNK_BYTES)
        self.assertEqual(min(offsets), 0)
        self.assertEqual(max(offsets), 511)
        # Broadcast stride _0 within the 16-element SFVecSize block
        self.assertEqual(sf_k_major_atom_offset(5, 0, 16), sf_k_major_atom_offset(5, 15, 16))
        # Consecutive 4 scale factors of row 0 occupy offsets 0..3 (single 32-bit TMEM word)
        self.assertEqual([sf_k_major_atom_offset(0, k * 16, 16) for k in range(4)], [0, 1, 2, 3])
        # Row 32 starts at offset 4 (Stride<_16, _4> for Shape<_32, _4>)
        self.assertEqual(sf_k_major_atom_offset(32, 0, 16), 4)

    def test_utccp_mma_n_192_padding_and_nvfp4_sqnr(self) -> None:
        res = build_results()
        tile_192 = [
            r
            for r in res["sm100_sf_tile_layouts"]
            if r["mma_tile"] == "128x192x256" and r["format"] == "NVFP4_SF16"
        ][0]
        self.assertEqual(tile_192["mma_n_sfb_utccp"], 256)
        self.assertAlmostEqual(tile_192["sfb_utccp_padding_pct"], 33.33, places=2)

        qf = res["quantization_fidelity"]
        self.assertGreater(qf["nvfp4_vs_mxfp4_mse_reduction_x"], 4.0)
        self.assertGreater(qf["nvfp4_sqnr_gain_over_mxfp4_db"], 6.0)
        self.assertEqual(res["b200_performance"]["cluster_8x_b200_128k_kv_capacity"]["concurrency_multiplier_x"], 2.06)

    def test_cli_check_mode(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "sm100_block_scaled_nvfp4_gemm" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        committed = json.loads((script.parent / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(committed["experiment"], "sm100_block_scaled_nvfp4_gemm")


if __name__ == "__main__":
    unittest.main()
