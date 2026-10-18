import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.torch_compile_inductor_cache.run import (
    build_results,
    evaluate_cold_start_modes,
    evaluate_pointwise_fusion_roofline,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "torch_compile_inductor_cache" / "run.py"
RESULTS = ROOT / "experiments" / "torch_compile_inductor_cache" / "results.json"


class TestTorchCompileInductorCache(unittest.TestCase):
    def test_cli_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("torch_compile_inductor_cache: OK", proc.stdout)

    def test_cold_start_modes_eliminate_recompiles_and_startup_wall(self) -> None:
        modes = {r["mode"]: r for r in evaluate_cold_start_modes()}
        stock = modes["stock_torch_compile_per_bucket_unshared"]
        symbolic = modes["vllm_symbolic_no_guards_cold_cache"]
        warm = modes["vllm_fxgraph_triton_warm_cache"]
        aot = modes["vllm_aot_compile_mega_artifact"]
        self.assertGreater(stock["total_startup_min"], 25.0)
        self.assertGreater(stock["runtime_recompilations_on_1000_reqs"], 0)
        self.assertEqual(symbolic["runtime_recompilations_on_1000_reqs"], 0)
        self.assertLess(warm["total_startup_s"], 20.0)
        self.assertGreater(warm["startup_speedup_vs_stock_x"], 85.0)
        self.assertLess(aot["total_startup_s"], warm["total_startup_s"])

    def test_pointwise_fusion_roofline_saves_hbm_traffic(self) -> None:
        fusion = evaluate_pointwise_fusion_roofline(256)
        self.assertGreater(fusion["pointwise_hbm_traffic_reduction_pct"], 50.0)
        self.assertGreater(fusion["end_to_end_throughput_gain_pct"], 25.0)
        self.assertEqual(fusion["eager_pointwise_kernels_80layers"], 1120)
        self.assertEqual(fusion["fused_pointwise_kernels_80layers"], 320)

    def test_stored_summary_invariants(self) -> None:
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(data, build_results())


if __name__ == "__main__":
    unittest.main()
