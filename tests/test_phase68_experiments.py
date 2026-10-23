"""Unit tests for Phase 68 (priority: 77) deterministic experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, module_name: str):
    full_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase68Experiments(unittest.TestCase):
    def test_track_a_mhc_tailpipe_pdl_rmsnorm_tilelang_fp4(self) -> None:
        mod = _load_module(
            "experiments/flashinfer-mhc-tailpipe-pdl-rmsnorm-tilelang-fp4/run.py",
            "phase68_track_a",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 68)
        self.assertEqual(summary["priority"], 77)
        self.assertTrue(all(summary["invariants"].values()))

        # Check R200 (208 SM) dispatch boundary
        self.assertEqual(mod.mhc_infer_tail_program(256, 208)["regime"], "bf16")
        self.assertEqual(mod.mhc_infer_tail_program(385, 208)["regime"], "tailpipe")
        self.assertEqual(mod.mhc_infer_tail_program(257, 148)["regime"], "tailpipe")

        # Check PDL contract
        pdl = summary["pdl_rmsnorm_contract"]
        self.assertEqual(pdl["total_test_cases"], 242)
        self.assertEqual(pdl["wait_removed_2d_failures"], 56)
        self.assertEqual(pdl["with_griddepcontrol_wait_failures"], 0)

        # Check Thor deslop
        thor = summary["sm110_thor_deslop"]
        self.assertEqual(thor["total_units"], 50)
        self.assertEqual(thor["moved_tvm_ffi_check_sites"], 248)
        self.assertEqual(thor["net_lines_delta"], -14131)

        # Check TileLang guards
        tl = summary["tilelang_guards"]
        self.assertEqual(tl["ascend_fp4_mn_align_for_18"], 64)
        self.assertEqual(tl["ascend_fp8_mn_align_for_18"], 32)
        self.assertTrue(tl["shifted_waw_requires_sync"])
        self.assertFalse(tl["disjoint_waw_requires_sync"])

    def test_track_b_sglang_response_template_logits_cosmos3_edge(self) -> None:
        mod = _load_module(
            "experiments/sglang-response-template-logits-trtllm-cosmos3-edge/run.py",
            "phase68_track_b",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 68)
        self.assertEqual(summary["priority"], 77)
        self.assertTrue(all(summary["invariants"].values()))

        # Check invalid response_template rejection
        with self.assertRaises(ValueError):
            mod.validate_sglang_response_template({"fields": {"unknown_field": {}}})
        with self.assertRaises(ValueError):
            mod.validate_sglang_response_template({"fields": {"tool_calls": {"content": "json"}}})

        # Check Cosmos3-Edge weight mapping
        cosmos = summary["cosmos3_edge_and_vllm"]["cosmos3_edge"]
        self.assertEqual(cosmos["drop_patterns_count"], 17)
        self.assertEqual(cosmos["dropped_keys_count"], 5)
        self.assertEqual(cosmos["kept_keys_count"], 8)

    def test_track_c_te_wide_mxfp8_lmcache_addr_nixl_ucx(self) -> None:
        mod = _load_module(
            "experiments/te-wide-mxfp8-requant-lmcache-addr-nixl-ucx-notif/run.py",
            "phase68_track_c",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 68)
        self.assertEqual(summary["priority"], 77)
        self.assertTrue(all(summary["invariants"].values()))

        te = summary["te_wide_mxfp8"]
        self.assertTrue(te["bijective_512b_swizzle"])
        self.assertEqual(te["wide_vs_narrow_tile_reduction_x"], 4)

        lmc = summary["lmcache_address_transfer"]
        self.assertEqual(lmc["five_chunk_pinned_plan"]["num_launches"], 2)
        self.assertEqual(
            [x["skip"] for x in lmc["five_chunk_pinned_plan"]["launches"]],
            [1, 0],
        )

    def test_cli_check_mode_and_summary_files(self) -> None:
        exp_dirs = [
            "experiments/flashinfer-mhc-tailpipe-pdl-rmsnorm-tilelang-fp4",
            "experiments/sglang-response-template-logits-trtllm-cosmos3-edge",
            "experiments/te-wide-mxfp8-requant-lmcache-addr-nixl-ucx-notif",
        ]
        for rel_dir in exp_dirs:
            script = REPO_ROOT / rel_dir / "run.py"
            res = subprocess.run(
                [sys.executable, str(script), "--check"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("OK:", res.stdout)
            summary_path = REPO_ROOT / rel_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["evidence_tier"], "own-offline")
            self.assertEqual(data["phase"], 68)


if __name__ == "__main__":
    unittest.main()
