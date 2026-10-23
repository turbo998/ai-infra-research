"""Unit tests for Phase 65 (Priority 74) experiments across Track A, Track B, and Track C."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase65Experiments(unittest.TestCase):
    def test_track_a_flashinfer_mhc_frost_stepfun_tilelang(self) -> None:
        mod = load_module("experiments/flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline/run.py")
        res = mod.run_experiment()
        self.assertEqual(res["phase"], 65)
        self.assertEqual(res["priority"], 74)
        self.assertTrue(all(res["invariants"].values()))

        mhc = res["mega_mhc_h4096"]
        self.assertEqual(mhc["fp8_group128"]["y_fp8_shape"], [128, 4096])
        self.assertEqual(mhc["fp8_group128"]["y_fp8_sf_shape"], [128, 32])
        self.assertEqual(mhc["nvfp4"]["y_fp4_shape"], [128, 2048])
        self.assertEqual(mhc["nvfp4"]["y_fp4_sf_shape"], [128, 256])

        frost = res["frost_tvm_ffi"]
        self.assertGreaterEqual(frost["max_public_speedup_x"], 4.2)
        self.assertGreaterEqual(frost["max_prepacked_speedup_x"], 8.7)

        stepfun = res["stepfun_mxfp8_tile128"]
        self.assertEqual(stepfun["selected_tile_at_1024_before"], 64)
        self.assertEqual(stepfun["selected_tile_at_1024_after"], 128)

        tl = res["tilelang_pipeline"]
        self.assertEqual(tl["transitive_worklist_producers"], [1, 0])
        self.assertEqual(tl["pto_mad_predicate_nodes_32_guards_after"], 1)

    def test_track_b_sglang_offloader_trtllm_ulysses_vllm_bad_words(self) -> None:
        mod = load_module("experiments/sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words/run.py")
        res = mod.run_experiment()
        self.assertEqual(res["phase"], 65)
        self.assertEqual(res["priority"], 74)
        self.assertTrue(all(res["invariants"].values()))

        sg = res["sglang_offloader_and_compile"]
        self.assertIn("kda.qkv_conv1d.weight", sg["resident_parameters"])
        self.assertEqual(sg["dynamo_compilations_after_fix"], 1)

        uly = res["trtllm_ulysses_packed_qkv"]
        self.assertEqual(uly["threads_per_block"], 384)
        self.assertEqual(uly["hbm_traffic_reduction_pct"], 50.0)

        bw = res["vllm_rust_bad_words"]
        self.assertTrue(bw["empty_word_rejected_before_tokenize"])
        self.assertTrue(bw["cap_exceeded_rejected"])
        self.assertEqual(bw["valid_case"]["unique_words"], 3)

    def test_track_c_te_ce_argmax_lmcache_2d_nixl_s3_rdma(self) -> None:
        mod = load_module("experiments/te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma/run.py")
        res = mod.run_experiment()
        self.assertEqual(res["phase"], 65)
        self.assertEqual(res["priority"], 74)
        self.assertTrue(all(res["invariants"].values()))

        te = res["transformer_engine"]
        self.assertEqual(te["nan_row_argmax_index"], 2)
        self.assertEqual(te["multi_nan_first_wins_index"], 4)
        self.assertEqual(te["tie_break_first_wins_index"], 1)
        self.assertEqual(te["grouped_mxfp8_fixed_scale_offset"], 2048)

        lmc = res["lmcache_2d_copy"]
        self.assertEqual(lmc["batch_memcpy_entries_1d_before"], 1952)
        self.assertEqual(lmc["batch_memcpy_entries_2d_after"], 32)
        self.assertEqual(lmc["descriptor_reduction_factor_x"], 61.0)

        nixl = res["nixl_s3_rdma_and_gpunetio"]
        self.assertEqual(nixl["transfer_9gib"]["num_parts"], 3)
        self.assertEqual(nixl["transfer_64gib"]["num_parts"], 17)
        self.assertEqual(nixl["gpunetio_wqes_two_sge_after"], 32)

    def test_cli_check_mode_for_all_phase65_experiments(self) -> None:
        slugs = [
            "flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline",
            "sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words",
            "te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma",
        ]
        for slug in slugs:
            script = ROOT / "experiments" / slug / "run.py"
            summary = ROOT / "experiments" / slug / "summary.json"
            self.assertTrue(summary.exists(), f"Missing {summary}")
            proc = subprocess.run(
                [sys.executable, str(script), "--check"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            data = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(data["experiment"], slug)


if __name__ == "__main__":
    unittest.main()
