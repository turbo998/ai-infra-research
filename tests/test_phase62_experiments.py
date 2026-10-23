from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(exp_slug: str):
    run_py = REPO_ROOT / "experiments" / exp_slug / "run.py"
    spec = importlib.util.spec_from_file_location(f"exp_{exp_slug.replace('-', '_')}", run_py)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPhase62Experiments(unittest.TestCase):
    def test_track_a_flashinfer_and_tilelang_invariants(self) -> None:
        slug = "flashinfer-tp-moe-finalize-muse-norm-kcp-carry-qsa-tilelang"
        mod = _load_module(slug)
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / slug / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)

        summary = json.loads((REPO_ROOT / "experiments" / slug / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 62)
        self.assertEqual(summary["priority"], 71)
        self.assertTrue(summary["tp_moe_route_guards"]["sm_103a_tp4"])
        self.assertFalse(summary["tp_moe_route_guards"]["sm_103a_tp8"])
        self.assertTrue(summary["tp_moe_route_guards"]["sm_100a_tp8"])
        self.assertFalse(summary["tp_moe_route_guards"]["sm_100a_tp4"])

        self.assertAlmostEqual(summary["headline_metrics"]["tp_moe_gb300_tp4_t256_speedup_x"], 1.334)
        self.assertAlmostEqual(summary["headline_metrics"]["muse_next_rmsnorm_b200_v1_m16384_speedup_x"], 1.818)
        self.assertAlmostEqual(summary["headline_metrics"]["kcp_q4_cp16_b3_speedup_x"], 1.0387)
        self.assertAlmostEqual(summary["headline_metrics"]["kcp_q4_cp16_carry_plane_mib"], 72.0)

        carry_p1 = mod.compute_kcp_saved_carry_plane_bytes(
            num_heads=64,
            n_frag=2,
            parallel_degree_p=1,
            n_in=1,
            n_out=1,
            saved_carry=True,
        )
        self.assertEqual(carry_p1["carry_plane_bytes"], 0)
        self.assertFalse(carry_p1["skips_backward_offset_rewalk"])

        epoch_guards = summary["mega_mhc_launch_epochs_guards"]
        self.assertTrue(epoch_guards["valid_same_grid"]["accepted"])
        self.assertFalse(epoch_guards["refused_mismatched_grid"]["accepted"])
        self.assertFalse(epoch_guards["refused_sliced_view"]["accepted"])

        tl_guards = summary["tilelang_ascend_guards"]
        self.assertTrue(tl_guards["int64_prevents_endpoint_wrap"])
        self.assertEqual(tl_guards["fixed_strided_offset"], 776)
        self.assertTrue(tl_guards["child_only_lifetime_rejected_on_sibling_writer"])

    def test_track_b_sglang_trtllm_vllm_invariants(self) -> None:
        slug = "sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva"
        _load_module(slug)
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / slug / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)

        summary = json.loads((REPO_ROOT / "experiments" / slug / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 62)
        self.assertEqual(summary["priority"], 71)

        cutlass = summary["sglang_cutlass_prepare_moe_input"]
        self.assertEqual(cutlass["expert_offsets"], [0, 3, 4, 6, 6])
        self.assertEqual(cutlass["blockscale_offsets"], [0, 128, 256, 384, 384])
        self.assertEqual(cutlass["inactive_routes"], 2)
        self.assertTrue(cutlass["tail_padding_points_to_token_zero"])

        marlin = summary["sglang_marlin_inactive_zeroing"]
        self.assertEqual(marlin["zeroed_routes"], [1, 4])
        self.assertTrue(marlin["all_valid_routes_covered"])

        tail_own = summary["sglang_decoder_tail_ownership"]
        self.assertTrue(tail_own["tp_sharded_global_bound"]["can_capture_moe_bound"])
        self.assertFalse(tail_own["unsharded_global_bound"]["can_capture_moe_bound"])

        kv_reuse = summary["trtllm_connector_kv_reuse"]
        self.assertEqual(kv_reuse["aligned_0_256"]["reused_full_blocks"], 4)
        self.assertEqual(kv_reuse["unaligned_16_240"]["reused_full_blocks"], 2)
        self.assertEqual(kv_reuse["unaligned_16_240"]["partial_boundary_blocks_kept_as_miss"], 2)

        self.assertAlmostEqual(summary["vllm_mamba_sibling_tail"]["step_reduction_factor"], 8.0)
        self.assertAlmostEqual(summary["vllm_uva_host_resident_moe"]["pcie_traffic_reduction_x"], 3.2)

    def test_track_c_megatron_torchtitan_mooncake_nixl_dynamo_invariants(self) -> None:
        slug = "megatron-mfsdp-grad-comm-titan-muon-mxfp8-mooncake-nixl-dynamo"
        _load_module(slug)
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / slug / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)

        summary = json.loads((REPO_ROOT / "experiments" / slug / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 62)
        self.assertEqual(summary["priority"], 71)

        mfsdp = summary["mfsdp_v2_grad_comm_dtype"]
        self.assertEqual(mfsdp["effective_grad_comm_dtype_bytes"], 4)
        self.assertGreater(mfsdp["fixed_partial_grad_mib"], mfsdp["buggy_partial_grad_mib"])

        muon = summary["torchtitan_dist_muon_bf16"]
        self.assertAlmostEqual(muon["rtx3090_pcie_speedup_x"], 2.2502)
        self.assertAlmostEqual(muon["a100_speedup_x"], 1.024)
        self.assertTrue(muon["deferred_row_scale_in_storage_fp32"])

        mxfp8 = summary["torchtitan_mxfp8_dist_moe"]
        self.assertEqual(mxfp8["generator_quantizations_after_sync"], 96)
        self.assertEqual(mxfp8["forward_re_quantizations_after_fix"], 0)
        self.assertAlmostEqual(mxfp8["quantization_reduction_x"], 10.0)

        lmcache = summary["lmcache_trtllm_batch_limit"]
        self.assertEqual(lmcache["new_batch_limit"], 4)
        self.assertEqual(lmcache["new_kernel_calls"], 8)

        mooncake = summary["mooncake_descriptor_free_readability"]
        self.assertEqual(mooncake["new_descriptor_allocs"], 0)
        self.assertEqual(mooncake["new_atomic_shared_ptr_loads"], 0)

        nixl = summary["nixl_libfabric_source_address"]
        self.assertFalse(nixl["peer_1024_v2"]["old_8bit_wire_accepted"])
        self.assertTrue(nixl["peer_1024_v2"]["new_32bit_local_accepted"])
        self.assertEqual(nixl["peer_70000_v2"]["recovered_peer_idx"], 70000)
        self.assertFalse(nixl["peer_12_v1_rejected"]["new_32bit_local_accepted"])

        dynamo = summary["dynamo_mooncake_tenant_isolation"]
        self.assertTrue(dynamo["cross_tenant_leak_prevented"])


if __name__ == "__main__":
    unittest.main()
