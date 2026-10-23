import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase63Experiments(unittest.TestCase):
    def test_track_a_flashinfer_dsa_dense_gemm_kda_tilelang(self):
        mod = load_module(
            "experiments/flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang/run.py",
            "phase63_track_a",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 63)
        self.assertEqual(summary["priority"], 72)

        dsa = summary["dsa_indexer_far_sample"]
        self.assertEqual(dsa["z1_gb200_speedup"], 1.7797)
        self.assertEqual(dsa["s6sink_gb200_speedup"], 1.7338)
        self.assertEqual(dsa["smem_saved_bytes"], 12160)

        gemm = summary["dense_projection_gemm"]
        self.assertEqual(gemm["program_instances"]["pr_total"], 247)
        self.assertEqual(gemm["router_splits"]["pr"]["nn_t"], 6)
        self.assertEqual(gemm["gb300_highlights"][0]["speedup"], 3.548)

        kda = summary["kda_doc_group_batching"]
        self.assertEqual(kda["partitions_s261393"]["g3"], [[0, 2], [2, 3], [3, 4]])
        self.assertEqual(kda["gb300_arena_rows"][2]["reduction_factor"], 3.62)
        self.assertEqual(kda["gb300_arena_rows"][4]["reduction_factor"], 6.36)

        tl = summary["code_folding_and_tilelang"]
        self.assertEqual(tl["finalize_reducer_simulation"]["fixed_emitted_offsets"], [0, 2, 4, 6])

        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang/run.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track_b_sglang_trtllm_vllm(self):
        mod = load_module(
            "experiments/sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope/run.py",
            "phase63_track_b",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 63)
        self.assertEqual(summary["priority"], 72)

        sgl = summary["sglang_moe_lora_and_gguf"]
        self.assertEqual(sgl["virtual_expert_cases"]["active_slot2_expert5"], 21)
        self.assertEqual(sgl["virtual_expert_cases"]["disabled_slot1_expert5"], -1)
        self.assertEqual(sgl["virtual_expert_cases"]["shared_outer_slot2_expert3"], 2)
        self.assertTrue(sgl["gguf_moe_weighting"]["ieee_preserved_nonzero"])

        trt = summary["trtllm_nvfp4_and_spec_dec"]
        self.assertEqual(trt["nvfp4_scratch_padding"]["tail_zero_bytes"], 73728)
        self.assertEqual(trt["device_copy_buffer"]["copies_skipped"], 2)
        self.assertEqual(trt["device_copy_buffer"]["final_device_buffer"], [10, 99, 30, 40])

        vllm = summary["vllm_rope_cpu_and_vl2_cache"]
        self.assertEqual(vllm["muse_spark_shared_rope"]["saved_mib"], 2496.0)
        self.assertTrue(vllm["deepseek_vl2_cropping_hash"]["collision_prevented"])

        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope/run.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track_c_torchtitan_te_megatron_mooncake_nixl(self):
        mod = load_module(
            "experiments/torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl/run.py",
            "phase63_track_c",
        )
        summary = mod.build_summary()
        self.assertEqual(summary["phase"], 63)
        self.assertEqual(summary["priority"], 72)

        titan = summary["torchtitan_mla_rope_and_swiglu"]
        self.assertEqual(titan["mla_qk_rows"][1]["speedup"], 1.5496)
        self.assertEqual(titan["mla_qk_rows"][3]["bw_floor_utilization_pct"], 99.6)
        self.assertEqual(titan["swiglu_rows"][2]["speedup_vs_main"], 1.3743)

        te = summary["te_mxfp8_and_megatron"]
        self.assertEqual(te["mxfp8_per_tile_swizzle"]["cta_parallelism_gain"], 32)
        self.assertEqual(te["megatron_async_staging"]["device_bytes_held_after_mib"], 0)

        kv = summary["kv_fabric_and_rdma_recovery"]
        recovered_count = sum(
            1 for r in kv["mooncake_rdma_1hz_poll_matrix"] if r["probes_gid_and_recovers"]
        )
        self.assertEqual(recovered_count, 1)
        self.assertEqual(kv["tent_device_plugin_error_code"], -3)

        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl/run.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
