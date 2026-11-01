import subprocess
import sys
import unittest


class TestPhase179Experiments(unittest.TestCase):
    def test_flashinfer_gdn_sm107a_kimi_k3_attn_res_cutlass_sm110_triton_nvfp4_phase179(self):
        res = subprocess.run(
            [
                sys.executable,
                "experiments/flashinfer_gdn_sm107a_kimi_k3_attn_res_cutlass_sm110_triton_nvfp4_phase179/run.py",
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_vllm_kv_dtype_moe_pad_skip_sglang_dsv41_wave_pd_linker_phase179(self):
        res = subprocess.run(
            [
                sys.executable,
                "experiments/vllm_kv_dtype_moe_pad_skip_sglang_dsv41_wave_pd_linker_phase179/run.py",
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_megatron_mfsdp_v2_outer_stream_mxfp8_te_thd_cp_nccl_mooncake_urma_phase179(self):
        res = subprocess.run(
            [
                sys.executable,
                "experiments/megatron_mfsdp_v2_outer_stream_mxfp8_te_thd_cp_nccl_mooncake_urma_phase179/run.py",
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
