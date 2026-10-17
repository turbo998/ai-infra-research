"""Unit tests for the own-offline capacity_economics experiment."""

import json
from pathlib import Path
import unittest

from experiments.capacity_economics.run import (
    calculate,
    evaluate_inference_capacity_economics,
    evaluate_innovator_economics,
    evaluate_training_economics,
    main,
    render,
)


HERE = Path(__file__).resolve().parents[1] / "experiments" / "capacity_economics"


class CapacityEconomicsTests(unittest.TestCase):
    def test_fixture_check_and_determinism(self) -> None:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        outputs_first = render(fixture)
        outputs_second = render(fixture)
        self.assertEqual(outputs_first, outputs_second)
        result = calculate(fixture)
        self.assertEqual(result["evidence_status"], "own-offline")
        self.assertTrue(result["synthetic"])
        self.assertEqual(len(result["cases"]), 3)

    def test_torchtitan_moe_flops_and_quantization_mfu_guard(self) -> None:
        bf16_res = evaluate_training_economics(
            is_moe=True,
            nparams_dense=4_800_000_000,
            nparams_embedding=900_000_000,
            enable_weight_tying=False,
            nparams_moe_router=200_000_000,
            nparams_shared_experts=2_100_000_000,
            nparams_experts=664_000_000_000,
            num_experts=256,
            top_k=8,
            n_layers=61,
            n_heads=128,
            dim=7168,
            seq_len=8192,
            tps_per_gpu=1680.0,
            gpu_peak_flops_bf16=989e12,
            has_quantization=False,
            world_size_gpus=2048,
            cluster_goodput=0.88,
            target_training_tokens=1_000_000_000_000,
            gpu_hourly_cost_usd=2.85,
            slurm_poll_interval_sec=30,
            gpu_watt0=670,
            gpu_watt1=695,
            electricity_price_per_kwh_usd=0.08,
        )
        # Active sparse params = 0.2B + 2.1B + 664B * 8 / 256 = 23.05B
        self.assertEqual(bf16_res["nparams_sparse_active"], 23_050_000_000)
        # Matmul params = (4.8B - 0.9B) + 23.05B = 26.95B
        self.assertEqual(bf16_res["nparams_for_matmul"], 26_950_000_000)
        self.assertIsNotNone(bf16_res["mfu_bf16_pct"])
        # Slurm trapezoidal energy: 30 * (670 + 695) / 2 = 20475 J
        self.assertEqual(bf16_res["slurm_step_joules_per_gpu"], 20475)

        fp8_res = evaluate_training_economics(
            is_moe=True,
            nparams_dense=4_800_000_000,
            nparams_embedding=900_000_000,
            enable_weight_tying=False,
            nparams_moe_router=200_000_000,
            nparams_shared_experts=2_100_000_000,
            nparams_experts=664_000_000_000,
            num_experts=256,
            top_k=8,
            n_layers=61,
            n_heads=128,
            dim=7168,
            seq_len=8192,
            tps_per_gpu=2350.0,
            gpu_peak_flops_bf16=989e12,
            has_quantization=True,
            world_size_gpus=2048,
            cluster_goodput=0.90,
            target_training_tokens=1_000_000_000_000,
            gpu_hourly_cost_usd=2.85,
            slurm_poll_interval_sec=30,
            gpu_watt0=680,
            gpu_watt1=700,
            electricity_price_per_kwh_usd=0.08,
        )
        self.assertIsNone(fp8_res["mfu_bf16_pct"])

    def test_vllm_burstiness_and_pd_disaggregation(self) -> None:
        poisson = evaluate_inference_capacity_economics(
            arrival_rate_rps=36.0,
            burstiness_gamma_shape=1.0,
            prompt_tokens=4096,
            cached_prefix_tokens=3072,
            output_tokens=512,
            slo_ttft_ms=350.0,
            slo_tpot_ms=25.0,
            prefill_tokens_per_sec_per_instance=16384.0,
            decode_max_concurrent_seqs_per_instance=96,
            decode_base_step_ms=8.0,
            decode_per_seq_step_ms=0.2,
            colocated_interference_penalty=0.22,
            gpus_per_prefill_instance=4,
            gpus_per_decode_instance=4,
            gpu_hourly_cost_usd=2.85,
        )
        bursty = evaluate_inference_capacity_economics(
            arrival_rate_rps=36.0,
            burstiness_gamma_shape=0.25,
            prompt_tokens=4096,
            cached_prefix_tokens=3072,
            output_tokens=512,
            slo_ttft_ms=350.0,
            slo_tpot_ms=25.0,
            prefill_tokens_per_sec_per_instance=16384.0,
            decode_max_concurrent_seqs_per_instance=96,
            decode_base_step_ms=8.0,
            decode_per_seq_step_ms=0.2,
            colocated_interference_penalty=0.22,
            gpus_per_prefill_instance=4,
            gpus_per_decode_instance=4,
            gpu_hourly_cost_usd=2.85,
        )
        self.assertGreater(bursty["arrival_cv_squared"], poisson["arrival_cv_squared"])
        self.assertLess(
            bursty["naive_92pct_ttft_goodput_fraction"],
            poisson["naive_92pct_ttft_goodput_fraction"],
        )
        self.assertGreater(poisson["gpu_savings_from_disagg"], 0)

    def test_innovator_models(self) -> None:
        inn = evaluate_innovator_economics(
            fw_num_servers=2,
            fw_num_gens_per_server=2,
            fw_prompt_tokens=4096,
            fw_cached_tokens=3072,
            fw_min_tokens_to_batch=8192,
            fw_worker_durations_sec=[0.50, 0.50, 0.50, 1.00],
            sequoia_num_layers=80,
            sequoia_stay_layers=20,
            sequoia_bytes_per_layer=1_700_000_000,
            sequoia_pcie_bw_gbps=56.0,
            sequoia_gpu_compute_ms_per_step=18.0,
            sequoia_accepted_tokens_per_step=5.0,
            groq_on_chip_cycles=900_000,
            groq_num_chips=8,
            groq_pipelined=True,
            groq_input_bytes=24_000,
            groq_output_bytes=24_000,
            cerebras_fabric_offset_x=1,
            cerebras_width=64,
            cerebras_pe_length=256,
            cerebras_loop_count=100,
            cerebras_raw_start_cycles=10_000,
            cerebras_raw_end_cycles=521_600,
            cerebras_raw_ref_cycles=1_080,
            cerebras_px=50,
            cerebras_py=30,
        )
        self.assertEqual(inn["fireworks"]["warmup_tokens_per_worker"], 3073)
        self.assertEqual(inn["fireworks"]["dynamic_batch_size"], 8)
        self.assertEqual(inn["together_sequoia"]["speculative_speedup_x"], 5.0)
        # 900,000 cycles at 900 MHz = 1.0 ms per stage, * 8 chips pipelined = 8.0 ms
        self.assertEqual(inn["groq"]["stage_compute_ms"], 1.0)
        self.assertEqual(inn["groq"]["deterministic_compute_ms"], 8.0)
        # 64 * 256 * 4 = 65536 bytes -> ceil(65536 / 46000) = 2 west buffer cols
        self.assertEqual(inn["cerebras"]["required_west_buf_cols_46kb_fifo"], 2)
        self.assertEqual(inn["cerebras"]["core_fabric_offset_x"], 6)
        self.assertEqual(inn["cerebras"]["compensated_ref_cycles"], 1000)


if __name__ == "__main__":
    unittest.main()
