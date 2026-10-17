"""Evaluate GPU training token economics (TorchTitan MFU & Slurm trapezoidal energy), inference SLO goodput capacity sizing (vLLM Gamma burstiness & P/D disaggregation), and AI Infra innovator capacity/I/O models (Fireworks, Together/Sequoia, Groq, Cerebras)."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
GROQ_CLOCK_HZ = 900_000_000.0
GROQ_PCIE_BW_BYTES_PER_S = 24_000_000_000.0
GROQ_PCIE_LATENCY_SEC = 1e-5
CEREBRAS_CLOCK_GHZ = 0.85
CEREBRAS_FIFO_BYTES_PER_PE = 46_000
LIMITATIONS = [
    "own-offline synthetic model; no live GPU cluster, Slurm daemon, NVML energy sensor, vLLM server, Groq LPU, or Cerebras WSE hardware execution",
    "reproduces static formulas from torchtitan@086bf6c166ec85c1298eb5596fa9bf95f6a2d840, vllm@25f560a62c4f955672e2c6080b17ab3a48f96201, slurm@ad46cad8e0969e8e5945b88f0a21f6bed0f9a3bf, fw-ai/benchmark@94cd111967c7be51a35777c475d7c5d2b66ec94a, Sequoia@0ff3ff71475ca148f2dfb7545e0de5caf5c67b39, groqflow@720e024414be401a7853931a2622a26446883746, and sdk-examples@d19f2ca3e51285fc5904c78643b38819a2f26c44",
    "queueing headroom uses a first-order Kingman G/G/m waiting-time approximation parameterized by vLLM Gamma arrival burstiness (CV^2 = 1 / burstiness)",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def _pos_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}: expected finite positive number")
    return float(value)


def _nonneg_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: expected non-negative int")
    return value


def evaluate_training_economics(
    *,
    is_moe: bool,
    nparams_dense: int,
    nparams_embedding: int,
    enable_weight_tying: bool,
    nparams_moe_router: int,
    nparams_shared_experts: int,
    nparams_experts: int,
    num_experts: int,
    top_k: int,
    n_layers: int,
    n_heads: int,
    dim: int,
    seq_len: int,
    tps_per_gpu: float,
    gpu_peak_flops_bf16: float,
    has_quantization: bool,
    world_size_gpus: int,
    cluster_goodput: float,
    target_training_tokens: int,
    gpu_hourly_cost_usd: float,
    slurm_poll_interval_sec: int,
    gpu_watt0: int,
    gpu_watt1: int,
    electricity_price_per_kwh_usd: float,
) -> dict:
    """Reproduce TorchTitan model FLOPs/MFU (models/utils.py & components/metrics.py) and Slurm GPU energy integration (acct_gather_energy_gpu.c)."""
    if type(is_moe) is not bool or type(enable_weight_tying) is not bool or type(has_quantization) is not bool:
        raise ValueError("boolean flags: expected bool")
    dense_params = integer(nparams_dense, "nparams_dense")
    emb_params = _nonneg_int(nparams_embedding, "nparams_embedding")
    if emb_params >= dense_params:
        raise ValueError("nparams_embedding: must be smaller than nparams_dense")

    layers = integer(n_layers, "n_layers")
    heads = integer(n_heads, "n_heads")
    model_dim = integer(dim, "dim")
    if model_dim % heads != 0:
        raise ValueError("dim: must be divisible by n_heads")
    seq = integer(seq_len, "seq_len")
    tps = _pos_float(tps_per_gpu, "tps_per_gpu")
    peak_flops = _pos_float(gpu_peak_flops_bf16, "gpu_peak_flops_bf16")
    gpus = integer(world_size_gpus, "world_size_gpus")
    goodput = _pos_float(cluster_goodput, "cluster_goodput")
    if goodput > 1.0:
        raise ValueError("cluster_goodput: must be in (0, 1]")
    target_tokens = integer(target_training_tokens, "target_training_tokens")
    gpu_cost = _pos_float(gpu_hourly_cost_usd, "gpu_hourly_cost_usd")
    poll_sec = integer(slurm_poll_interval_sec, "slurm_poll_interval_sec")
    w0 = integer(gpu_watt0, "gpu_watt0")
    w1 = integer(gpu_watt1, "gpu_watt1")
    kwh_price = _pos_float(electricity_price_per_kwh_usd, "electricity_price_per_kwh_usd")

    dense_non_emb = dense_params if enable_weight_tying else (dense_params - emb_params)
    if is_moe:
        router_params = _nonneg_int(nparams_moe_router, "nparams_moe_router")
        shared_params = _nonneg_int(nparams_shared_experts, "nparams_shared_experts")
        routed_params = integer(nparams_experts, "nparams_experts")
        experts_total = integer(num_experts, "num_experts")
        k_active = integer(top_k, "top_k")
        if k_active > experts_total:
            raise ValueError("top_k: cannot exceed num_experts")
        nparams_sparse = router_params + shared_params + routed_params
        nparams_sparse_active = router_params + shared_params + (routed_params * k_active // experts_total)
        total_nparams = dense_params + nparams_sparse
        nparams_for_matmul = dense_non_emb + nparams_sparse_active
    else:
        nparams_sparse_active = 0
        total_nparams = dense_params
        nparams_for_matmul = dense_non_emb

    # torchtitan/models/llama3/model.py:95 sets head_dims = 2 * (dim // n_heads)
    head_dims = 2 * (model_dim // heads)
    # torchtitan/models/utils.py:400 & 480
    num_flops_per_token = 6 * nparams_for_matmul + 6 * layers * heads * head_dims * seq

    # torchtitan/components/metrics.py:402-409
    achieved_tflops_per_gpu = num_flops_per_token * tps / 1e12
    mfu_bf16_pct = None if has_quantization else round(100.0 * num_flops_per_token * tps / peak_flops, 2)

    cluster_raw_tps = gpus * tps
    cluster_effective_tps = cluster_raw_tps * goodput
    training_wall_hours = target_tokens / (cluster_effective_tps * 3600.0)
    total_gpu_hours = training_wall_hours * gpus
    compute_rental_cost_usd = total_gpu_hours * gpu_cost
    cost_per_million_training_tokens_usd = compute_rental_cost_usd / (target_tokens / 1e6)

    # SchedMD/slurm acct_gather_energy_gpu.c:147-151 _get_additional_consumption trapezoidal rule
    slurm_step_joules_per_gpu = int(poll_sec * (w0 + w1) / 2)
    avg_watts_per_gpu = (w0 + w1) / 2.0
    total_gpu_energy_kwh = (avg_watts_per_gpu * total_gpu_hours) / 1000.0
    gpu_electricity_cost_usd = total_gpu_energy_kwh * kwh_price

    return {
        "total_nparams": total_nparams,
        "nparams_sparse_active": nparams_sparse_active,
        "nparams_for_matmul": nparams_for_matmul,
        "num_flops_per_token": num_flops_per_token,
        "achieved_tflops_per_gpu": round(achieved_tflops_per_gpu, 2),
        "mfu_bf16_pct": mfu_bf16_pct,
        "cluster_effective_tps": round(cluster_effective_tps, 2),
        "training_wall_hours": round(training_wall_hours, 2),
        "total_gpu_hours": round(total_gpu_hours, 2),
        "cost_per_million_training_tokens_usd": round(cost_per_million_training_tokens_usd, 4),
        "slurm_step_joules_per_gpu": slurm_step_joules_per_gpu,
        "total_gpu_energy_kwh": round(total_gpu_energy_kwh, 2),
        "gpu_electricity_cost_usd": round(gpu_electricity_cost_usd, 2),
    }


def evaluate_inference_capacity_economics(
    *,
    arrival_rate_rps: float,
    burstiness_gamma_shape: float,
    prompt_tokens: int,
    cached_prefix_tokens: int,
    output_tokens: int,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
    prefill_tokens_per_sec_per_instance: float,
    decode_max_concurrent_seqs_per_instance: int,
    decode_base_step_ms: float,
    decode_per_seq_step_ms: float,
    colocated_interference_penalty: float,
    gpus_per_prefill_instance: int,
    gpus_per_decode_instance: int,
    gpu_hourly_cost_usd: float,
) -> dict:
    """Model vLLM Gamma burstiness, SLO goodput filtering, and Prefill/Decode disaggregated vs colocated GPU sizing."""
    rps = _pos_float(arrival_rate_rps, "arrival_rate_rps")
    burstiness = _pos_float(burstiness_gamma_shape, "burstiness_gamma_shape")
    l_in = integer(prompt_tokens, "prompt_tokens")
    l_cached = _nonneg_int(cached_prefix_tokens, "cached_prefix_tokens")
    if l_cached >= l_in:
        raise ValueError("cached_prefix_tokens: must be strictly less than prompt_tokens")
    l_out = integer(output_tokens, "output_tokens")
    ttft_slo = _pos_float(slo_ttft_ms, "slo_ttft_ms")
    tpot_slo = _pos_float(slo_tpot_ms, "slo_tpot_ms")
    prefill_tok_rate = _pos_float(prefill_tokens_per_sec_per_instance, "prefill_tokens_per_sec_per_instance")
    decode_hbm_cap = integer(decode_max_concurrent_seqs_per_instance, "decode_max_concurrent_seqs_per_instance")
    t_base = _pos_float(decode_base_step_ms, "decode_base_step_ms")
    t_slope = _pos_float(decode_per_seq_step_ms, "decode_per_seq_step_ms")
    interference = _pos_float(colocated_interference_penalty, "colocated_interference_penalty")
    if interference >= 1.0:
        raise ValueError("colocated_interference_penalty: must be in (0, 1)")
    gpus_p = integer(gpus_per_prefill_instance, "gpus_per_prefill_instance")
    gpus_d = integer(gpus_per_decode_instance, "gpus_per_decode_instance")
    gpu_cost = _pos_float(gpu_hourly_cost_usd, "gpu_hourly_cost_usd")

    # vLLM benchmark_serving.py:106 uses Gamma(shape=burstiness, scale=1/(rate*burstiness)) -> CV_a^2 = 1/burstiness
    arrival_cv_squared = 1.0 / burstiness
    uncached_tokens = max(1, l_in - l_cached)
    service_prefill_ms = (uncached_tokens / prefill_tok_rate) * 1000.0
    if service_prefill_ms >= ttft_slo:
        raise ValueError("service_prefill_ms: single-request prefill exceeds slo_ttft_ms even at zero queueing")

    # Maximum allowed queueing ratio W_q / S_p <= (SLO_TTFT - S_p) / S_p
    # Using Kingman factor (CV_a^2 + CV_s^2)/2 * (rho / (1 - rho)) <= max_wait_ratio with deterministic service CV_s^2 = 0
    max_wait_ratio = (ttft_slo - service_prefill_ms) / service_prefill_ms
    kingman_coeff = 0.5 * arrival_cv_squared
    max_safe_prefill_utilization = max_wait_ratio / (max_wait_ratio + kingman_coeff)
    # Cap safe steady-state utilization at 0.85 for tail stability
    safe_prefill_utilization = min(0.85, max_safe_prefill_utilization)

    raw_prefill_rps_per_instance = 1000.0 / service_prefill_ms
    effective_prefill_rps_per_instance = raw_prefill_rps_per_instance * safe_prefill_utilization
    required_prefill_instances = math.ceil(rps / effective_prefill_rps_per_instance)

    # Decode batch concurrency limited by both HBM capacity and TPOT SLO: t_base + B * t_slope <= slo_tpot_ms
    if t_base >= tpot_slo:
        raise ValueError("decode_base_step_ms: base decode step exceeds slo_tpot_ms")
    slo_max_batch = int((tpot_slo - t_base) // t_slope)
    if slo_max_batch < 1:
        raise ValueError("slo_tpot_ms: cannot fit even 1 sequence within TPOT SLO")
    effective_decode_batch = min(decode_hbm_cap, slo_max_batch)
    achieved_tpot_ms = t_base + effective_decode_batch * t_slope
    decode_duration_sec = (max(1, l_out - 1) * achieved_tpot_ms) / 1000.0
    # Little's Law: sustainable decode RPS per instance = B_eff / T_decode
    effective_decode_rps_per_instance = effective_decode_batch / decode_duration_sec
    required_decode_instances = math.ceil(rps / effective_decode_rps_per_instance)

    disagg_total_gpus = required_prefill_instances * gpus_p + required_decode_instances * gpus_d

    # Colocated sizing: each replica must supply both prefill and decode time slices, degraded by prefill-decode interference
    colocated_gpu_unit = max(gpus_p, gpus_d)
    raw_combined_instances = (rps / effective_prefill_rps_per_instance) + (rps / effective_decode_rps_per_instance)
    required_colocated_instances = math.ceil(raw_combined_instances / (1.0 - interference))
    colocated_total_gpus = required_colocated_instances * colocated_gpu_unit

    # Goodput comparison if an architect naively provisions at 92% utilization without burstiness headroom
    naive_prefill_instances = max(1, math.ceil(rps / (raw_prefill_rps_per_instance * 0.92)))
    naive_rho = min(0.99, rps / (naive_prefill_instances * raw_prefill_rps_per_instance))
    naive_expected_wait_ms = service_prefill_ms * kingman_coeff * (naive_rho / (1.0 - naive_rho))
    naive_expected_ttft_ms = service_prefill_ms + naive_expected_wait_ms
    # First-order exponential tail probability P(TTFT <= SLO_TTFT)
    if naive_expected_wait_ms <= 0.0:
        naive_goodput_fraction = 1.0
    else:
        slack_ms = max(0.0, ttft_slo - service_prefill_ms)
        naive_goodput_fraction = 1.0 - math.exp(-slack_ms / naive_expected_wait_ms)

    disagg_hourly_cost_usd = disagg_total_gpus * gpu_cost
    goodput_output_tokens_per_hour = rps * l_out * 3600.0
    disagg_cost_per_million_goodput_output_tokens_usd = (
        disagg_hourly_cost_usd / (goodput_output_tokens_per_hour / 1e6)
    )

    return {
        "arrival_cv_squared": round(arrival_cv_squared, 4),
        "uncached_tokens": uncached_tokens,
        "service_prefill_ms": round(service_prefill_ms, 2),
        "safe_prefill_utilization": round(safe_prefill_utilization, 4),
        "effective_prefill_rps_per_instance": round(effective_prefill_rps_per_instance, 4),
        "effective_decode_batch": effective_decode_batch,
        "achieved_tpot_ms": round(achieved_tpot_ms, 2),
        "effective_decode_rps_per_instance": round(effective_decode_rps_per_instance, 4),
        "required_prefill_instances": required_prefill_instances,
        "required_decode_instances": required_decode_instances,
        "prefill_to_decode_instance_ratio": round(required_prefill_instances / required_decode_instances, 4),
        "disagg_total_gpus": disagg_total_gpus,
        "colocated_total_gpus": colocated_total_gpus,
        "gpu_savings_from_disagg": colocated_total_gpus - disagg_total_gpus,
        "naive_92pct_expected_ttft_ms": round(naive_expected_ttft_ms, 2),
        "naive_92pct_ttft_goodput_fraction": round(naive_goodput_fraction, 4),
        "disagg_cost_per_million_goodput_output_tokens_usd": round(
            disagg_cost_per_million_goodput_output_tokens_usd, 4
        ),
    }


def evaluate_innovator_economics(
    *,
    fw_num_servers: int,
    fw_num_gens_per_server: int,
    fw_prompt_tokens: int,
    fw_cached_tokens: int,
    fw_min_tokens_to_batch: int,
    fw_worker_durations_sec: list[float],
    sequoia_num_layers: int,
    sequoia_stay_layers: int,
    sequoia_bytes_per_layer: int,
    sequoia_pcie_bw_gbps: float,
    sequoia_gpu_compute_ms_per_step: float,
    sequoia_accepted_tokens_per_step: float,
    groq_on_chip_cycles: int,
    groq_num_chips: int,
    groq_pipelined: bool,
    groq_input_bytes: int,
    groq_output_bytes: int,
    cerebras_fabric_offset_x: int,
    cerebras_width: int,
    cerebras_pe_length: int,
    cerebras_loop_count: int,
    cerebras_raw_start_cycles: int,
    cerebras_raw_end_cycles: int,
    cerebras_raw_ref_cycles: int,
    cerebras_px: int,
    cerebras_py: int,
) -> dict:
    """Reproduce exact capacity/latency/bandwidth equations from Fireworks, Together/Sequoia, Groq, and Cerebras primary sources."""
    # 1. Fireworks AI (fw-ai/benchmark/llm_bench/prefill_load_test.py:42-60, 128-140, 297-318, 434-441)
    num_srv = integer(fw_num_servers, "fw_num_servers")
    num_gen = integer(fw_num_gens_per_server, "fw_num_gens_per_server")
    fw_p_tok = integer(fw_prompt_tokens, "fw_prompt_tokens")
    fw_c_tok = _nonneg_int(fw_cached_tokens, "fw_cached_tokens")
    if fw_c_tok >= fw_p_tok:
        raise ValueError("fw_cached_tokens: must be less than fw_prompt_tokens")
    fw_min_batch_tok = integer(fw_min_tokens_to_batch, "fw_min_tokens_to_batch")
    if not isinstance(fw_worker_durations_sec, list) or len(fw_worker_durations_sec) != num_srv * num_gen:
        raise ValueError("fw_worker_durations_sec: length must equal fw_num_servers * fw_num_gens_per_server")
    worker_durs = [_pos_float(d, "fw_worker_durations_sec") for d in fw_worker_durations_sec]

    fw_total_workers = num_srv * num_gen
    fw_warmup_tokens_per_worker = (fw_c_tok + 1) if fw_c_tok > 0 else 0
    fw_uncached_tokens = fw_p_tok - fw_c_tok
    fw_batch_size = max(1, fw_min_batch_tok // fw_uncached_tokens)
    fw_straggler_duration_sec = max(worker_durs)
    fw_total_uncached_tokens_processed = fw_total_workers * fw_batch_size * fw_uncached_tokens
    fw_effective_uncached_tok_per_s = fw_total_uncached_tokens_processed / fw_straggler_duration_sec

    # 2. Together AI / Sequoia (Infini-AI-Lab/Sequoia/Engine/offload_engine.py:66-127, 179-215)
    seq_layers = integer(sequoia_num_layers, "sequoia_num_layers")
    seq_stay = _nonneg_int(sequoia_stay_layers, "sequoia_stay_layers")
    if seq_stay > seq_layers:
        raise ValueError("sequoia_stay_layers: cannot exceed sequoia_num_layers")
    seq_layer_bytes = integer(sequoia_bytes_per_layer, "sequoia_bytes_per_layer")
    seq_pcie_bw = _pos_float(sequoia_pcie_bw_gbps, "sequoia_pcie_bw_gbps")
    seq_comp_ms = _pos_float(sequoia_gpu_compute_ms_per_step, "sequoia_gpu_compute_ms_per_step")
    seq_accept = _pos_float(sequoia_accepted_tokens_per_step, "sequoia_accepted_tokens_per_step")

    seq_offloaded_layers = seq_layers - seq_stay
    seq_pcie_transfer_ms_per_step = (seq_offloaded_layers * seq_layer_bytes) / (seq_pcie_bw * 1e6)
    # Ping-pong double buffer (self.buffer[0..1] on load_stream) overlaps PCIe H2D with GPU compute
    seq_step_latency_ms = max(seq_comp_ms, seq_pcie_transfer_ms_per_step)
    seq_baseline_tok_per_s = 1000.0 / seq_step_latency_ms
    seq_speculative_tok_per_s = (1000.0 * seq_accept) / seq_step_latency_ms

    # 3. Groq (groq/groqflow/groqflow/groqmodel/groqmodel.py:126-164)
    if type(groq_pipelined) is not bool:
        raise ValueError("groq_pipelined: expected bool")
    g_cycles = integer(groq_on_chip_cycles, "groq_on_chip_cycles")
    g_chips = integer(groq_num_chips, "groq_num_chips")
    g_in_bytes = integer(groq_input_bytes, "groq_input_bytes")
    g_out_bytes = integer(groq_output_bytes, "groq_output_bytes")

    groq_pcie_in_sec = (g_in_bytes / GROQ_PCIE_BW_BYTES_PER_S) + GROQ_PCIE_LATENCY_SEC
    groq_stage_compute_sec = g_cycles / GROQ_CLOCK_HZ
    groq_deterministic_compute_sec = groq_stage_compute_sec * (g_chips if groq_pipelined else 1)
    groq_pcie_out_sec = (g_out_bytes / GROQ_PCIE_BW_BYTES_PER_S) + GROQ_PCIE_LATENCY_SEC
    groq_total_e2e_latency_ms = (groq_pcie_in_sec + groq_deterministic_compute_sec + groq_pcie_out_sec) * 1000.0
    groq_pcie_overhead_pct = (
        100.0 * (groq_pcie_in_sec + groq_pcie_out_sec)
        / (groq_pcie_in_sec + groq_deterministic_compute_sec + groq_pcie_out_sec)
    )

    # 4. Cerebras (Cerebras/sdk-examples/benchmarks/bandwidth-test/run.py:92-122, 232-244, 336-380)
    c_off_x = integer(cerebras_fabric_offset_x, "cerebras_fabric_offset_x")
    c_width = integer(cerebras_width, "cerebras_width")
    c_pe_len = integer(cerebras_pe_length, "cerebras_pe_length")
    c_loops = integer(cerebras_loop_count, "cerebras_loop_count")
    c_start = integer(cerebras_raw_start_cycles, "cerebras_raw_start_cycles")
    c_end = integer(cerebras_raw_end_cycles, "cerebras_raw_end_cycles")
    c_ref = _nonneg_int(cerebras_raw_ref_cycles, "cerebras_raw_ref_cycles")
    c_px = _nonneg_int(cerebras_px, "cerebras_px")
    c_py = _nonneg_int(cerebras_py, "cerebras_py")

    c_wvlts = c_width * c_pe_len
    c_required_buf_cols = math.ceil((c_wvlts * 4) / CEREBRAS_FIFO_BYTES_PER_PE)
    c_core_fabric_offset_x = c_off_x + 3 + c_required_buf_cols
    c_min_fabric_width = c_core_fabric_offset_x + c_width + 2 + 1 + c_required_buf_cols
    c_compensated_ref = c_ref - (c_px + c_py)
    c_rel_start = c_start - c_compensated_ref
    c_rel_end = c_end - c_compensated_ref
    if c_rel_end <= c_rel_start:
        raise ValueError("cerebras_raw_end_cycles: must exceed cerebras_raw_start_cycles")
    c_cycles_send = c_rel_end - c_rel_start
    c_time_send_us = (c_cycles_send / CEREBRAS_CLOCK_GHZ) * 1e-3
    c_bandwidth_mb_per_s = ((c_wvlts * 4) / c_time_send_us) * c_loops

    return {
        "fireworks": {
            "total_workers": fw_total_workers,
            "warmup_tokens_per_worker": fw_warmup_tokens_per_worker,
            "uncached_tokens_per_req": fw_uncached_tokens,
            "dynamic_batch_size": fw_batch_size,
            "straggler_duration_sec": round(fw_straggler_duration_sec, 4),
            "effective_uncached_tok_per_s": round(fw_effective_uncached_tok_per_s, 2),
        },
        "together_sequoia": {
            "offloaded_layers": seq_offloaded_layers,
            "pcie_transfer_ms_per_step": round(seq_pcie_transfer_ms_per_step, 2),
            "step_latency_ms": round(seq_step_latency_ms, 2),
            "baseline_offload_tok_per_s": round(seq_baseline_tok_per_s, 2),
            "speculative_offload_tok_per_s": round(seq_speculative_tok_per_s, 2),
            "speculative_speedup_x": round(seq_speculative_tok_per_s / seq_baseline_tok_per_s, 2),
        },
        "groq": {
            "stage_compute_ms": round(groq_stage_compute_sec * 1000.0, 4),
            "deterministic_compute_ms": round(groq_deterministic_compute_sec * 1000.0, 4),
            "pcie_transfer_total_ms": round((groq_pcie_in_sec + groq_pcie_out_sec) * 1000.0, 4),
            "total_e2e_latency_ms": round(groq_total_e2e_latency_ms, 4),
            "pcie_overhead_pct": round(groq_pcie_overhead_pct, 2),
        },
        "cerebras": {
            "total_wavelets": c_wvlts,
            "required_west_buf_cols_46kb_fifo": c_required_buf_cols,
            "core_fabric_offset_x": c_core_fabric_offset_x,
            "min_fabric_width": c_min_fabric_width,
            "compensated_ref_cycles": c_compensated_ref,
            "cycles_send": c_cycles_send,
            "time_send_us": round(c_time_send_us, 4),
            "measured_wafer_io_bw_mb_per_s": round(c_bandwidth_mb_per_s, 2),
        },
    }


def evaluate_capacity_economics_case(*, case_id: str, training: dict, inference: dict, innovators: dict) -> dict:
    """Evaluate a complete capacity economics scenario across training, inference, and AI Infra innovators."""
    label = _ascii_label(case_id, "case_id")
    if not isinstance(training, dict) or not isinstance(inference, dict) or not isinstance(innovators, dict):
        raise ValueError("case sections: expected training, inference, and innovators dicts")
    return {
        "case_id": label,
        "training": evaluate_training_economics(**training),
        "inference": evaluate_inference_capacity_economics(**inference),
        "innovators": evaluate_innovator_economics(**innovators),
    }


def calculate(fixture: dict) -> dict:
    """Calculate deterministic own-offline results from an explicit synthetic fixture."""
    if (
        not isinstance(fixture, dict)
        or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1
        or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("cases"), list)
        or not fixture["cases"]
    ):
        raise ValueError("fixture: expected version-one synthetic cases list")

    cases = []
    seen_ids = set()
    for raw_case in fixture["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError("cases: expected case dict")
        evaluated = evaluate_capacity_economics_case(**raw_case)
        if evaluated["case_id"] in seen_ids:
            raise ValueError("cases: duplicate case_id")
        seen_ids.add(evaluated["case_id"])
        cases.append(evaluated)

    return {
        "schema_version": 1,
        "evidence_status": "own-offline",
        "synthetic": True,
        "cases": cases,
        "limitations": LIMITATIONS,
    }


def render(fixture: dict) -> dict[str, str]:
    """Return deterministic JSON and flat CSV representations of the same results."""
    result = calculate(fixture)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "evidence_status",
            "case_id",
            "mfu_bf16_pct",
            "cost_per_million_training_tokens_usd",
            "slurm_step_joules_per_gpu",
            "disagg_total_gpus",
            "colocated_total_gpus",
            "gpu_savings_from_disagg",
            "naive_92pct_ttft_goodput_fraction",
            "disagg_cost_per_million_goodput_output_tokens_usd",
            "fw_effective_uncached_tok_per_s",
            "sequoia_speculative_tok_per_s",
            "groq_total_e2e_latency_ms",
            "cerebras_measured_wafer_io_bw_mb_per_s",
        )
    )
    for case in result["cases"]:
        tr = case["training"]
        inf = case["inference"]
        inn = case["innovators"]
        mfu_str = "not-assessed-quantized" if tr["mfu_bf16_pct"] is None else f"{tr['mfu_bf16_pct']:.2f}"
        writer.writerow(
            (
                "own-offline",
                case["case_id"],
                mfu_str,
                f"{tr['cost_per_million_training_tokens_usd']:.4f}",
                tr["slurm_step_joules_per_gpu"],
                inf["disagg_total_gpus"],
                inf["colocated_total_gpus"],
                inf["gpu_savings_from_disagg"],
                f"{inf['naive_92pct_ttft_goodput_fraction']:.4f}",
                f"{inf['disagg_cost_per_million_goodput_output_tokens_usd']:.4f}",
                f"{inn['fireworks']['effective_uncached_tok_per_s']:.2f}",
                f"{inn['together_sequoia']['speculative_offload_tok_per_s']:.2f}",
                f"{inn['groq']['total_e2e_latency_ms']:.4f}",
                f"{inn['cerebras']['measured_wafer_io_bw_mb_per_s']:.2f}",
            )
        )
    return {
        "results.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "results.csv": stream.getvalue(),
    }


def main(argv: list[str] | None = None) -> int:
    """Regenerate the fixed outputs, or verify their exact bytes using --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "input.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads(args.input.read_text(encoding="utf-8"))
        outputs = render(fixture)
        if args.check:
            for name, content in outputs.items():
                path = HERE / name
                if not path.is_file() or path.read_bytes() != content.encode("utf-8"):
                    print(f"Experiment check failed: missing or different {name}", file=sys.stderr)
                    return 1
        else:
            for name, content in outputs.items():
                with (HERE / name).open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1
    print("own-offline capacity_economics outputs verified" if args.check else "own-offline capacity_economics outputs regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
