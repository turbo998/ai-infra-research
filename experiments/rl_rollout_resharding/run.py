"""CPU-only offline model for RL post-training weight resharding, CuMem sleep/wake, and Long-CoT rollout stragglers.

This experiment does not run distributed GPU training or vLLM servers. It evaluates:
1. Weight resharding and transfer latency across five `volcengine/verl` CheckpointEngine and IPC regimes
   for a 671B-parameter MoE model (`verl/checkpoint_engine/nccl_checkpoint_engine.py`,
   `verl/checkpoint_engine/delta_checkpoint_engine.py`, `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`,
   and `delta_weight_transfer.py`):
   - `colocated_cuda_ipc_bucketed_512mb`: Same-GPU CUDA IPC (`BucketedWeightSender`, 512 MiB double buffer)
   - `colocated_shm_fallback_512mb`: Host shared-memory fallback (`use_shm=True` when IPC is unavailable)
   - `disaggregated_nccl_single_sender`: Cross-node NCCL broadcast from actor rank 0 only (`multi_sender=False`)
   - `disaggregated_nccl_multi_sender_nvlink`: NVLink-local actor relays widening root fan-out (`multi_sender=True`)
   - `disaggregated_delta_sparse_indices`: Sparse `(int32 idx, bf16 val)` `DeltaFlush` (`MAX_ENTRY_ELEMS`, `4 + element_size()` bytes/elem)
2. Colocated RL GPU HBM peak footprint and transition latency across `No Sleep` (OOM on 80 GiB H100),
   `vLLM CuMemAllocator Sleep Level 1` (weights offloaded to pinned CPU RAM via `cudaMemcpy` + `python_unmap_and_release`,
   KV cache discarded), and `Sleep Level 2` (both weights and KV cache unmapped/discarded, weights refit directly from trainer).
3. Long-CoT GRPO rollout tail-straggler bubble across synchronous micro-batch barriers vs partial rollout preemption /
   continuous async token-budget scheduling.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"


def load_input(path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_weight_transfer_modes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    total_params_b = float(spec["model_params_billions"])
    active_params_b = float(spec["active_params_billions"])
    rollout_replicas = int(spec["rollout_tp_ep_group_gpus"])
    bf16_bpe = 2
    int32_idx_bytes = 4

    full_model_bytes = int(total_params_b * 1e9 * bf16_bpe)
    full_model_gib = full_model_bytes / (1024 ** 3)
    per_gpu_shard_bytes = full_model_bytes // rollout_replicas
    bucket_size_bytes = int(spec["bucket_size_mb"]) << 20

    rows: list[dict[str, Any]] = []
    for mode in spec["transfer_modes"]:
        mode_id = mode["id"]
        transport = mode["transport"]
        effective_bw_gbps = float(mode["effective_bw_gbps"])
        sync_overhead_ms = float(mode["fixed_control_overhead_ms"])

        if transport == "colocated_cuda_ipc":
            # Same-GPU IPC copies local shard through 512 MiB bucket buffer
            wire_bytes = per_gpu_shard_bytes
            receiver_staging_mib = (bucket_size_bytes) / (1024 ** 2)
            num_buckets = (wire_bytes + bucket_size_bytes - 1) // bucket_size_bytes
            transfer_sec = (wire_bytes / (effective_bw_gbps * 1e9)) + (num_buckets * sync_overhead_ms / 1000.0)
        elif transport == "colocated_shm_fallback":
            # D2H + H2D through POSIX shared memory (/dev/shm)
            wire_bytes = per_gpu_shard_bytes * 2
            receiver_staging_mib = (bucket_size_bytes * 2) / (1024 ** 2)
            num_buckets = (per_gpu_shard_bytes + bucket_size_bytes - 1) // bucket_size_bytes
            transfer_sec = (wire_bytes / (effective_bw_gbps * 1e9)) + (num_buckets * sync_overhead_ms / 1000.0)
        elif transport == "disaggregated_nccl_single":
            # Actor rank 0 broadcasts entire model over single NIC egress
            wire_bytes = full_model_bytes
            receiver_staging_mib = (2 * bucket_size_bytes) / (1024 ** 2)
            num_buckets = (wire_bytes + bucket_size_bytes - 1) // bucket_size_bytes
            transfer_sec = (wire_bytes / (effective_bw_gbps * 1e9)) + (num_buckets * sync_overhead_ms / 1000.0)
        elif transport == "disaggregated_nccl_multi_sender":
            # 8 NVLink-local actor ranks act as parallel NIC senders
            num_senders = int(mode["nvlink_local_senders"])
            wire_bytes = full_model_bytes
            receiver_staging_mib = (2 * bucket_size_bytes) / (1024 ** 2)
            num_buckets = (wire_bytes + bucket_size_bytes - 1) // bucket_size_bytes
            transfer_sec = (wire_bytes / (effective_bw_gbps * num_senders * 1e9)) + (
                (num_buckets / num_senders) * sync_overhead_ms / 1000.0
            )
        elif transport == "disaggregated_delta_indices":
            # Sparse DeltaFlush: changed fraction of parameters encoded as (int32 pos + bf16 val) = 6 bytes/elem
            changed_fraction = float(mode["changed_param_fraction"])
            changed_elems = int(total_params_b * 1e9 * changed_fraction)
            wire_bytes = changed_elems * (int32_idx_bytes + bf16_bpe)
            num_senders = int(mode["nvlink_local_senders"])
            # Receiver peak transient is 1 bucket + 1 decode chunk (MAX_ENTRY_ELEMS * 6 B)
            max_entry_elems = int(mode["max_entry_elems"])
            decode_chunk_bytes = max_entry_elems * (int32_idx_bytes + bf16_bpe)
            receiver_staging_mib = (2 * bucket_size_bytes + decode_chunk_bytes) / (1024 ** 2)
            num_buckets = max(1, (wire_bytes + bucket_size_bytes - 1) // bucket_size_bytes)
            transfer_sec = (wire_bytes / (effective_bw_gbps * num_senders * 1e9)) + (
                num_buckets * sync_overhead_ms / 1000.0
            )
        else:
            raise ValueError(f"Unknown transport: {transport}")

        rows.append(
            {
                "id": mode_id,
                "label": mode["label"],
                "transport": transport,
                "full_model_gib": round(full_model_gib, 4),
                "wire_volume_gib": round(wire_bytes / (1024 ** 3), 4),
                "receiver_peak_staging_mib": round(receiver_staging_mib, 4),
                "sync_latency_sec": round(transfer_sec, 4),
                "speedup_vs_single_sender_nccl": 0.0,  # populated below
            }
        )

    baseline_sec = next(r["sync_latency_sec"] for r in rows if r["id"] == "disaggregated_nccl_single_sender")
    for row in rows:
        row["speedup_vs_single_sender_nccl"] = round(baseline_sec / row["sync_latency_sec"], 4)

    return rows


def evaluate_cumem_sleep_regimes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    gpu_hbm_capacity_gib = float(spec["gpu_hbm_capacity_gib"])
    trainer_states_gib = float(spec["trainer_fsdp_states_gib_per_gpu"])
    rollout_weights_gib = float(spec["rollout_weights_gib_per_gpu"])
    rollout_kv_cache_gib = float(spec["rollout_kv_cache_gib_per_gpu"])
    pcie_bw_gbps = float(spec["pcie_pinned_memcpy_bw_gbps"])
    vm_unmap_map_ms = float(spec["cumem_unmap_map_overhead_ms"])

    regimes: list[dict[str, Any]] = []
    for item in spec["sleep_regimes"]:
        rid = item["id"]
        if rid == "no_sleep_co_resident":
            peak_training_hbm_gib = trainer_states_gib + rollout_weights_gib + rollout_kv_cache_gib
            cpu_pinned_backup_gib = 0.0
            sleep_wake_sec = 0.0
        elif rid == "sleep_level_1_offload_weights":
            # CuMemAllocator.sleep(offload_tags=("weights",)): copies weights to pinned CPU tensor, unmaps weights + kv_cache
            peak_training_hbm_gib = trainer_states_gib
            cpu_pinned_backup_gib = rollout_weights_gib
            bytes_copied = rollout_weights_gib * (1024 ** 3) * 2.0  # D2H on sleep + H2D on wake_up
            sleep_wake_sec = (bytes_copied / (pcie_bw_gbps * 1e9)) + (vm_unmap_map_ms / 1000.0)
        elif rid == "sleep_level_2_discard_all_direct_refit":
            # CuMemAllocator.sleep(offload_tags=()): unmaps both weights & kv_cache without D2H copy; wake_up remaps + IPC refit
            peak_training_hbm_gib = trainer_states_gib
            cpu_pinned_backup_gib = 0.0
            sleep_wake_sec = vm_unmap_map_ms / 1000.0
        else:
            raise ValueError(f"Unknown sleep regime: {rid}")

        fits_in_80gib = peak_training_hbm_gib <= gpu_hbm_capacity_gib
        headroom_gib = gpu_hbm_capacity_gib - peak_training_hbm_gib

        regimes.append(
            {
                "id": rid,
                "label": item["label"],
                "peak_training_hbm_gib": round(peak_training_hbm_gib, 4),
                "hbm_headroom_vs_80gib": round(headroom_gib, 4),
                "fits_on_80gib_gpu": fits_in_80gib,
                "cpu_pinned_backup_gib_per_gpu": round(cpu_pinned_backup_gib, 4),
                "sleep_wake_transition_sec": round(sleep_wake_sec, 4),
            }
        )
    return regimes


def evaluate_long_cot_straggler_scheduling(spec: dict[str, Any]) -> list[dict[str, Any]]:
    decode_tps_per_req = float(spec["decode_tokens_per_sec_per_req"])
    length_buckets = spec["prompt_group_response_lengths"]
    total_tokens = sum(int(b["count"]) * int(b["response_tokens"]) for b in length_buckets)
    max_tokens = max(int(b["response_tokens"]) for b in length_buckets)
    total_reqs = sum(int(b["count"]) for b in length_buckets)
    ideal_gpu_token_slots = total_reqs * max_tokens
    sync_utilization = total_tokens / ideal_gpu_token_slots
    sync_wall_sec = max_tokens / decode_tps_per_req
    sync_bubble_ratio = 1.0 - sync_utilization

    rows: list[dict[str, Any]] = []
    for sched in spec["schedulers"]:
        sid = sched["id"]
        if sid == "sync_barrier_grpo":
            wall_sec = sync_wall_sec
            bubble_ratio = sync_bubble_ratio
            effective_tps_ratio = sync_utilization
        elif sid == "partial_rollout_checkpoint_recycle":
            cap_tokens = int(sched["truncation_cap_tokens"])
            capped_total = sum(int(b["count"]) * min(int(b["response_tokens"]), cap_tokens) for b in length_buckets)
            capped_max = min(max_tokens, cap_tokens)
            util = capped_total / (total_reqs * capped_max)
            wall_sec = capped_max / decode_tps_per_req
            bubble_ratio = 1.0 - util
            effective_tps_ratio = util
        elif sid == "async_continuous_staleness_bounded":
            overhead = float(sched["async_queue_drain_bubble_fraction"])
            bubble_ratio = overhead
            effective_tps_ratio = 1.0 - overhead
            wall_sec = (total_tokens / total_reqs) / (decode_tps_per_req * effective_tps_ratio)
        else:
            raise ValueError(f"Unknown scheduler: {sid}")

        rows.append(
            {
                "id": sid,
                "label": sched["label"],
                "stage_wall_time_sec": round(wall_sec, 4),
                "gpu_idle_bubble_fraction": round(bubble_ratio, 4),
                "effective_compute_utilization": round(effective_tps_ratio, 4),
                "speedup_vs_sync_barrier": round(sync_wall_sec / wall_sec, 4),
            }
        )
    return rows


def run_experiment(spec: dict[str, Any]) -> dict[str, Any]:
    transfer_rows = evaluate_weight_transfer_modes(spec["weight_transfer_section"])
    sleep_rows = evaluate_cumem_sleep_regimes(spec["cumem_sleep_section"])
    straggler_rows = evaluate_long_cot_straggler_scheduling(spec["straggler_section"])

    single_nccl = next(r for r in transfer_rows if r["id"] == "disaggregated_nccl_single_sender")
    multi_nccl = next(r for r in transfer_rows if r["id"] == "disaggregated_nccl_multi_sender_nvlink")
    delta_ipc = next(r for r in transfer_rows if r["id"] == "disaggregated_delta_sparse_indices")
    coloc_ipc = next(r for r in transfer_rows if r["id"] == "colocated_cuda_ipc_bucketed_512mb")

    no_sleep = next(r for r in sleep_rows if r["id"] == "no_sleep_co_resident")
    sleep_l1 = next(r for r in sleep_rows if r["id"] == "sleep_level_1_offload_weights")
    sleep_l2 = next(r for r in sleep_rows if r["id"] == "sleep_level_2_discard_all_direct_refit")

    sync_sched = next(r for r in straggler_rows if r["id"] == "sync_barrier_grpo")
    async_sched = next(r for r in straggler_rows if r["id"] == "async_continuous_staleness_bounded")

    return {
        "experiment_id": "rl_rollout_resharding",
        "measurement_type": "own-offline",
        "Clear_Boundary": "CPU-only analytical model derived from volcengine/verl and vllm-project/vllm source code; not a live cluster RL run.",
        "weight_transfer_comparison": transfer_rows,
        "cumem_sleep_comparison": sleep_rows,
        "long_cot_straggler_comparison": straggler_rows,
        "headline_findings": {
            "single_sender_nccl_sync_sec": single_nccl["sync_latency_sec"],
            "multi_sender_nccl_sync_sec": multi_nccl["sync_latency_sec"],
            "sparse_delta_sync_sec": delta_ipc["sync_latency_sec"],
            "colocated_cuda_ipc_sync_sec": coloc_ipc["sync_latency_sec"],
            "no_sleep_peak_hbm_gib": no_sleep["peak_training_hbm_gib"],
            "sleep_l1_peak_hbm_gib": sleep_l1["peak_training_hbm_gib"],
            "sleep_l1_transition_sec": sleep_l1["sleep_wake_transition_sec"],
            "sleep_l2_transition_sec": sleep_l2["sleep_wake_transition_sec"],
            "sync_grpo_idle_bubble_fraction": sync_sched["gpu_idle_bubble_fraction"],
            "async_grpo_idle_bubble_fraction": async_sched["gpu_idle_bubble_fraction"],
        },
    }


def write_csv(results: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "id",
        "transport",
        "full_model_gib",
        "wire_volume_gib",
        "receiver_peak_staging_mib",
        "sync_latency_sec",
        "speedup_vs_single_sender_nccl",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["weight_transfer_comparison"]:
            writer.writerow({k: row[k] for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    spec = load_input(args.input)
    results = run_experiment(spec)

    if args.check:
        expected = json.loads(args.output_json.read_text(encoding="utf-8"))
        if results != expected:
            raise SystemExit("rl_rollout_resharding results.json is out of date; re-run without --check.")
        print("rl_rollout_resharding check passed.")
        return 0

    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(results, args.output_csv)
    print(f"Wrote {args.output_json} and {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
