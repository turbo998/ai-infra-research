"""Evaluate 6D training parallelism (DP/TP/PP/CP/EP + FSDP2/ZeRO) state sharding, transient buffers, CP causal skew, and DeepEP compact dispatch."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import bytes_to_gib, integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import bytes_to_gib, integer


def _mib(byte_count: int) -> float:
    return byte_count / (2**20)


HERE = Path(__file__).resolve().parent
LIMITATIONS = [
    "own-offline synthetic arithmetic; no multi-node GPU cluster, NVLink/RoCE fabric, or Nsight profiling executed",
    "models static mesh invariants in pytorch/torchtitan@086bf6c166ec85c1298eb5596fa9bf95f6a2d840 torchtitan/distributed/parallel_dims.py and deepep.py",
    "models FSDP2 double-buffered prefetch, Shard(dim!=0) copy-out allocation, TP+SP redistribute, and CP Ring Attention load balancer in pytorch/pytorch@1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}: expected finite positive number")
    return float(value)


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: expected non-negative int")
    return value


def _deepep_compact_expected_sends(num_experts: int, ep_degree: int, top_k: int) -> float:
    """Compute expected unique destination ranks per token under uniform top-k routing without replacement.

    Reproduces the deduplication factor of TorchTitan DeepEP v2 training mode (`do_expand=False`),
    where `recv_x` transfers at most 1 row per unique token to each destination EP rank even if
    multiple local experts on that rank are selected.
    """
    if num_experts % ep_degree != 0:
        raise ValueError("num_experts must be divisible by ep_degree")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be in 1..num_experts")
    local_experts = num_experts // ep_degree
    if local_experts == 1:
        return float(top_k)
    total_ways = math.comb(num_experts, top_k)
    miss_ways = math.comb(num_experts - local_experts, top_k) if (num_experts - local_experts) >= top_k else 0
    hit_prob = 1.0 - (miss_ways / total_ways)
    return ep_degree * hit_prob


def evaluate_parallelism_case(
    *,
    case_id: str,
    world_size: int,
    pp: int,
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    ep: int = 1,
    etp: int = 1,
    sequence_parallel: bool = True,
    selective_activation_checkpointing: bool = True,
    fsdp_shard_dim: int = 0,
    num_layers: int,
    hidden_size: int,
    microbatch_size: int,
    num_microbatches: int,
    num_virtual_stages_per_rank: int = 1,
    seq_len: int,
    dense_params_per_layer: int,
    moe_routed_params_per_layer: int = 0,
    num_experts: int = 1,
    moe_top_k: int = 1,
    bytes_per_param_static: int = 18,
    param_dtype_bytes: int = 2,
    grad_reduce_dtype_bytes: int = 2,
    activation_dtype_bytes: int = 2,
    intra_node_bw_bytes_per_sec: int,
    inter_node_bw_bytes_per_sec: int,
    hbm_capacity_gib: float = 80.0,
) -> dict:
    """Evaluate 6D parallelism memory, transient FSDP buffers, CP causal skew, PP bubbles, and communication time."""
    label = _ascii_label(case_id, "case_id")
    ws = integer(world_size, "world_size")
    pp_deg = integer(pp, "pp")
    dp_rep = integer(dp_replicate, "dp_replicate")
    dp_sh = integer(dp_shard, "dp_shard")
    cp_deg = integer(cp, "cp")
    tp_deg = integer(tp, "tp")
    ep_deg = integer(ep, "ep")
    etp_deg = integer(etp, "etp")
    layers = integer(num_layers, "num_layers")
    hidden = integer(hidden_size, "hidden_size")
    mb_size = integer(microbatch_size, "microbatch_size")
    n_mb = integer(num_microbatches, "num_microbatches")
    v_stages = integer(num_virtual_stages_per_rank, "num_virtual_stages_per_rank")
    s_len = integer(seq_len, "seq_len")
    dense_p_layer = integer(dense_params_per_layer, "dense_params_per_layer")
    moe_p_layer = _non_negative_int(moe_routed_params_per_layer, "moe_routed_params_per_layer")
    experts = integer(num_experts, "num_experts")
    top_k = integer(moe_top_k, "moe_top_k")
    static_b = integer(bytes_per_param_static, "bytes_per_param_static")
    param_b = integer(param_dtype_bytes, "param_dtype_bytes")
    grad_b = integer(grad_reduce_dtype_bytes, "grad_reduce_dtype_bytes")
    act_b = integer(activation_dtype_bytes, "activation_dtype_bytes")
    intra_bw = integer(intra_node_bw_bytes_per_sec, "intra_node_bw_bytes_per_sec")
    inter_bw = integer(inter_node_bw_bytes_per_sec, "inter_node_bw_bytes_per_sec")
    cap_gib = _positive_float(hbm_capacity_gib, "hbm_capacity_gib")
    shard_dim = _non_negative_int(fsdp_shard_dim, "fsdp_shard_dim")

    for flag_name, flag_val in (
        ("sequence_parallel", sequence_parallel),
        ("selective_activation_checkpointing", selective_activation_checkpointing),
    ):
        if type(flag_val) is not bool:
            raise ValueError(f"{flag_name}: expected bool")

    # 1. Validate TorchTitan ParallelDims invariants (parallel_dims.py:186-218).
    if pp_deg * dp_rep * dp_sh * cp_deg * tp_deg != ws:
        raise ValueError(
            f"Invalid 6D mesh product: pp({pp_deg})*dp_replicate({dp_rep})*dp_shard({dp_sh})*cp({cp_deg})*tp({tp_deg}) != world_size({ws})"
        )
    if etp_deg not in (1, tp_deg):
        raise ValueError("etp must be equal to 1 or tp")
    if layers % (pp_deg * v_stages) != 0:
        raise ValueError("num_layers must be divisible by pp * num_virtual_stages_per_rank")
    if s_len % (2 * cp_deg * (tp_deg if sequence_parallel else 1)) != 0:
        raise ValueError("seq_len must be divisible by 2 * cp * (tp if sequence_parallel else 1)")

    batch_mesh_size = dp_rep * dp_sh
    loss_mesh_size = dp_rep * dp_sh * cp_deg
    fsdp_mesh_size = dp_sh * cp_deg  # parallel_dims.py:279: fsdp = dp_shard * cp
    dense_shard_divisor = fsdp_mesh_size * tp_deg

    if ep_deg > 1:
        sparse_pool = fsdp_mesh_size * (tp_deg // etp_deg)
        if sparse_pool % ep_deg != 0:
            raise ValueError(f"(dp_shard * cp * (tp // etp))={sparse_pool} must be divisible by ep={ep_deg}")
        efsdp_mesh_size = sparse_pool // ep_deg
    else:
        efsdp_mesh_size = fsdp_mesh_size
    moe_shard_divisor = efsdp_mesh_size * ep_deg * etp_deg

    layers_per_pp_rank = layers // pp_deg

    # 2. Static state memory per rank (Parameters + Gradients + Optimizer states).
    dense_params_per_pp_rank = layers_per_pp_rank * dense_p_layer
    moe_params_per_pp_rank = layers_per_pp_rank * moe_p_layer
    static_dense_bytes = (dense_params_per_pp_rank * static_b) // dense_shard_divisor
    static_moe_bytes = (moe_params_per_pp_rank * static_b) // moe_shard_divisor if moe_p_layer > 0 else 0
    static_state_bytes = static_dense_bytes + static_moe_bytes

    # 3. FSDP2 transient buffer peak (_fsdp_param_group.py:271-307 & _fsdp_collectives.py:274-316).
    # Unsharded parameter size for one transformer layer on a TP rank:
    unsharded_layer_dense_bytes = (dense_p_layer * param_b) // tp_deg
    unsharded_layer_moe_bytes = (moe_p_layer * param_b) // (ep_deg * etp_deg) if moe_p_layer > 0 else 0
    unsharded_layer_param_bytes = unsharded_layer_dense_bytes + unsharded_layer_moe_bytes

    # Standard FSDP2 double-buffered prefetch holds:
    # 1x current unsharded_param + 2x all_gather_output buffers (current in copy_out + next prefetched) = 3x layer.
    # If fsdp_shard_dim != 0, foreach_all_gather_copy_out allocates an extra torch.empty_like(t) buffer (+1x) and runs chunk+cat.
    non_dim0_extra_copy_out_bytes = unsharded_layer_param_bytes if shard_dim != 0 else 0
    fsdp_transient_peak_bytes = (3 * unsharded_layer_param_bytes) + non_dim0_extra_copy_out_bytes

    # 4. Dynamic activation memory per rank.
    # Under selective activation checkpointing + FlashAttention, per-layer activation is proportional to B * S * h.
    # Standard full activation without checkpointing stores ~34 * B * S * h bytes (for BF16=2B, 17 * B * S * h elements);
    # with selective activation checkpointing (recomputing attention score/softmax), ~10 * B * S * h bytes (5 * B * S * h elements).
    # Without Sequence Parallel (SP), LayerNorm/dropout/residual activations are replicated across TP; with SP they are divided by TP.
    act_elements_multiplier = 5 if selective_activation_checkpointing else 17
    tp_act_divisor = tp_deg if sequence_parallel else max(1, tp_deg // 2)
    inflight_microbatches = pp_deg if pp_deg > 1 else 1
    per_layer_activation_bytes = (
        mb_size * s_len * hidden * act_elements_multiplier * act_b
    ) // (cp_deg * tp_act_divisor)
    total_activation_bytes = layers_per_pp_rank * per_layer_activation_bytes * inflight_microbatches

    peak_memory_bytes = static_state_bytes + fsdp_transient_peak_bytes + total_activation_bytes
    peak_memory_gib = round(bytes_to_gib(peak_memory_bytes), 4)
    exceeds_hbm_capacity = peak_memory_gib > cap_gib

    # 5. Context Parallel (CP) causal mask load-balance analysis (_attention.py:495-516, 1144-1191).
    # Contiguous partitioning: rank r computes (r + 0.5) * (S/CP)^2 causal attention elements;
    # max rank (CP - 1) vs rank 0 ratio is (2*CP - 1) : 1, and max-to-mean ratio is (2*CP - 1) / CP.
    cp_contiguous_max_to_min_skew = float(2 * cp_deg - 1) if cp_deg > 1 else 1.0
    cp_contiguous_max_to_mean_ratio = round((2 * cp_deg - 1) / cp_deg, 6) if cp_deg > 1 else 1.0
    cp_round_robin_skew = 1.0

    # 6. Pipeline Parallel (PP) bubble fraction (pipeline_parallel.py:261-318).
    total_virtual_stages = pp_deg * v_stages
    pp_bubble_fraction = (
        round((pp_deg - 1) / (pp_deg - 1 + v_stages * n_mb), 6) if pp_deg > 1 else 0.0
    )

    # 7. DeepEP v2 MoE All-to-All dispatch/combine volume (deepep.py:244-395).
    tokens_per_rank_per_mb = (mb_size * s_len) // (cp_deg * (tp_deg if sequence_parallel else 1))
    if ep_deg > 1 and moe_p_layer > 0:
        expected_compact_sends = _deepep_compact_expected_sends(experts, ep_deg, top_k)
        dedup_savings_ratio = round(1.0 - (expected_compact_sends / top_k), 6)
        # Forward dispatch + combine + backward dispatch + combine = 4 transfers per layer per microbatch
        expanded_ep_bytes_per_step = int(
            round(n_mb * layers_per_pp_rank * 4 * tokens_per_rank_per_mb * top_k * hidden * act_b)
        )
        compact_ep_bytes_per_step = int(
            round(n_mb * layers_per_pp_rank * 4 * tokens_per_rank_per_mb * expected_compact_sends * hidden * act_b)
        )
    else:
        expected_compact_sends = 0.0
        dedup_savings_ratio = 0.0
        expanded_ep_bytes_per_step = 0
        compact_ep_bytes_per_step = 0

    # 8. Communication volume & exposed time estimation per training step.
    # TP/SP collective volume per step (forward + backward = 8 * (tp-1)/tp * B * S / cp * h * act_b per layer).
    tp_comm_bytes_per_step = (
        int(
            round(
                n_mb
                * layers_per_pp_rank
                * 8
                * ((tp_deg - 1) / tp_deg)
                * (mb_size * s_len / cp_deg)
                * hidden
                * act_b
            )
        )
        if tp_deg > 1
        else 0
    )

    # FSDP2 collective volume per step: Forward AllGather + Backward AllGather + Backward ReduceScatter
    # Dense layers use fsdp_mesh_size; MoE layers use efsdp_mesh_size.
    fsdp_dense_factor = (fsdp_mesh_size - 1) / fsdp_mesh_size if fsdp_mesh_size > 1 else 0.0
    efsdp_moe_factor = (efsdp_mesh_size - 1) / efsdp_mesh_size if efsdp_mesh_size > 1 else 0.0
    fsdp_bytes_per_step = int(
        round(
            layers_per_pp_rank
            * (
                fsdp_dense_factor * (2 * unsharded_layer_dense_bytes + (dense_p_layer * grad_b) // tp_deg)
                + efsdp_moe_factor
                * (2 * unsharded_layer_moe_bytes + (moe_p_layer * grad_b) // (ep_deg * etp_deg))
            )
        )
    )

    # HSDP cross-replica AllReduce at step end if dp_replicate > 1:
    hsdp_allreduce_bytes = (
        int(
            round(
                2
                * ((dp_rep - 1) / dp_rep)
                * (
                    (dense_params_per_pp_rank * grad_b) // dense_shard_divisor
                    + (moe_params_per_pp_rank * grad_b) // moe_shard_divisor
                )
            )
        )
        if dp_rep > 1
        else 0
    )

    # Determine whether FSDP collectives traverse inter-node links (when tp * fsdp_mesh_size > 8 GPUs per node).
    fsdp_bw = inter_bw if (tp_deg * fsdp_mesh_size) > 8 else intra_bw
    ep_bw = inter_bw if (etp_deg * ep_deg) > 8 else intra_bw

    tp_comm_ms = (tp_comm_bytes_per_step / intra_bw) * 1000.0
    fsdp_comm_ms = (fsdp_bytes_per_step / fsdp_bw) * 1000.0
    hsdp_comm_ms = (hsdp_allreduce_bytes / inter_bw) * 1000.0
    ep_compact_comm_ms = (compact_ep_bytes_per_step / ep_bw) * 1000.0
    total_comm_ms = round(tp_comm_ms + fsdp_comm_ms + hsdp_comm_ms + ep_compact_comm_ms, 2)

    return {
        "case_id": label,
        "world_size": ws,
        "mesh": {
            "pp": pp_deg,
            "dp_replicate": dp_rep,
            "dp_shard": dp_sh,
            "cp": cp_deg,
            "tp": tp_deg,
            "ep": ep_deg,
            "etp": etp_deg,
            "batch_mesh_size": batch_mesh_size,
            "loss_mesh_size": loss_mesh_size,
            "fsdp_mesh_size": fsdp_mesh_size,
            "efsdp_mesh_size": efsdp_mesh_size,
            "dense_shard_divisor": dense_shard_divisor,
            "moe_shard_divisor": moe_shard_divisor,
        },
        "memory_breakdown": {
            "static_state_gib": round(bytes_to_gib(static_state_bytes), 4),
            "fsdp_transient_peak_gib": round(bytes_to_gib(fsdp_transient_peak_bytes), 4),
            "non_dim0_extra_copy_out_mib": round(_mib(non_dim0_extra_copy_out_bytes), 2),
            "activation_gib": round(bytes_to_gib(total_activation_bytes), 4),
            "peak_memory_gib": peak_memory_gib,
            "exceeds_hbm_capacity": exceeds_hbm_capacity,
        },
        "cp_causal_balance": {
            "cp_contiguous_max_to_min_skew": cp_contiguous_max_to_min_skew,
            "cp_contiguous_max_to_mean_ratio": cp_contiguous_max_to_mean_ratio,
            "cp_round_robin_skew": cp_round_robin_skew,
        },
        "pipeline_metrics": {
            "total_virtual_stages": total_virtual_stages,
            "pp_bubble_fraction": pp_bubble_fraction,
        },
        "deepep_metrics": {
            "expected_compact_sends_per_token": round(expected_compact_sends, 4),
            "dedup_savings_ratio": dedup_savings_ratio,
            "expanded_ep_bytes_per_step": expanded_ep_bytes_per_step,
            "compact_ep_bytes_per_step": compact_ep_bytes_per_step,
        },
        "communication_Step_metrics": {
            "tp_comm_bytes_per_step": tp_comm_bytes_per_step,
            "fsdp_bytes_per_step": fsdp_bytes_per_step,
            "hsdp_allreduce_bytes": hsdp_allreduce_bytes,
            "total_comm_ms": total_comm_ms,
        },
    }


def calculate(fixture: dict) -> dict:
    if not isinstance(fixture, dict) or not isinstance(fixture.get("cases"), list) or not fixture["cases"]:
        raise ValueError("fixture.cases: expected non-empty list")
    results = []
    for item in fixture["cases"]:
        if not isinstance(item, dict):
            raise ValueError("case entry must be a dict")
        res = evaluate_parallelism_case(**item)
        results.append({"input": item, "result": res})
    return {
        "experiment_id": "training_parallelism",
        "evidence_status": "own-offline",
        "limitations": LIMITATIONS,
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "input.json")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--check", action="store_true", help="Verify results.json matches input.json calculation")
    args = parser.parse_args(argv)

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    computed = calculate(fixture)
    rendered = json.dumps(computed, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        expected = args.output.read_text(encoding="utf-8")
        if rendered != expected:
            print("Mismatch in experiments/training_parallelism/results.json; re-run without --check.", file=sys.stderr)
            return 1
        print("training_parallelism check passed.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
