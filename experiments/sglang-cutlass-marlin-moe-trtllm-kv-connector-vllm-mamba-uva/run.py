from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary.json"


def simulate_cutlass_prepare_moe_input(
    *,
    topk_ids: list[int],
    num_experts: int,
    n_dim: int,
    k_dim: int,
    topk: int,
    input_map_capacity: int,
    transpose_problem_sizes: bool = False,
) -> dict[str, Any]:
    """Model SGLang PR #45917 (`ac3db731b3`): `cutlass_prepare_moe_input` complete metadata & blockscale offsets."""
    counts = [0] * num_experts
    for eid in topk_ids:
        if 0 <= eid < num_experts:
            counts[eid] += 1

    expert_offsets = [0]
    blockscale_offsets = [0]
    for c in counts:
        expert_offsets.append(expert_offsets[-1] + c)
        blockscale_offsets.append(blockscale_offsets[-1] + ((c + 127) // 128) * 128)

    problem_sizes1 = []
    problem_sizes2 = []
    for c in counts:
        if not transpose_problem_sizes:
            problem_sizes1.append([c, 2 * n_dim, k_dim])
            problem_sizes2.append([c, k_dim, n_dim])
        else:
            problem_sizes1.append([2 * n_dim, c, k_dim])
            problem_sizes2.append([k_dim, c, n_dim])

    write_cursors = list(expert_offsets[:-1])
    num_routes = len(topk_ids)
    input_permutation = [0] * max(input_map_capacity, expert_offsets[-1])
    output_permutation = [-1] * num_routes

    for route_idx, eid in enumerate(topk_ids):
        if 0 <= eid < num_experts:
            slot = write_cursors[eid]
            write_cursors[eid] += 1
            input_permutation[slot] = route_idx // topk
            output_permutation[route_idx] = slot
        else:
            output_permutation[route_idx] = -1

    # Tail slots beyond expert_offsets[-1] point at token 0 for safe full gathers
    for slot in range(expert_offsets[-1], len(input_permutation)):
        input_permutation[slot] = 0

    return {
        "num_routes": num_routes,
        "active_routes": expert_offsets[-1],
        "inactive_routes": sum(1 for x in output_permutation if x == -1),
        "expert_offsets": expert_offsets,
        "blockscale_offsets": blockscale_offsets,
        "problem_sizes1": problem_sizes1,
        "problem_sizes2": problem_sizes2,
        "input_permutation": input_permutation,
        "output_permutation": output_permutation,
        "tail_padding_points_to_token_zero": all(
            x == 0 for x in input_permutation[expert_offsets[-1] :]
        ),
    }


def simulate_marlin_inactive_block_zeroing(
    *,
    expert_ids: list[int],
    sorted_token_ids: list[int],
    moe_block_size: int,
    num_routes: int,
) -> dict[str, Any]:
    """Model SGLang PR #45462 (`8b9b992e41`): `zero_inactive_output_rows` in Marlin MoE."""
    zeroed_routes: list[int] = []
    active_computed_routes: list[int] = []
    for block_idx, eid in enumerate(expert_ids):
        block_tokens = sorted_token_ids[
            block_idx * moe_block_size : (block_idx + 1) * moe_block_size
        ]
        if eid == -1:
            for route in block_tokens:
                if route < num_routes:
                    zeroed_routes.append(route)
        else:
            for route in block_tokens:
                if route < num_routes:
                    active_computed_routes.append(route)

    return {
        "num_blocks": len(expert_ids),
        "inactive_blocks": sum(1 for e in expert_ids if e == -1),
        "zeroed_routes": sorted(zeroed_routes),
        "active_computed_routes": sorted(active_computed_routes),
        "all_valid_routes_covered": sorted(zeroed_routes + active_computed_routes)
        == list(range(num_routes)),
    }


def verify_sglang_decoder_tail_token_count_ownership(
    *,
    local_rows: int,
    global_rows: int,
    attn_tp_sequence_sharded: bool,
    bound_kind: str,
) -> dict[str, Any]:
    """Model SGLang PR #45915 (`1340044e97`): `_pack_token_counts` and `_can_capture_moe_bound`."""
    packed = (global_rows << 32) | local_rows
    unpacked_local = packed & 0xFFFFFFFF
    unpacked_global = packed >> 32
    can_capture = (
        bound_kind == "none"
        or bound_kind == "local"
        or (attn_tp_sequence_sharded and bound_kind == "global")
    )
    return {
        "local_rows": local_rows,
        "global_rows": global_rows,
        "packed_uint64": packed,
        "unpacked_matches": (unpacked_local == local_rows and unpacked_global == global_rows),
        "attn_tp_sequence_sharded": attn_tp_sequence_sharded,
        "bound_kind": bound_kind,
        "can_capture_moe_bound": can_capture,
    }


def compute_trtllm_connector_kv_reuse(
    *,
    token_begin: int,
    token_end: int,
    tokens_per_block: int,
    total_prompt_tokens: int,
) -> dict[str, Any]:
    """Model TRT-LLM PR #20180 (`2706f4dfb6`): `KvCache::recordConnectorReuse(tokenBegin, tokenEnd)`."""
    if token_begin >= token_end:
        return {
            "token_begin": token_begin,
            "token_end": token_end,
            "tokens_per_block": tokens_per_block,
            "block_begin": 0,
            "block_end": 0,
            "reused_full_blocks": 0,
            "partial_boundary_blocks_kept_as_miss": 0,
            "connector_cached_tokens": 0,
        }
    block_begin = (token_begin + tokens_per_block - 1) // tokens_per_block
    block_end = token_end // tokens_per_block
    reused_full_blocks = max(0, block_end - block_begin)

    # Any block touched by [token_begin, token_end) that is not completely filled stays a miss
    first_touched_block = token_begin // tokens_per_block
    last_touched_block = (token_end - 1) // tokens_per_block
    total_touched_blocks = last_touched_block - first_touched_block + 1
    partial_boundary_blocks = total_touched_blocks - reused_full_blocks

    return {
        "token_begin": token_begin,
        "token_end": token_end,
        "tokens_per_block": tokens_per_block,
        "total_prompt_tokens": total_prompt_tokens,
        "block_begin": block_begin,
        "block_end": block_end,
        "reused_full_blocks": reused_full_blocks,
        "total_touched_blocks": total_touched_blocks,
        "partial_boundary_blocks_kept_as_miss": partial_boundary_blocks,
        "connector_cached_tokens": token_end - token_begin,
    }


def simulate_vllm_mamba_sibling_tail_boundary(
    *,
    num_siblings: int,
    block_hash_already_committed: bool,
) -> dict[str, Any]:
    """Model vLLM PR #62841 (`d561af5dfc`): `_cache_partial_tail_block` committed boundary guard."""
    # Without fix: re-caching marks the block hash as cached "this step", deferring every sibling to subsequent steps
    steps_without_fix = num_siblings if block_hash_already_committed else 1
    # With fix: `if source_block.block_hash_num_tokens == num_tokens: return None`
    steps_with_fix = 1
    return {
        "num_siblings": num_siblings,
        "block_hash_already_committed": block_hash_already_committed,
        "engine_steps_without_fix": steps_without_fix,
        "engine_steps_with_fix": steps_with_fix,
        "step_reduction_factor": round(steps_without_fix / steps_with_fix, 2),
    }


def simulate_vllm_uva_host_resident_moe_routing(
    *,
    num_tokens: int,
    top_k: int,
    global_num_experts: int,
    unique_active_experts: int,
    expert_weight_mib: float,
    weights_host_resident: bool,
) -> dict[str, Any]:
    """Model vLLM PR #62054 (`3b364baef5`): `_prepare_expert_assignment` forces sorted path for UVA weights."""
    total_pairs = num_tokens * top_k
    sparsity_heuristic = total_pairs * 4 <= global_num_experts

    buggy_naive_assignment = sparsity_heuristic
    fixed_naive_assignment = sparsity_heuristic and (not weights_host_resident)

    buggy_pcie_reads = total_pairs if buggy_naive_assignment else unique_active_experts
    fixed_pcie_reads = total_pairs if fixed_naive_assignment else unique_active_experts

    return {
        "num_tokens": num_tokens,
        "top_k": top_k,
        "global_num_experts": global_num_experts,
        "unique_active_experts": unique_active_experts,
        "weights_host_resident": weights_host_resident,
        "buggy_uses_naive_path": buggy_naive_assignment,
        "fixed_uses_naive_path": fixed_naive_assignment,
        "buggy_pcie_expert_reads": buggy_pcie_reads,
        "fixed_pcie_expert_reads": fixed_pcie_reads,
        "buggy_pcie_traffic_mib": round(buggy_pcie_reads * expert_weight_mib, 2),
        "fixed_pcie_traffic_mib": round(fixed_pcie_reads * expert_weight_mib, 2),
        "pcie_traffic_reduction_x": round(buggy_pcie_reads / fixed_pcie_reads, 3),
    }


def build_summary() -> dict[str, Any]:
    cutlass_case = simulate_cutlass_prepare_moe_input(
        topk_ids=[0, 2, -1, 0, 1, 2, -1, 0],
        num_experts=4,
        n_dim=2048,
        k_dim=4096,
        topk=2,
        input_map_capacity=8,
        transpose_problem_sizes=False,
    )
    marlin_case = simulate_marlin_inactive_block_zeroing(
        expert_ids=[0, -1, 2],
        sorted_token_ids=[0, 3, 1, 4, 2, 5],
        moe_block_size=2,
        num_routes=6,
    )
    tail_pack_sharded = verify_sglang_decoder_tail_token_count_ownership(
        local_rows=16,
        global_rows=128,
        attn_tp_sequence_sharded=True,
        bound_kind="global",
    )
    tail_pack_unsharded = verify_sglang_decoder_tail_token_count_ownership(
        local_rows=16,
        global_rows=128,
        attn_tp_sequence_sharded=False,
        bound_kind="global",
    )
    trtllm_aligned = compute_trtllm_connector_kv_reuse(
        token_begin=0,
        token_end=256,
        tokens_per_block=64,
        total_prompt_tokens=320,
    )
    trtllm_unaligned = compute_trtllm_connector_kv_reuse(
        token_begin=16,
        token_end=240,
        tokens_per_block=64,
        total_prompt_tokens=320,
    )
    mamba_siblings = simulate_vllm_mamba_sibling_tail_boundary(
        num_siblings=8,
        block_hash_already_committed=True,
    )
    uva_moe = simulate_vllm_uva_host_resident_moe_routing(
        num_tokens=4,
        top_k=8,
        global_num_experts=256,
        unique_active_experts=10,
        expert_weight_mib=48.0,
        weights_host_resident=True,
    )

    return {
        "experiment_id": "sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva",
        "phase": 62,
        "priority": 71,
        "track": "Track B — LLM Serving, MoE Routing & KV/Hybrid State Management",
        "evidence_boundary": "own-offline deterministic Python stdlib model parameterized by upstream commits",
        "sglang_cutlass_prepare_moe_input": cutlass_case,
        "sglang_marlin_inactive_zeroing": marlin_case,
        "sglang_decoder_tail_ownership": {
            "tp_sharded_global_bound": tail_pack_sharded,
            "unsharded_global_bound": tail_pack_unsharded,
        },
        "trtllm_connector_kv_reuse": {
            "aligned_0_256": trtllm_aligned,
            "unaligned_16_240": trtllm_unaligned,
        },
        "vllm_mamba_sibling_tail": mamba_siblings,
        "vllm_uva_host_resident_moe": uva_moe,
        "headline_metrics": {
            "cutlass_active_routes": cutlass_case["active_routes"],
            "cutlass_inactive_routes": cutlass_case["inactive_routes"],
            "trtllm_aligned_reused_blocks": trtllm_aligned["reused_full_blocks"],
            "trtllm_unaligned_reused_blocks": trtllm_unaligned["reused_full_blocks"],
            "vllm_mamba_sibling_step_reduction_x": mamba_siblings["step_reduction_factor"],
            "vllm_uva_pcie_traffic_reduction_x": uva_moe["pcie_traffic_reduction_x"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["sglang_cutlass_prepare_moe_input"]["expert_offsets"] == [0, 3, 4, 6, 6]
        assert summary["sglang_cutlass_prepare_moe_input"]["blockscale_offsets"] == [0, 128, 256, 384, 384]
        assert summary["sglang_cutlass_prepare_moe_input"]["tail_padding_points_to_token_zero"] is True
        assert summary["sglang_marlin_inactive_zeroing"]["all_valid_routes_covered"] is True
        assert summary["sglang_decoder_tail_ownership"]["tp_sharded_global_bound"]["can_capture_moe_bound"] is True
        assert summary["sglang_decoder_tail_ownership"]["unsharded_global_bound"]["can_capture_moe_bound"] is False
        assert summary["trtllm_connector_kv_reuse"]["aligned_0_256"]["reused_full_blocks"] == 4
        assert summary["trtllm_connector_kv_reuse"]["unaligned_16_240"]["reused_full_blocks"] == 2
        assert summary["trtllm_connector_kv_reuse"]["unaligned_16_240"]["partial_boundary_blocks_kept_as_miss"] == 2
        assert summary["vllm_mamba_sibling_tail"]["step_reduction_factor"] == 8.0
        assert summary["vllm_uva_host_resident_moe"]["pcie_traffic_reduction_x"] == 3.2

    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
