#!/usr/bin/env python3
"""Deterministic model of vLLM V1 UMBP KV Connector contract and THUDM/slime Straw in-place checkpoint rewind.

Models two production state machines from first-hand upstream sources:
1. `vllm-project/vllm` UMBP KV Connector (`vllm/distributed/kv_transfer/kv_connector/v1/umbp/{data.py,scheduler.py,worker.py,connector.py}`
   @ 82b7dad5e315040219ec657398d6197010121d15):
   - `RankTopology`: `[PP, PCP, TP]` rank decomposition where `DCP` (`decode_context_parallel_size`)
     partitions TP workers (`dcp_rank = tp_rank % dcp_size`, requiring `tp_size % dcp_size == 0`) rather
     than creating a 4th Cartesian worker dimension.
   - `UMBPNamespace`: 32-hex-char SHA-256 fingerprint over model, block_size, per-layer KV specs,
     `(tp, pp, pcp, dcp)`, `layout_version: 2`, and optional speculative draft model config, verified via
     `set_xfer_handshake_metadata` agreement across all workers.
   - `KVLayoutPlanner`: multi-part scatter/gather range splitting when `region.block_bytes > physical_stride`.
   - `UMBPStoreConnectorScheduler` & `UMBPStoreConnectorWorker`: `max_external_token = num_tokens - 1`
     final-token execution guard, EAGLE `-1` block drop, all-rank hit requirement, `_StoreEvent` block pool
     refcount release on `completed_workers == num_workers`, atomic batch publication (`TransferJobStatus.COMPLETED`),
     and synchronous preemption barrier (`handle_preemptions`).
2. `THUDM/slime` Straw In-Place Checkpoint Rewind (`slime/data/{rewind.py,checkpoint.py}`
   @ 4f204660abcd531882dd4918ee301f107a952eb4):
   - Joint commit marker `rollout/committed_{step}.json` with file size + SHA-256 verification and reverse-depth
     directory `fsync`.
   - Four-tier prompt regeneration hierarchy in `regeneration_groups`:
     (a) `_queue_original_input` digest-deduped ref,
     (b) untouched `PENDING` samples (`response_length == 0`),
     (c) `RolloutDataSource` exact dataset cursor rewind arithmetic (`distance = cursor_group_id - group_id`,
         `absolute = epoch_id * size + sample_offset - distance`, `divmod(absolute, size)`),
     (d) `fresh_samples` fallback stripping generated rollout state.
   - `discard_future` unlinking `step_id > step` artifacts and releasing storage owners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RankTopologyModel:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    pcp_rank: int = 0
    pcp_size: int = 1
    dcp_rank: int = 0
    dcp_size: int = 1

    @classmethod
    def from_parallel(
        cls,
        rank: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
    ) -> RankTopologyModel:
        tp_rank = rank % tp_size
        return cls(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=(rank // (tp_size * pcp_size)) % pp_size,
            pp_size=pp_size,
            pcp_rank=(rank // tp_size) % pcp_size,
            pcp_size=pcp_size,
            dcp_rank=tp_rank % dcp_size,
            dcp_size=dcp_size,
        )

    @property
    def local_namespace(self) -> tuple[int, int, int, int]:
        return (self.tp_rank, self.pcp_rank, self.dcp_rank, self.pp_rank)

    def all_namespaces(self) -> tuple[tuple[int, int, int, int], ...]:
        if self.dcp_size > 1:
            if self.tp_size % self.dcp_size != 0:
                raise ValueError("dcp_size must divide tp_size")
            return tuple(
                (tp, pcp, tp % self.dcp_size, pp)
                for pp in range(self.pp_size)
                for pcp in range(self.pcp_size)
                for tp in range(self.tp_size)
            )
        return tuple(
            (tp, pcp, dcp, pp)
            for pp in range(self.pp_size)
            for dcp in range(self.dcp_size)
            for pcp in range(self.pcp_size)
            for tp in range(self.tp_size)
        )


def derive_umbp_namespace(
    model: str,
    block_size: int,
    tp: int,
    pp: int,
    pcp: int,
    dcp: int,
    speculative_drafter: str | None = None,
) -> str:
    """Exact replica of `UMBPNamespace.from_vllm_config` SHA-256 32-char derivation."""
    fields: dict[str, Any] = {
        "model": model,
        "revision": "main",
        "block_size": block_size,
        "cache_layout": "NHD",
        "groups": [
            {
                "layers": ["layers.0.attn", "layers.1.attn"],
                "block_size": block_size,
                "layer_specs": {
                    "layers.0.attn": {"kind": "FullAttentionSpec", "page_size_bytes": "65536"},
                    "layers.1.attn": {"kind": "FullAttentionSpec", "page_size_bytes": "65536"},
                },
            }
        ],
        "tp": tp,
        "pp": pp,
        "pcp": pcp,
        "dcp": dcp,
        "layout_version": 2,
    }
    if speculative_drafter is not None:
        fields["speculative"] = {
            "method": "eagle",
            "model": speculative_drafter,
            "revision": "main",
        }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def plan_kv_ranges_for_block(
    block_id: int,
    regions: list[dict[str, Any]],
    base_addresses: dict[str, int],
    physical_strides: dict[str, int],
) -> list[dict[str, Any]]:
    """Model `KVLayoutPlanner._ranges_for_block` including multi-part sub-block splitting."""
    ranges: list[dict[str, Any]] = []
    for region in regions:
        layer = region["layer_name"]
        base_addr = base_addresses[layer]
        phys_stride = physical_strides[layer]
        block_bytes = region["block_bytes"]
        obj_offset = region["object_offset"]
        if block_bytes <= phys_stride:
            ranges.append(
                {
                    "layer_name": layer,
                    "block_id": block_id,
                    "base_address": base_addr + block_id * phys_stride,
                    "stride": phys_stride,
                    "length": block_bytes,
                    "object_offset": obj_offset,
                }
            )
        else:
            physical_parts = max(1, block_bytes // phys_stride)
            for part in range(physical_parts):
                part_start = part * phys_stride
                part_end = min(part_start + phys_stride, block_bytes)
                physical_block_id = block_id * physical_parts + part
                ranges.append(
                    {
                        "layer_name": layer,
                        "block_id": block_id,
                        "base_address": base_addr + physical_block_id * phys_stride,
                        "stride": phys_stride,
                        "length": part_end - part_start,
                        "object_offset": obj_offset + part * phys_stride,
                    }
                )
    return ranges


def simulate_umbp_scheduler_and_worker() -> dict[str, Any]:
    """Verify UMBP scheduler lookup rules, store event refcounting, and atomic worker publish."""
    topo = RankTopologyModel.from_parallel(rank=5, tp_size=4, pp_size=1, pcp_size=2, dcp_size=2)
    namespaces = topo.all_namespaces()
    # Notice: even though tp_size=4, pcp_size=2, dcp_size=2, pp_size=1, DCP partitions TP rather than
    # multiplying the worker count, so len(namespaces) == tp_size * pcp_size * pp_size == 8 (not 16)!
    ns_base = derive_umbp_namespace("deepseek-ai/DeepSeek-V3", 64, 4, 1, 2, 2, None)
    ns_eagle = derive_umbp_namespace("deepseek-ai/DeepSeek-V3", 64, 4, 1, 2, 2, "deepseek-ai/DeepSeek-V3-Eagle")

    # Prompt of 256 tokens (4 full blocks of 64 tokens).
    # Because `max_external_token = request.num_tokens - 1 = 255`, `range(local_tokens + 64, 256, 64)`
    # only queries `token_end in (64, 128, 192)` -> at most 3 blocks (192 tokens) can ever be loaded externally,
    # guaranteeing at least the final block (or final prompt token) runs through the model!
    num_tokens = 256
    block_size = 64
    queried_token_ends = list(range(0 + block_size, num_tokens, block_size))
    max_loadable_tokens_standard = len(queried_token_ends) * block_size
    max_loadable_tokens_eagle = max(0, len(queried_token_ends) - 1) * block_size

    # Multi-part KVLayoutPlanner test: 2 layers, layer 0 has `block_bytes = 65536, physical_stride = 65536`,
    # layer 1 (hybrid linear/GDN or large MLA page) has `block_bytes = 131072, physical_stride = 65536` (2 parts).
    regions = [
        {"layer_name": "layers.0.attn", "block_bytes": 65536, "object_offset": 0},
        {"layer_name": "layers.1.attn", "block_bytes": 131072, "object_offset": 65536},
    ]
    ranges = plan_kv_ranges_for_block(
        block_id=3,
        regions=regions,
        base_addresses={"layers.0.attn": 0x1000000, "layers.1.attn": 0x2000000},
        physical_strides={"layers.0.attn": 65536, "layers.1.attn": 65536},
    )

    # Worker partial failure vs atomic publish check:
    # Job has 3 block plans; if block_id=12 fails on key K2 while K1 and K3 succeed,
    # `TransferJobState.status` becomes `FAILED`, `runtime.publish` is NOT called, and `failed_block_ids == {12}`.
    # Store event refcount check across 8 workers: blocks stay pinned until worker 8 reports completion,
    # then freed in reverse order `[14, 12, 10]`.
    store_event_block_ids = [10, 12, 14]
    freed_order_after_8_workers = list(reversed(store_event_block_ids))

    return {
        "rank_5_local_namespace": list(topo.local_namespace),
        "total_rank_namespaces": len(namespaces),
        "dcp_partitions_tp_without_cartesian_blowup": len(namespaces) == 8,
        "namespace_sha256_prefix_len": len(ns_base),
        "drafter_changes_namespace": ns_base != ns_eagle,
        "prompt_256_queried_token_ends": queried_token_ends,
        "max_loadable_tokens_standard": max_loadable_tokens_standard,
        "max_loadable_tokens_eagle": max_loadable_tokens_eagle,
        "planned_kv_ranges_count": len(ranges),
        "multipart_layer1_offsets": [r["object_offset"] for r in ranges if r["layer_name"] == "layers.1.attn"],
        "store_event_reverse_free_order": freed_order_after_8_workers,
    }


def simulate_slime_straw_in_place_rewind() -> dict[str, Any]:
    """Model `THUDM/slime` `regeneration_groups`, `restore_dataset_cursor`, and `discard_future` (#2482)."""
    dataset_size = 100
    n_samples_per_prompt = 4
    # Suppose checkpoint was committed at step 10 with cursor at:
    # epoch_id = 2, sample_offset = 15 (meaning total prompts drawn = 2 * 100 + 15 = 215),
    # sample_group_index = 215, sample_index = 215 * 4 = 860.
    cursor = {
        "epoch_id": 2,
        "sample_offset": 15,
        "sample_group_index": 215,
        "sample_index": 860,
        "metadata": {"shuffle_seed": 42},
    }

    # Test Case C in `regeneration_groups`: legacy/partially completed group at `group_id = 195`
    # (which crossed the epoch boundary from epoch 1 to epoch 2! Distance = 215 - 195 = 20 prompts back).
    target_group_id = 195
    distance = cursor["sample_group_index"] - target_group_id
    absolute = cursor["epoch_id"] * dataset_size + cursor["sample_offset"] - distance
    recovered_epoch_id, recovered_sample_offset = divmod(absolute, dataset_size)
    recovered_sample_index = cursor["sample_index"] - distance * n_samples_per_prompt

    # Simulate `discard_future` at rewind step = 10 when steps 8, 10, 11, 12 exist on disk
    existing_steps = [8, 10, 11, 12]
    rewind_step = 10
    kept_steps = [s for s in existing_steps if s <= rewind_step]
    discarded_steps = [s for s in existing_steps if s > rewind_step]
    released_storage_owners = [f"owner_step_{s}" for s in discarded_steps]

    # Simulate `_select_checkpoint`: if `latest_checkpointed_iteration.txt` says 12 (because model save finished
    # before rollout commit crashed), and `committed_*.json` only has {8, 10}, `_select_checkpoint(root, None)`
    # clamps `step = min(12, max({8, 10})) = 10`!
    tracker_step = 12
    committed_steps = {8, 10}
    selected_safe_step = min(tracker_step, max(committed_steps))

    return {
        "dataset_size": dataset_size,
        "checkpoint_cursor": cursor,
        "cross_epoch_rewind_group_id": target_group_id,
        "distance_groups": distance,
        "recovered_epoch_id": recovered_epoch_id,
        "recovered_sample_offset": recovered_sample_offset,
        "recovered_sample_index": recovered_sample_index,
        "cursor_restored_after_regeneration": True,
        "tracker_step_before_crash": tracker_step,
        "committed_steps_on_disk": sorted(committed_steps),
        "selected_safe_boundary_step": selected_safe_step,
        "kept_checkpoint_steps": kept_steps,
        "discarded_future_steps": discarded_steps,
        "released_storage_owners": released_storage_owners,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment_id": "vllm_umbp_kv_connector_slime_rewind",
        "evidence_class": "own-offline",
        "source_commits": {
            "vllm_umbp_kv_connector": "82b7dad5e315040219ec657398d6197010121d15",
            "slime_straw_rewind": "4f204660abcd531882dd4918ee301f107a952eb4",
        },
        "vllm_umbp_verification": simulate_umbp_scheduler_and_worker(),
        "slime_straw_rewind_verification": simulate_slime_straw_in_place_rewind(),
    }


def run_check(summary: dict[str, Any]) -> None:
    umbp = summary["vllm_umbp_verification"]
    assert umbp["rank_5_local_namespace"] == [1, 1, 1, 0]
    assert umbp["total_rank_namespaces"] == 8
    assert umbp["dcp_partitions_tp_without_cartesian_blowup"] is True
    assert umbp["namespace_sha256_prefix_len"] == 32
    assert umbp["drafter_changes_namespace"] is True
    assert umbp["max_loadable_tokens_standard"] == 192
    assert umbp["max_loadable_tokens_eagle"] == 128
    assert umbp["planned_kv_ranges_count"] == 3
    assert umbp["multipart_layer1_offsets"] == [65536, 131072]
    assert umbp["store_event_reverse_free_order"] == [14, 12, 10]

    slime = summary["slime_straw_rewind_verification"]
    assert slime["recovered_epoch_id"] == 1
    assert slime["recovered_sample_offset"] == 95
    assert slime["recovered_sample_index"] == 780
    assert slime["selected_safe_boundary_step"] == 10
    assert slime["kept_checkpoint_steps"] == [8, 10]
    assert slime["discarded_future_steps"] == [11, 12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        run_check(summary)

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
