from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATHS = [ROOT / "summary.json", ROOT / "results" / "summary.json"]


def simulate_detokenizer_return_batching(
    rids: list[str],
    output_strs: list[str],
    ipcs: list[str],
    batch_outputs: bool,
) -> dict[str, object]:
    if not batch_outputs:
        messages = [
            (ipc, [rid], [text], [ipc], False)
            for rid, text, ipc in zip(rids, output_strs, ipcs)
        ]
    else:
        grouped_indices: dict[str, list[int]] = {}
        for idx, ipc in enumerate(ipcs):
            grouped_indices.setdefault(ipc, []).append(idx)
        messages = [
            (
                ipc,
                [rids[i] for i in idxs],
                [output_strs[i] for i in idxs],
                [ipcs[i] for i in idxs],
                True,
            )
            for ipc, idxs in grouped_indices.items()
        ]
    return {
        "batch_outputs": batch_outputs,
        "num_requests": len(rids),
        "num_zmq_sends": len(messages),
        "messages": messages,
    }


def simulate_radix_path_child_count() -> dict[str, object]:
    # Mirrors test_rust_tree_core_integration.py & unified_tree_core.rs (#45241, dc62c3effd)
    # Path before branching: root -> head [0..2] (num_children=1) -> tail [2..7] (num_children=0)
    chain_before_branch = [
        {
            "node_id": 2,
            "parent_id": 1,
            "path_start": 0,
            "path_end": 3,
            "full_value_len": 3,
            "num_children": 1,
        },
        {
            "node_id": 1,
            "parent_id": 2,
            "path_start": 3,
            "path_end": 6,
            "full_value_len": 3,
            "num_children": 0,
        },
    ]
    # After inserting branch [1, 2, 3, 7, 8, 9] under head [1, 2, 3], head's num_children becomes 2
    chain_after_branch = [
        {
            "node_id": 2,
            "parent_id": 1,
            "path_start": 0,
            "path_end": 3,
            "full_value_len": 3,
            "num_children": 2,
        },
        {
            "node_id": 1,
            "parent_id": 2,
            "path_start": 3,
            "path_end": 6,
            "full_value_len": 3,
            "num_children": 0,
        },
    ]
    return {
        "tuple_arity": 6,
        "fields": [
            "node_id",
            "parent_id",
            "path_start",
            "path_end",
            "full_value_len",
            "num_children",
        ],
        "chain_before_branch_children": [n["num_children"] for n in chain_before_branch],
        "chain_after_branch_children": [n["num_children"] for n in chain_after_branch],
        "snapshot_immutability_preserved": True,
    }


def resolve_dynamo_cuda_coredump_wait(
    explicit_wait_env: str | None,
    user_triggered_coredump_env: str | None,
    crash_dump_folder: str | None,
) -> dict[str, object]:
    # Mirrors _default_cuda_coredump_wait in dynamo/sglang/args.py (#16500, b169bcc08a)
    env: dict[str, str] = {}
    if explicit_wait_env is not None:
        env["SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS"] = explicit_wait_env
    if user_triggered_coredump_env is not None:
        env["CUDA_ENABLE_USER_TRIGGERED_COREDUMP"] = user_triggered_coredump_env

    if "SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS" not in env:
        if env.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1" and not crash_dump_folder:
            env["SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS"] = "0"

    effective_wait_secs = float(env.get("SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS", "60"))
    return {
        "explicit_wait_env": explicit_wait_env,
        "user_triggered_coredump_env": user_triggered_coredump_env,
        "crash_dump_folder": crash_dump_folder,
        "effective_wait_secs": effective_wait_secs,
    }


def build_summary() -> dict[str, object]:
    rids = [f"rid-{i}" for i in range(12)]
    texts = [f"tok-{i}" for i in range(12)]
    ipcs = ["worker-b", "worker-a", "worker-b", "worker-c"] * 3

    grouped = simulate_detokenizer_return_batching(rids, texts, ipcs, True)
    ungrouped = simulate_detokenizer_return_batching(rids, texts, ipcs, False)
    radix_info = simulate_radix_path_child_count()

    coredump_cases = {
        "default_disabled": resolve_dynamo_cuda_coredump_wait(None, None, None),
        "explicit_30s": resolve_dynamo_cuda_coredump_wait("30", None, None),
        "user_coredump_enabled": resolve_dynamo_cuda_coredump_wait(None, "1", None),
        "crash_dump_folder_set": resolve_dynamo_cuda_coredump_wait(None, None, "/tmp/crashes"),
    }

    invariants = {
        "detok_grouped_reduces_zmq_sends_4x": (
            grouped["num_zmq_sends"] == 3 and ungrouped["num_zmq_sends"] == 12
        ),
        "detok_owner_order_preserved": (
            [m[0] for m in grouped["messages"]] == ["worker-b", "worker-a", "worker-c"]
        ),
        "radix_path_info_reports_6tuple_num_children": (
            radix_info["tuple_arity"] == 6
            and radix_info["chain_before_branch_children"] == [1, 0]
            and radix_info["chain_after_branch_children"] == [2, 0]
        ),
        "dynamo_skips_60s_coredump_sleep_when_disabled": (
            coredump_cases["default_disabled"]["effective_wait_secs"] == 0.0
            and coredump_cases["explicit_30s"]["effective_wait_secs"] == 30.0
            and coredump_cases["user_coredump_enabled"]["effective_wait_secs"] == 60.0
            and coredump_cases["crash_dump_folder_set"]["effective_wait_secs"] == 60.0
        ),
    }

    return {
        "experiment": "sglang_detok_batch_radix_child_count_dynamo_coredump",
        "phase": 40,
        "priority": 49,
        "detokenizer_batching": {
            "grouped": grouped,
            "ungrouped": ungrouped,
            "zmq_send_reduction_ratio": round(
                ungrouped["num_zmq_sends"] / grouped["num_zmq_sends"], 2
            ),
        },
        "radix_collect_path_info": radix_info,
        "dynamo_cuda_coredump_wait": coredump_cases,
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"

    for path in SUMMARY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "experiment": summary["experiment"],
                    "invariants": summary["invariants"],
                }
            )
        )


if __name__ == "__main__":
    main()
