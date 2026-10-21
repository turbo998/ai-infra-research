"""Deterministic model for TRT-LLM CuTe DSL uneven next_n DSA MQA logits split, KV transfer Backend contract & MPI teardown ACK, and vLLM ThreadedWatchDog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick_dsl_expand(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: tuple[int, ...] = (1, 2, 3),
) -> tuple[int, int]:
    """Select exact divisor caller-side (expand_factor, atom) for CuTe DSL paged MQA logits."""
    best_factor = next_n
    best_atom = 1
    for atom in sorted(kernel_atoms, reverse=True):
        if atom > 0 and next_n % atom == 0:
            best_atom = atom
            best_factor = next_n // atom
            break
    return best_factor, best_atom


def atom_ctx_offsets(next_n: int, num_atoms: int) -> list[int]:
    """Per-atom causal stagger of an in-kernel next_n split (TRT-LLM PR #19976)."""
    base, rem = divmod(next_n, num_atoms)
    return [
        next_n - (i * base + min(i, rem) + base + (1 if i < rem else 0))
        for i in range(num_atoms)
    ]


def pick_dsl_split(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    """Reproduce TRT-LLM PR #19976 `_pick_dsl_split` for prime/uneven `next_n` MTP windows."""
    factor, atom = pick_dsl_expand(next_n, num_sms, batch_size, max_ctx, kernel_atoms)
    tiles = -(-max_ctx // 256) if max_ctx > 0 else 0
    waves_atom1 = -(-batch_size * next_n * tiles // num_sms) if (batch_size > 0 and max_ctx > 0) else 0

    if atom != 1 or next_n <= 1 or any(next_n % a == 0 for a in kernel_atoms if a > 1):
        kv_reads = factor
        return {
            "next_n": next_n,
            "batch_size": batch_size,
            "max_ctx": max_ctx,
            "expand_factor": factor,
            "atom": atom,
            "num_kernel_atoms": 1,
            "mode": "exact_caller_expand" if factor > 1 else "native_single_atom",
            "kv_reads_per_step": kv_reads,
            "ctx_offsets": [0],
            "waves_atom1": waves_atom1,
            "waves_split": waves_atom1 // max(atom, 1),
        }

    num_atoms = -(-next_n // max(kernel_atoms))
    waves_split = (
        -(-batch_size * num_atoms * tiles // num_sms)
        if (batch_size > 0 and max_ctx > 0)
        else 0
    )
    if batch_size > 0 and max_ctx > 0:
        if waves_split >= waves_atom1 or waves_atom1 < 16:
            return {
                "next_n": next_n,
                "batch_size": batch_size,
                "max_ctx": max_ctx,
                "expand_factor": factor,
                "atom": atom,
                "num_kernel_atoms": 1,
                "mode": "fallback_atom1_low_waves",
                "kv_reads_per_step": next_n,
                "ctx_offsets": list(range(next_n - 1, -1, -1)),
                "waves_atom1": waves_atom1,
                "waves_split": waves_split,
            }

    in_kernel_atom = -(-next_n // num_atoms)
    return {
        "next_n": next_n,
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "expand_factor": 1,
        "atom": in_kernel_atom,
        "num_kernel_atoms": num_atoms,
        "mode": "in_kernel_uneven_split",
        "kv_reads_per_step": num_atoms,
        "kv_read_reduction_pct": round((1.0 - num_atoms / next_n) * 100.0, 2),
        "ctx_offsets": atom_ctx_offsets(next_n, num_atoms),
        "waves_atom1": waves_atom1,
        "waves_split": waves_split,
    }


def simulate_kv_backend_contract_and_mpi_stop(
    group_lengths: list[int],
    delivered_rows_per_group: list[list[bool]],
    fault: str | None,
    quiesced: bool,
    worker_busy_seconds: float,
    poll_interval_seconds: float = 0.1,
) -> dict[str, object]:
    """Model TRT-LLM PR #20117 `Outcome` / `quiesced()` safety and PR #20608 `SHUTDOWN_ACK` handshake."""
    if len(group_lengths) != len(delivered_rows_per_group):
        raise ValueError("Mismatched group count in Outcome check")
    for idx, (expected_len, mask) in enumerate(zip(group_lengths, delivered_rows_per_group)):
        if len(mask) != expected_len:
            raise ValueError(f"Group {idx}: mask length {len(mask)} != {expected_len}")

    delivered_counts = [sum(1 for flag in mask if flag) for mask in delivered_rows_per_group]
    undelivered_counts = [
        expected_len - delivered
        for expected_len, delivered in zip(group_lengths, delivered_counts)
    ]
    # Undelivered rows may be partially written; memory can be reused ONLY once quiesced() is True
    undelivered_memory_reusable = quiesced
    poll_ticks_during_teardown = int(round(worker_busy_seconds / poll_interval_seconds))
    ack_latency_seconds = round(poll_interval_seconds, 3)

    return {
        "group_lengths": group_lengths,
        "delivered_counts": delivered_counts,
        "undelivered_counts": undelivered_counts,
        "fault": fault,
        "partial_delivery_with_fault": any(c > 0 for c in delivered_counts) and fault is not None,
        "quiesced": quiesced,
        "undelivered_memory_reusable": undelivered_memory_reusable,
        "mpi_teardown": {
            "worker_busy_seconds": worker_busy_seconds,
            "pending_poll_seconds": poll_interval_seconds,
            "control_polls_during_teardown": poll_ticks_during_teardown,
            "shutdown_ack_received": True,
            "ack_latency_seconds": ack_latency_seconds,
        },
    }


def simulate_vllm_watchdog_timeline(
    events: list[tuple[float, str]],
    *,
    timeout: float = 300.0,
) -> dict[str, object]:
    """Simulate vLLM PR #55700 `ThreadedWatchDog` timeline with `feed`, `disable`, `enable`, and `check`."""
    active = True
    last_feed_time = 0.0
    last_timeout_log = -1.0
    dumps: list[dict[str, object]] = []

    for timestamp, action in events:
        if action == "feed":
            last_feed_time = timestamp
        elif action == "disable":
            if not active:
                raise AssertionError("WatchDog.disable() is not re-entrant")
            active = False
        elif action == "enable":
            active = True
        elif action == "check":
            if not active:
                last_feed_time = timestamp
                continue
            if (timestamp - last_feed_time > timeout) and (last_timeout_log < last_feed_time):
                last_timeout_log = timestamp
                dumps.append(
                    {
                        "dump_seq": len(dumps) + 1,
                        "timestamp": timestamp,
                        "stall_duration_s": round(timestamp - last_feed_time, 1),
                    }
                )
        else:
            raise ValueError(f"Unknown watchdog event: {action}")

    return {
        "timeout_s": timeout,
        "dump_count": len(dumps),
        "dumps": dumps,
        "last_feed_time": last_feed_time,
    }


def build_summary() -> dict[str, object]:
    # 1. CuTe DSL uneven next_n split on GB300 (148 SMs)
    split_cases = [
        pick_dsl_split(next_n=5, num_sms=148, batch_size=32, max_ctx=16384),
        pick_dsl_split(next_n=7, num_sms=148, batch_size=32, max_ctx=16384),
        pick_dsl_split(next_n=7, num_sms=148, batch_size=1, max_ctx=4096),
        pick_dsl_split(next_n=6, num_sms=148, batch_size=32, max_ctx=16384),
    ]

    # 2. KV Transfer Backend contract + MPI session server teardown ACK
    contract_unquiesced = simulate_kv_backend_contract_and_mpi_stop(
        group_lengths=[4, 4],
        delivered_rows_per_group=[[True, True, False, False], [True, False, False, False]],
        fault="peer_cancelled",
        quiesced=False,
        worker_busy_seconds=2.5,
    )
    contract_quiesced = simulate_kv_backend_contract_and_mpi_stop(
        group_lengths=[4, 4],
        delivered_rows_per_group=[[True, True, False, False], [True, False, False, False]],
        fault="peer_cancelled",
        quiesced=True,
        worker_busy_seconds=2.5,
    )

    # 3. vLLM ThreadedWatchDog timeline:
    # - 0..600s idle in shm_broadcast.acquire_read under `with wd.disable():` -> 0 false dumps
    # - 600s feed on RPC start, stall until 950s & 1000s -> exactly 1 dump (one-shot per stall)
    # - 1010s feed on next RPC, stall until 1350s -> 2nd dump
    watchdog_sim = simulate_vllm_watchdog_timeline(
        [
            (0.0, "feed"),
            (10.0, "disable"),
            (200.0, "check"),
            (450.0, "check"),
            (590.0, "check"),
            (600.0, "enable"),
            (600.0, "feed"),
            (750.0, "check"),
            (910.0, "check"),  # first stall > 300s -> dump #1
            (950.0, "check"),  # still same stall -> suppressed!
            (1010.0, "feed"),  # RPC recovered and re-fed
            (1320.0, "check"),  # second stall > 300s -> dump #2
        ],
        timeout=300.0,
    )

    return {
        "source_commits": [
            "NVIDIA/TensorRT-LLM@033f2ea0205341b2f3ad55bbeb5593427030ec78",
            "NVIDIA/TensorRT-LLM@a9000fe10ecd62973dcf3670573419299b390acb",
            "NVIDIA/TensorRT-LLM@16d26df5a7675b0aa13a213305c5dd59e504e9d5",
            "vllm-project/vllm@1c879919786497217b3b5fb0258d1d72f2442cb0",
        ],
        "cute_dsl_paged_mqa_logits": {
            "default_enabled_on_sm100f": True,
            "split_cases": split_cases,
            "gb300_next_n7_high_wave_speedup_range": [1.04, 2.60],
        },
        "kv_transfer_backend_contract": {
            "unquiesced_attempt": contract_unquiesced,
            "quiesced_attempt": contract_quiesced,
        },
        "vllm_threaded_watchdog": watchdog_sim,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        cases = summary["cute_dsl_paged_mqa_logits"]["split_cases"]
        # next_n=5, B=32, ctx=16384 -> in_kernel_uneven_split with 2 atoms, offsets [2, 0]
        assert cases[0]["mode"] == "in_kernel_uneven_split"
        assert cases[0]["num_kernel_atoms"] == 2 and cases[0]["ctx_offsets"] == [2, 0]
        # next_n=7, B=32, ctx=16384 -> in_kernel_uneven_split with 3 atoms, offsets [4, 2, 0]
        assert cases[1]["mode"] == "in_kernel_uneven_split"
        assert cases[1]["num_kernel_atoms"] == 3 and cases[1]["ctx_offsets"] == [4, 2, 0]
        assert cases[1]["kv_read_reduction_pct"] == 57.14
        # next_n=7, B=1, ctx=4096 -> <16 waves -> fallback_atom1_low_waves
        assert cases[2]["mode"] == "fallback_atom1_low_waves" and cases[2]["num_kernel_atoms"] == 1
        # next_n=6 -> divisible by 3 -> exact_caller_expand (expand_factor=2, atom=3)
        assert cases[3]["mode"] == "exact_caller_expand" and cases[3]["expand_factor"] == 2
        assert summary["kv_transfer_backend_contract"]["unquiesced_attempt"]["undelivered_memory_reusable"] is False
        assert summary["kv_transfer_backend_contract"]["quiesced_attempt"]["undelivered_memory_reusable"] is True
        wd = summary["vllm_threaded_watchdog"]
        assert wd["dump_count"] == 2
        assert wd["dumps"][0]["timestamp"] == 910.0 and wd["dumps"][1]["timestamp"] == 1320.0

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
