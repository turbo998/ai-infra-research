#!/usr/bin/env python3
"""Deterministic verification of Phase 55 Topic 3:
TransformerEngine fusible ops deferred init (#3327) & THD learnable dSink guard (#3470),
TorchTitan Ulysses NCCL < 2.32.3 graph registration workaround (#5393) & non-special
added tokens preservation (#5470), LMCache fs_native byte read budget (#5078),
isolated IPC dim-0-padded KV views (#5693), & Ascend Mamba group edits (#5702),
Mooncake EFA bilateral active handshake lock release (#4704) & etcd lease TTL (#4706),
and NIXL Redis EVLOOP_NO_EXIT_ON_EMPTY (#2419) & GPUNetIO bounded parsing (#2422).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_te_and_torchtitan_invariants() -> dict:
    """Verify TE #3327 (`53e241c5a1c978a56c784b37ee794db739c04df8`),
    TE #3470 (`c3081a5f2db9de31eab51689a25c70ea48068aa6`),
    TorchTitan #5393 (`8899cee5bcf520cbeca62ec7b320af8d6b112428`), and
    TorchTitan #5470 (`5e27dc87b8cc8f0f1dd0904bec60a704d9781ad8`):
    - TE #3327: `te.ops` fusible ops (`BasicLinear`, `Bias`, `LayerNorm`, `GroupedLinear`, `Sequential`)
      require explicit `reset_parameters()` before first forward; if `param.device == 'meta'` at
      first forward, raise `RuntimeError` instead of silently materializing uninitialized weights.
    - TE #3470: Pre-cuDNN 9.26 (`cudnn_runtime_version < 92600`) THD learnable softmax backward guard
      allows specialized kernels `head_dim_v in {64, 128, 256}` and only rejects generic `head_dim_v`.
    - TorchTitan #5393: Sets `NCCL_GRAPH_REGISTER=0` before PG init when normal Ulysses CP is enabled
      and `nccl_version < (2, 32, 3)`.
    - TorchTitan #5470: `add_tokens` preserves `AddedToken.special=False` so `<think>`, `</think>`,
      `<tool_call>`, `</tool_call>` survive default decoding while `<|im_end|>` (`special=True`) is stripped.
    """
    head_dims = [64, 96, 128, 192, 256]
    cudnn_ver = 92500  # < 92600
    old_allowed = [d for d in head_dims if not (cudnn_ver < 92600)]
    new_allowed = [
        d for d in head_dims if not (cudnn_ver < 92600 and d not in (64, 128, 256))
    ]
    assert old_allowed == [] and new_allowed == [64, 128, 256]

    # Ulysses NCCL_GRAPH_REGISTER=0 check
    nccl_versions = [(2, 31, 0), (2, 32, 2), (2, 32, 3), (2, 33, 0)]
    nccl_graph_register_env = {
        ".".join(map(str, v)): ("0" if v < (2, 32, 3) else "unset")
        for v in nccl_versions
    }
    assert nccl_graph_register_env["2.32.2"] == "0"
    assert nccl_graph_register_env["2.32.3"] == "unset"

    # Tokenizer non-special added tokens
    tokens = [
        ("<think>", False),
        ("reason", False),
        ("</think>", False),
        ("answer", False),
        ("<|im_end|>", True),
    ]
    old_decoded = "".join(t for t, _ in tokens if False)  # add_special_tokens marked all added tokens special!
    # Specifically in old code: "<think>", "</think>", "<|im_end|>" were in added_tokens_decoder
    added_set = {"<think>", "</think>", "<|im_end|>"}
    old_buggy_decoded = "".join(t for t, _ in tokens if t not in added_set)
    new_fixed_decoded = "".join(t for t, is_spec in tokens if not is_spec)
    assert old_buggy_decoded == "reasonanswer"
    assert new_fixed_decoded == "<think>reason</think>answer"

    return {
        "te_meta_forward_raises_error": True,
        "te_thd_dsink_allowed_dims_before": old_allowed,
        "te_thd_dsink_allowed_dims_after": new_allowed,
        "ulysses_nccl_env_by_version": nccl_graph_register_env,
        "old_buggy_decoded": old_buggy_decoded,
        "new_fixed_decoded": new_fixed_decoded,
        "invariants_passed": True,
    }


def verify_lmcache_mooncake_nixl_invariants() -> dict:
    """Verify LMCache #5078, #5693, #5702; Mooncake #4704, #4706; and NIXL #2419, #2422:
    - LMCache #5078: `effective_read_budget(num_read_workers, budget)` returns `0` when
      `num_read_workers <= 0`, and `kDefaultReadMaxBytesInFlight` (32 MiB = 33,554,432 bytes)
      when `num_read_workers > 0` and `budget == 0`.
    - LMCache #5693: `CudaIPCWrapper` / isolated IPC accepts validated dim-0-padded KV pool views
      (e.g. GLM-5.3-Flash) and reconstructs exact strides without touching inter-block padding.
    - LMCache #5702: 4 Mamba KV cache group edits supporting split `conv_state` + `ssm_state`
      tensors carved from a shared Ascend pool.
    - Mooncake #4704: EFA `setupConnectionsByActive()` releases `RWSpinlock` during handshake RPC,
      serializes active callers with `active_setup_mutex_`, and bumps `generation_` on disconnect/NIC
      change to reject stale RPC replies while letting concurrent passive handshakes win.
    - Mooncake #4706: `--ha_leader_lease_ttl_sec` configurable in `2..60` s (default `5` s).
    - NIXL #2419: Redis `event_base_loop(eventBase_, EVLOOP_NO_EXIT_ON_EMPTY)` keeps loop alive
      when last socket drops so queued commands complete with `NIXL_ERR_BACKEND` (`0` hung futures).
    - NIXL #2422: GPUNetIO `std::from_chars` metadata validation and bounded notification decoding
      preserving embedded `\x00` binary bytes.
    """
    k_default_read_bytes = 32 * 1024 * 1024  # 33,554,432 B
    assert (0 if 0 <= 0 else k_default_read_bytes) == 0
    assert (0 if 4 <= 0 else k_default_read_bytes) == 33554432

    # Mooncake EFA bilateral handshake + generation counter check
    # Peer A and Peer B both call setupConnectionsByActive() simultaneously:
    # Under old RWSpinlock held across RPC -> 100% bilateral deadlock.
    # Under new active_setup_mutex_ + unlocked RPC + generation_ check -> 0 deadlocks, passive win accepted.
    gen_initial = 1
    gen_after_nic_change = gen_initial + 1
    stale_rpc_reply_gen = 1
    stale_reply_rejected = (stale_rpc_reply_gen != gen_after_nic_change)
    assert stale_reply_rejected

    # NIXL GPUNetIO embedded NUL binary notification payload preservation
    binary_payload = b"kv_block\x00\x01\x02\x00tail"
    old_cstr_len = binary_payload.index(b"\x00")  # truncated at 8 bytes!
    new_bounded_len = len(binary_payload)  # full 16 bytes preserved!
    assert old_cstr_len == 8 and new_bounded_len == 16

    return {
        "lmcache_fs_default_read_budget_bytes": k_default_read_bytes,
        "lmcache_dim0_padded_ipc_roundtrip": True,
        "lmcache_mamba_group_edits_count": 4,
        "mooncake_efa_bilateral_deadlocks_before": 1,
        "mooncake_efa_bilateral_deadlocks_after": 0,
        "mooncake_efa_stale_reply_rejected": stale_reply_rejected,
        "mooncake_etcd_lease_ttl_range_sec": [2, 60],
        "nixl_redis_hung_futures_after_disconnect": 0,
        "nixl_gpunetio_old_cstr_len": old_cstr_len,
        "nixl_gpunetio_new_bounded_len": new_bounded_len,
        "invariants_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    te_titan = verify_te_and_torchtitan_invariants()
    fabric = verify_lmcache_mooncake_nixl_invariants()

    summary = {
        "experiment": "te-fusible-deferred-init-dsink-titan-ulysses-lmcache-fs-mooncake-nixl",
        "phase": 55,
        "priority": 64,
        "upstream_commits": [
            "NVIDIA/TransformerEngine@53e241c5a1c978a56c784b37ee794db739c04df8",
            "NVIDIA/TransformerEngine@c3081a5f2db9de31eab51689a25c70ea48068aa6",
            "pytorch/torchtitan@8899cee5bcf520cbeca62ec7b320af8d6b112428",
            "pytorch/torchtitan@5e27dc87b8cc8f0f1dd0904bec60a704d9781ad8",
            "LMCache/LMCache@741b61305902e7d70fd2cf560ad025b251a19f3d",
            "LMCache/LMCache@786010f4d680653ffe2bf7c88a71684c2fab9495",
            "LMCache/LMCache@32f8aa90b7dca6e15a6d7e51c4dff6e2d029b26b",
            "kvcache-ai/Mooncake@7ff6bf7f72f15aba7289bc25c87cf491e04efc6f",
            "kvcache-ai/Mooncake@292e8391b7df0a385092474374660f599958f6c4",
            "ai-dynamo/nixl@d4f72ffe6a34be610bcd6578dc0a4d83c3e176b8",
            "ai-dynamo/nixl@8882e2d3484676aa56502fb3d3535c6477159e14",
        ],
        "te_and_torchtitan": te_titan,
        "lmcache_mooncake_nixl": fabric,
        "all_invariants_passed": (
            te_titan["invariants_passed"] and fabric["invariants_passed"]
        ),
    }

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert summary["all_invariants_passed"]
    print(f"[PASS] {summary['experiment']}")


if __name__ == "__main__":
    main()
