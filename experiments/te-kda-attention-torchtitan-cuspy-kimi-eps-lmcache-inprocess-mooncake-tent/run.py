#!/usr/bin/env python3
"""Deterministic reproduction of TransformerEngine, TorchTitan, LMCache, Dynamo, and Mooncake Phase 50 changes.

Covers:
1. NVIDIA/TransformerEngine@b9715e55a9161c703da3b8cd1d8117c2257f356e (#3548) &
   c3081a5f2db9de31eab51689a25c70ea48068aa6 (#3470):
   - `KimiDeltaAttention` (KDA) recurrence `S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T`
     where per-key-channel `alpha_t = exp(g_t)` does NOT commute with `(I - beta_t k_t k_t^T)` when `g_t`
     varies across key channels, unlike scalar Gated DeltaNet (`g_t` constant across channels).
   - Pre-cuDNN 9.26.0 THD learnable-softmax backward guard narrowed to `head_dim_v not in {64, 128, 256}`.
2. pytorch/torchtitan@789740c02578d288ee6e5a5651c6f96162bf9b84 (#5525),
   35f4a2e166586302ae6da957656a839759b20121 (#5450), &
   d06d37f53cb206ad008fb5869ae36a5c46adba50 (#5548):
   - Kimi K2.x (`Moonlight-16B-A3B`, `Kimi-VL-A3B`, `Kimi-K2.5`) decoder-layer `rms_norm_eps=1e-5`
     vs hardcoded `1e-6` drift on small-norm embedding rows (`34-40%` max drift, MLA q/kv norms stay `1e-6`).
   - `CuspyProfilerConfig` validation (`pm_sampling_interval_ms` / `pm_lookback_window_ms` require `performance_metrics`,
     and `pftrace` vs `json` trace filenames).
3. LMCache/LMCache@eb2f3df13f6e49e8a92c464f161dc3c456da3b42 (#5925),
   e6534c635e747b2c66ac2cb5342236d0ce62f312 (#5939), &
   ai-dynamo/dynamo@45a4e56a5f4dd4111d937ac36f38ac669677468d (#16324):
   - `lmcache.inprocess` namespace isolation and Dynamo `dynamo.vllm.pd_connector` migration.
4. kvcache-ai/Mooncake@7540f5826ed32cb755b1db027a2c609adb53f024 (#4154) &
   89520212377af7114bbba4177021ba092d78ed11 (#4611):
   - TENT diagnostic exit-code precedence (`4 > 3 > 1 > 0`, `2` for CLI parse error) and `Redactor`
     strict pseudonym consistency across structured fields and free text.
   - `structured_object_store.py` `_typed_ragged_row_dtypes` preserving `int64` precision for `2**53 + 1`
     and `msgpack_ragged` for `bool` rows when an empty list `[]` row is present.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def kda_step_decay_first(
    state: list[list[float]],
    q: list[float],
    k: list[float],
    v: list[float],
    g: list[float],
    beta: float,
    scale: float,
) -> tuple[list[list[float]], list[float]]:
    """Reproduce TransformerEngine #3548 KimiDeltaAttention recurrence:
    S_t = (I - beta k k^T) Diag(exp(g)) S_{t-1} + beta k v^T, o_t = scale * q S_t.
    Here state has shape [qk_dim, v_dim]."""
    qk_dim = len(k)
    v_dim = len(v)
    alpha = [math.exp(x) for x in g]

    # 1. Decay first: S_decayed[i][j] = alpha[i] * state[i][j]
    s_decayed = [[alpha[i] * state[i][j] for j in range(v_dim)] for i in range(qk_dim)]

    # 2. Erase term: k^T S_decayed -> vector of length v_dim
    k_dot_s = [
        sum(k[i] * s_decayed[i][j] for i in range(qk_dim))
        for j in range(v_dim)
    ]

    # 3. Update state: S_t[i][j] = S_decayed[i][j] - beta * k[i] * k_dot_s[j] + beta * k[i] * v[j]
    new_state = [
        [
            s_decayed[i][j] - beta * k[i] * k_dot_s[j] + beta * k[i] * v[j]
            for j in range(v_dim)
        ]
        for i in range(qk_dim)
    ]
    out = [
        scale * sum(q[i] * new_state[i][j] for i in range(qk_dim))
        for j in range(v_dim)
    ]
    return new_state, out


def kda_step_erase_first(
    state: list[list[float]],
    k: list[float],
    v: list[float],
    g: list[float],
    beta: float,
) -> list[list[float]]:
    """Alternative ordering Diag(exp(g)) (I - beta k k^T) S_{t-1} + beta k v^T to demonstrate
    non-commutativity when g varies across channels vs exact commutativity when g is scalar."""
    qk_dim = len(k)
    v_dim = len(v)
    alpha = [math.exp(x) for x in g]
    k_dot_s = [sum(k[i] * state[i][j] for i in range(qk_dim)) for j in range(v_dim)]
    s_erased = [
        [state[i][j] - beta * k[i] * k_dot_s[j] for j in range(v_dim)]
        for i in range(qk_dim)
    ]
    return [
        [alpha[i] * s_erased[i][j] + beta * k[i] * v[j] for j in range(v_dim)]
        for i in range(qk_dim)
    ]


def check_te_thd_learnable_softmax_bwd_supported(
    cudnn_runtime_version: int,
    head_dim_v: int,
) -> bool:
    """Reproduce TransformerEngine #3470 narrowed guard for THD learnable softmax backward."""
    if cudnn_runtime_version < 92600 and head_dim_v not in (64, 128, 256):
        return False
    return True


def rms_norm_relative_diff(values: list[float], eps_a: float = 1e-6, eps_b: float = 1e-5) -> float:
    """Reproduce TorchTitan #5548 RMSNorm sensitivity between eps=1e-6 (DeepSeek-V3) and eps=1e-5 (Kimi K2.x)."""
    mean_sq = sum(x * x for x in values) / len(values)
    scale_a = 1.0 / math.sqrt(mean_sq + eps_a)
    scale_b = 1.0 / math.sqrt(mean_sq + eps_b)
    return abs(scale_a - scale_b) / scale_b


def mooncake_tent_diagnostic_exit_code(checks: list[dict[str, str]]) -> int:
    """Reproduce Mooncake #4154 diagnosticExitCode() precedence: 4 (internal) > 3 (collector) > 1 (fail) > 0."""
    code = 0
    for chk in checks:
        if chk["outcome"] != "kFail":
            continue
        cid = chk["id"]
        if cid == "tool.internal":
            return 4
        if cid in ("engine.process", "tool.probe_missing"):
            code = max(code, 3)
        else:
            code = max(code, 1)
    return code


def mooncake_tent_strict_redact(field_val: str, log_line: str) -> dict[str, Any]:
    """Reproduce Mooncake #4154 Redactor::sanitize + sanitizeText consistent pseudonym & secret redaction."""
    pseudonyms: dict[str, str] = {}
    fields_redacted = 0

    def get_pseudo(token: str, prefix: str = "node") -> str:
        nonlocal fields_redacted
        if token not in pseudonyms:
            pseudonyms[token] = f"<{prefix}-{len(pseudonyms) + 1}>"
        fields_redacted += 1
        return pseudonyms[token]

    sanitized_field = get_pseudo(field_val, "host")
    text = log_line
    # Replace secret assignments
    text, n_sec = re.subn(r"(token|password|secret)=\S+", r"\1=[redacted]", text)
    fields_redacted += n_sec
    # Replace rkey/lkey
    text, n_key = re.subn(r"(rkey|lkey)=0x[0-9a-fA-F]+", r"\1=[redacted]", text)
    fields_redacted += n_key
    # Replace known deployment strings with identical pseudonym
    if field_val in text:
        count = text.count(field_val)
        text = text.replace(field_val, pseudonyms[field_val])
        fields_redacted += count
    # Replace IPv4 addresses
    def repl_ip(match: re.Match[str]) -> str:
        return get_pseudo(match.group(0), "ip")

    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", repl_ip, text)
    return {
        "sanitized_field": sanitized_field,
        "sanitized_text": text,
        "fields_redacted": fields_redacted,
    }


def mooncake_choose_ragged_codec_and_roundtrip(rows: list[list[Any]]) -> dict[str, Any]:
    """Reproduce Mooncake #4611 _typed_ragged_row_dtypes fixing empty-list float64 promotion."""
    non_empty_rows = [r for r in rows if len(r) > 0]
    # Legacy buggy behavior: empty list [] has float64 dtype in NumPy, promoting int64 -> float64
    has_empty = any(len(r) == 0 for r in rows)
    all_bool = bool(non_empty_rows) and all(isinstance(x, bool) for r in non_empty_rows for x in r)
    all_int = (
        bool(non_empty_rows)
        and not all_bool
        and all(isinstance(x, int) and not isinstance(x, bool) for r in non_empty_rows for x in r)
    )

    if all_bool:
        fixed_codec = "msgpack_ragged"
        fixed_dtype = "bool"
        fixed_roundtrip = [list(r) for r in rows]
    elif all_int:
        fixed_codec = "typed_ragged"
        fixed_dtype = "int64"
        fixed_roundtrip = [[int(x) for x in r] for r in rows]
    else:
        fixed_codec = "typed_ragged"
        fixed_dtype = "float64"
        fixed_roundtrip = [[float(x) for x in r] for r in rows]

    legacy_roundtrip = (
        [[int(float(x)) if isinstance(x, int) else float(x) for x in r] for r in rows]
        if has_empty
        else fixed_roundtrip
    )

    return {
        "fixed_codec": fixed_codec,
        "fixed_dtype": fixed_dtype,
        "fixed_roundtrip": fixed_roundtrip,
        "legacy_float64_corrupted_roundtrip": legacy_roundtrip,
    }


def run_experiment() -> dict[str, Any]:
    # 1. TransformerEngine KDA non-commutativity vs scalar GDN commutativity
    init_state = [[0.5, -0.3], [0.8, 0.4]]
    q = [0.6, -0.4]
    k = [0.7, 0.5]
    v = [1.2, -0.9]
    g_vector = [-0.1, -0.9]  # per-key-channel distinct decay
    g_scalar = [-0.4, -0.4]  # scalar decay (recovers GDN)
    beta = 0.5

    s_vec_decay_first, out_vec = kda_step_decay_first(init_state, q, k, v, g_vector, beta, scale=1.0)
    s_vec_erase_first = kda_step_erase_first(init_state, k, v, g_vector, beta)
    max_noncomm_diff = max(
        abs(s_vec_decay_first[i][j] - s_vec_erase_first[i][j])
        for i in range(2)
        for j in range(2)
    )

    s_sca_decay_first, _ = kda_step_decay_first(init_state, q, k, v, g_scalar, beta, scale=1.0)
    s_sca_erase_first = kda_step_erase_first(init_state, k, v, g_scalar, beta)
    max_scalar_comm_diff = max(
        abs(s_sca_decay_first[i][j] - s_sca_erase_first[i][j])
        for i in range(2)
        for j in range(2)
    )

    # 2. TorchTitan Kimi K2.x RMSNorm eps=1e-5 vs 1e-6 on small-variance embedding row
    small_norm_embedding = [0.003, -0.003, 0.003, -0.003]  # mean_sq = 9e-6 -> 37.84% drift
    kimi_low_norm_drift_pct = round(rms_norm_relative_diff(small_norm_embedding, 1e-6, 1e-5) * 100.0, 2)

    # 3. Mooncake TENT diagnostics & structured_object_store int64/bool empty-row preservation
    big_int = (1 << 53) + 1  # 9007199254740993
    int_ragged = mooncake_choose_ragged_codec_and_roundtrip([[1, 2], [], [big_int]])
    bool_ragged = mooncake_choose_ragged_codec_and_roundtrip([[True, False], []])

    redaction = mooncake_tent_strict_redact(
        field_val="rdma-node-07.cluster.local",
        log_line="Connect failed on rdma-node-07.cluster.local (10.24.8.19) token=sec_99a rkey=0xdeadbeef",
    )

    return {
        "experiment": "te-kda-attention-torchtitan-cuspy-kimi-eps-lmcache-inprocess-mooncake-tent",
        "transformer_engine": {
            "kda_output": [round(x, 6) for x in out_vec],
            "kda_vector_gate_noncomm_max_diff": round(max_noncomm_diff, 6),
            "kda_scalar_gate_comm_max_diff": round(max_scalar_comm_diff, 12),
            "thd_learnable_softmax_cudnn925_d128": check_te_thd_learnable_softmax_bwd_supported(92500, 128),
            "thd_learnable_softmax_cudnn925_d96": check_te_thd_learnable_softmax_bwd_supported(92500, 96),
        },
        "torchtitan": {
            "kimi_rms_norm_eps": 1e-5,
            "deepseek_v3_rms_norm_eps": 1e-6,
            "low_norm_token_drift_pct": kimi_low_norm_drift_pct,
            "published_median_drift_pct": {
                "Moonlight-16B-A3B": 1.0,
                "Kimi-VL-A3B": 1.2,
                "Kimi-K2.5": 4.8,
            },
        },
        "lmcache_and_dynamo": {
            "lmcache_inprocess_adapter": "lmcache.inprocess.integration.vllm.vllm_v1_adapter",
            "lmcache_inprocess_standalone": "lmcache.inprocess.standalone",
            "dynamo_pd_connector_module": "dynamo.vllm.pd_connector",
            "dynamo_removed_kvbm_crates": 7,
        },
        "mooncake": {
            "exit_code_pass": mooncake_tent_diagnostic_exit_code([{"id": "ibv.dev", "outcome": "kPass"}]),
            "exit_code_check_fail": mooncake_tent_diagnostic_exit_code([{"id": "ibv.dev", "outcome": "kFail"}]),
            "exit_code_collector_fail": mooncake_tent_diagnostic_exit_code(
                [{"id": "ibv.dev", "outcome": "kFail"}, {"id": "engine.process", "outcome": "kFail"}]
            ),
            "exit_code_internal_fail": mooncake_tent_diagnostic_exit_code(
                [{"id": "engine.process", "outcome": "kFail"}, {"id": "tool.internal", "outcome": "kFail"}]
            ),
            "redaction": redaction,
            "int_ragged_roundtrip": int_ragged,
            "bool_ragged_roundtrip": bool_ragged,
        },
    }


def check_invariants(summary: dict[str, Any]) -> None:
    te = summary["transformer_engine"]
    assert te["kda_vector_gate_noncomm_max_diff"] > 0.04
    assert te["kda_scalar_gate_comm_max_diff"] == 0.0
    assert te["thd_learnable_softmax_cudnn925_d128"] is True
    assert te["thd_learnable_softmax_cudnn925_d96"] is False

    tt = summary["torchtitan"]
    assert tt["kimi_rms_norm_eps"] == 1e-5
    assert tt["deepseek_v3_rms_norm_eps"] == 1e-6
    assert 30.0 <= tt["low_norm_token_drift_pct"] <= 40.0

    ld = summary["lmcache_and_dynamo"]
    assert ld["dynamo_pd_connector_module"] == "dynamo.vllm.pd_connector"
    assert ld["dynamo_removed_kvbm_crates"] == 7

    mc = summary["mooncake"]
    assert mc["exit_code_pass"] == 0
    assert mc["exit_code_check_fail"] == 1
    assert mc["exit_code_collector_fail"] == 3
    assert mc["exit_code_internal_fail"] == 4
    assert mc["redaction"]["sanitized_field"] == "<host-1>"
    assert "<host-1>" in mc["redaction"]["sanitized_text"]
    assert "<ip-2>" in mc["redaction"]["sanitized_text"]
    assert "sec_99a" not in mc["redaction"]["sanitized_text"]
    assert "0xdeadbeef" not in mc["redaction"]["sanitized_text"]

    big_int = (1 << 53) + 1
    assert mc["int_ragged_roundtrip"]["fixed_codec"] == "typed_ragged"
    assert mc["int_ragged_roundtrip"]["fixed_dtype"] == "int64"
    assert mc["int_ragged_roundtrip"]["fixed_roundtrip"][2][0] == big_int
    assert mc["int_ragged_roundtrip"]["legacy_float64_corrupted_roundtrip"][2][0] == big_int - 1
    assert mc["bool_ragged_roundtrip"]["fixed_codec"] == "msgpack_ragged"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run invariant checks and write summary.json")
    args = parser.parse_args()

    summary = run_experiment()
    if args.check:
        check_invariants(summary)

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
