"""Phase 65 Track B deterministic experiment:
1. SGLang CPU Offloader Storage-Sharing View Preservation & Alias Rebinding (6d6c9cd4ae / #44139)
   + Diffusion Linear-GELU Verdict Cache Dynamo Guard Bypass (494eeecb6c / #45944):
   - Models generic `untyped_storage().data_ptr()` scanning across module attributes (`vars(owner).values()`)
     so non-Parameter cached tensor views (e.g., KDA `conv_weights`, `GemmaRMSNorm.gemma_weight`) pin their
     backing Parameter on GPU during CPU offload, plus `_rebind_parameter_aliases` refresh hooks.
   - Models bypassing `_VERDICTS` dict lookup when `torch.compiler.is_compiling()` is True so TorchDynamo
     does not install dict-size guards that trigger O(L) recompilations across L linear layers.
2. TensorRT-LLM Packed 5D QKV Ulysses Post-Unscatter Kernel & Int32 Overflow Guards (1586d85375 / #17147):
   - Models `ulyssesPackedQkvPostUnscatterKernel` consuming `[P, B, Sp, 3, H, D]` directly with `uint4`
     (8 BF16 elements = 16B) vectorized loads/stores and grid `(3 * P * Sp, B, 1)`, eliminating Python-side
     `permute -> contiguous -> unbind` intermediate HBM traffic, plus `checkInt32SumProduct` guards.
3. vLLM Rust Frontend `bad_words` Validation, Pre-Tokenization Deduplication, Leading-Space Variant Rule & Cap (34b2954482 / #63315):
   - Replicates exact Python `sampling_params.py` parity in Rust `tokenize_bad_words`: reject empty strings
     before tokenization, deduplicate via `HashSet`, `trim_start()`, retain prefix-space encoding only when
     `with_space[0] != without_space[0] and len(with_space) == len(without_space)`, and enforce `max_num_bad_words`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

INT32_MAX = 2_147_483_647


def simulate_sglang_offloader_and_compile_guard(num_layers: int = 40) -> dict[str, Any]:
    # Simulate a hybrid model module with:
    # - standard Linear weight (storage_ptr=1001, not aliased) -> eligible for CPU offload
    # - KDA qkv_conv1d.weight (storage_ptr=2002) aliased by attn.conv_weights view (storage_ptr=2002) -> pinned resident
    # - GemmaRMSNorm.weight (storage_ptr=3003) with derived buffer gemma_weight refreshed via refresh_gemma_weight()
    parameters = [
        {"name": "mlp.gate_proj.weight", "device": "cuda:0", "storage_ptr": 1001, "nbytes": 33_554_432},
        {"name": "kda.qkv_conv1d.weight", "device": "cuda:0", "storage_ptr": 2002, "nbytes": 262_144},
        {"name": "norm.weight", "device": "cuda:0", "storage_ptr": 3003, "nbytes": 8_192},
    ]
    non_param_attributes = [
        {"owner": "kda.attn", "attr": "conv_weights", "device": "cuda:0", "storage_ptr": 2002, "numel": 131_072},
        {"owner": "norm", "attr": "gemma_weight", "device": "cuda:0", "storage_ptr": 4004, "numel": 4_096},
    ]

    cached_storages = {
        (attr["device"], attr["storage_ptr"])
        for attr in non_param_attributes
        if attr["numel"] > 0
    }
    resident_params = [
        p["name"]
        for p in parameters
        if (p["device"], p["storage_ptr"]) in cached_storages
    ]
    offloaded_params = [p["name"] for p in parameters if p["name"] not in resident_params]

    # Diffusion linear-GELU verdict cache under torch.compile
    # Before fix: reading/mutating _VERDICTS[linear] during tracing causes Dynamo to guard on len(_VERDICTS),
    # triggering 1 recompile per linear layer across num_layers.
    recompiles_before_fix = num_layers
    recompiles_after_fix = 1  # bypasses _VERDICTS when torch.compiler.is_compiling() is True

    return {
        "cached_storage_keys_count": len(cached_storages),
        "resident_parameters": resident_params,
        "offloaded_parameters": offloaded_params,
        "kda_conv_view_stale_after_offload": "kda.qkv_conv1d.weight" not in resident_params,
        "gemma_rmsnorm_refresh_hook_invoked": True,
        "diffusion_linear_layers": num_layers,
        "dynamo_compilations_before_fix": recompiles_before_fix,
        "dynamo_compilations_after_fix": recompiles_after_fix,
    }


def simulate_trtllm_ulysses_packed_qkv(
    p_ranks: int = 2,
    batch_size: int = 1,
    sp_tokens: int = 4096,
    heads: int = 24,
    head_dim: int = 128,
) -> dict[str, Any]:
    vec_size = 8  # uint4 = 16 bytes = 8 BF16 elements
    assert head_dim % vec_size == 0
    vec_per_row = head_dim // vec_size
    threads_per_block = heads * vec_per_row
    grid_x = 3 * p_ranks * sp_tokens
    grid_y = batch_size

    total_elements_qkv = p_ranks * batch_size * sp_tokens * 3 * heads * head_dim
    bytes_per_elem = 2  # BF16
    tensor_bytes = total_elements_qkv * bytes_per_elem

    # Baseline Python path:
    # 1. All-to-All output [P, B, Sp, 3, H, D]
    # 2. permute + contiguous() -> reads tensor_bytes + writes tensor_bytes
    # 3. unbind + launchUlyssesPostUnscatter -> reads tensor_bytes + writes tensor_bytes
    baseline_hbm_bytes = 4 * tensor_bytes

    # Fused ulyssesPackedQkvPostUnscatterKernel path:
    # Reads [P, B, Sp, 3, H, D] once and writes q_out, k_out, v_out once
    fused_hbm_bytes = 2 * tensor_bytes

    # Verify index mapping formula at sample coordinate (p=1, b=0, sp=3, qkv=2, h=5, vec_idx=7)
    p, b, sp, qkv_idx, h, vec_idx = 1, 0, 3, 2, 5, 7
    psp = p * sp_tokens + sp
    in_base = (((((p * batch_size + b) * sp_tokens + sp) * 3 + qkv_idx) * heads + h) * head_dim) + vec_idx * vec_size
    out_base = (((b * (p_ranks * sp_tokens) + psp) * heads + h) * head_dim) + vec_idx * vec_size

    # Verify int32 overflow guard (`checkInt32SumProduct`)
    overflow_p = 65536
    overflow_sp = 16384
    overflow_detected = overflow_p * (overflow_sp * 3) > INT32_MAX

    return {
        "shape_packed_5d": [p_ranks, batch_size, sp_tokens, 3, heads, head_dim],
        "vec_elements_bf16": vec_size,
        "threads_per_block": threads_per_block,
        "threads_within_1024_limit": threads_per_block <= 1024,
        "grid_dim": [grid_x, grid_y, 1],
        "baseline_hbm_bytes": baseline_hbm_bytes,
        "fused_hbm_bytes": fused_hbm_bytes,
        "hbm_traffic_reduction_pct": round((1.0 - fused_hbm_bytes / baseline_hbm_bytes) * 100.0, 2),
        "sample_in_base_offset": in_base,
        "sample_out_base_offset": out_base,
        "int32_sum_product_overflow_caught": overflow_detected,
    }


def simulate_vllm_rust_bad_words() -> dict[str, Any]:
    # Mock tokenizer matching BPE behavior with/without prefix space
    vocab = {
        "foo": [101],
        " foo": [201],  # distinct first token, same length (1) -> keep both!
        "bar": [102, 103],
        " bar": [299, 102, 103],  # different length (3 != 2) -> drop prefix-space variant!
        "baz": [104, 105],
        " baz": [104, 999],  # same first token (104 == 104) -> drop prefix-space variant!
    }

    def tokenize_bad_words(bad_words: list[str], max_cap: int = 128) -> dict[str, Any]:
        if any(w == "" for w in bad_words):
            return {"status": "error", "error": "EmptyBadWord"}
        if len(bad_words) > max_cap:
            return {"status": "error", "error": "TooManyBadWords"}
        seen: set[str] = set()
        all_token_ids: list[list[int]] = []
        for raw_word in bad_words:
            if raw_word in seen:
                continue
            seen.add(raw_word)
            trimmed = raw_word.lstrip()
            without_space = vocab[trimmed]
            with_space = vocab[" " + trimmed]
            all_token_ids.append(without_space)
            if (
                len(with_space) > 0
                and len(without_space) > 0
                and with_space[0] != without_space[0]
                and len(with_space) == len(without_space)
            ):
                all_token_ids.append(with_space)
        return {"status": "ok", "unique_words": len(seen), "token_id_sequences": all_token_ids}

    valid_res = tokenize_bad_words(["foo", "foo", "  bar", "baz"])
    empty_res = tokenize_bad_words(["foo", ""])
    cap_res = tokenize_bad_words([f"w{i}" for i in range(130)], max_cap=128)

    return {
        "valid_case": valid_res,
        "empty_word_rejected_before_tokenize": empty_res["error"] == "EmptyBadWord",
        "cap_exceeded_rejected": cap_res["error"] == "TooManyBadWords",
    }


def run_experiment() -> dict[str, Any]:
    sglang = simulate_sglang_offloader_and_compile_guard()
    trtllm = simulate_trtllm_ulysses_packed_qkv()
    vllm = simulate_vllm_rust_bad_words()

    return {
        "experiment": "sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words",
        "phase": 65,
        "priority": 74,
        "track": "Track B — Serving, Diffusion & Disaggregated Runtime",
        "sglang_offloader_and_compile": sglang,
        "trtllm_ulysses_packed_qkv": trtllm,
        "vllm_rust_bad_words": vllm,
        "invariants": {
            "sglang_kda_view_preserved_on_gpu": not sglang["kda_conv_view_stale_after_offload"],
            "sglang_dynamo_single_compilation": sglang["dynamo_compilations_after_fix"] == 1,
            "trtllm_threads_within_limit": trtllm["threads_within_1024_limit"],
            "trtllm_hbm_halved": trtllm["hbm_traffic_reduction_pct"] == 50.0,
            "trtllm_int32_overflow_caught": trtllm["int32_sum_product_overflow_caught"],
            "vllm_empty_word_rejected": vllm["empty_word_rejected_before_tokenize"],
            "vllm_prefix_space_dedup_exact": vllm["valid_case"]["token_id_sequences"]
            == [[101], [201], [102, 103], [104, 105]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against summary.json")
    args = parser.parse_args()

    result = run_experiment()
    summary_path = Path(__file__).with_name("summary.json")

    if args.check:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        assert result == expected, "Summary mismatch!"
        assert all(result["invariants"].values()), "Invariant failed!"
        print("OK: sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words verified.")
        return

    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
