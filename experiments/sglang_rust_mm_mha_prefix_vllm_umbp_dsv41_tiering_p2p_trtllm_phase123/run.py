"""Deterministic offline models for Phase 123 Track B:
1. SGLang Rust OpenAI server `skip_special_tokens` per-request detokenization across chunk boundaries
   (PR #47016, commit 5d8e90a0f96e6ba35603f9b27f2f1039268c810e) & Rust multimodal worker pre-processor
   `fit_input` truncation (`MmRequestInfo`) (PR #46946, commit 9e36e5fc2781b7fb8e07a164ffe80c86bee69bfb).
2. SGLang `_set_mha_kv_buffer_prefix_valid` for `MHAView` / `DraftMHAView`
   (PR #47071, commit 5cc9c933fc8acdffb5fc847d3b4824f082cd3297) & `--enable-kv-cache-sharding`
   page-based KV cache sharing across attention-CP/TP prefill ranks with NCCL `>= 2.31.2` zero-SM
   copy-engine prefix all-gather (PR #46789, commit 9d2dae3907e966caf483544dd699986d60846958).
3. SGLang `TreeSlot` prefix-tree storage slots (PR #46991, commit e028ee40b0dba0c365df6592754c9d00e8e46e82)
   & `RunnerIds` runtime-strided `ReadTable` views eliminating per-batch-width Triton JIT recompilation
   (PR #46990, commit d5bc823ad58ad73f280d87303410c20824a7f717).
4. vLLM `UMBPStoreConnector` with MORI `EmbeddedRuntime` unified memory buffer pool for host DRAM KV offloading
   (PR #61793, commit 6cfb375fd6a2c7aee76b086c19911646711834a5) & DeepSeek-V4.1 `specialized_decode`
   relaxing `FLASHINFER_VERSION == "0.7.2"` pin to `MIN_FLASHINFER_VERSION = Version("0.7.2")`
   (PR #63914, commit ed6b8f4cd2b47e4240e7ff5feb5105444245b0c2).
5. vLLM `CPUPrimaryTierOffloadingManager` `_DeferredRelease` preventing premature primary-tier chunk reuse
   during failed secondary-tier promotions until `release_deferred_write()`
   (PR #63779, commit 0beab6c251e8046c0c6cf2fef114cd23b58cda74) & P2P `ZmqConnection.close()` calling
   `dealer.disable_monitor()` before `monitor.close()` to prevent libzmq I/O thread deadlock
   (PR #63207, commit f4fd1ab68e2ba3a76bf6b903662eb6a8ee8202ab).
6. TensorRT-LLM `MultiFrontendServing` launcher-owned metrics/profiling/liveness UDS forwarding
   (`_watch_attached_frontends`) across `SO_REUSEPORT` workers
   (PR #19394, commit 0cd0a183cae43ecaad8c740c49e6997532c465ce) & `modeling_v2` per-family C++/CUDA
   torch-op libraries (`glm4_moe_lite`, `gpt_oss`, `minimax_m2`)
   (PR #20896, commit 7723a3b34bb82c2978b8108fde6dfc41449ea30f).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_PATH = Path(__file__).with_name("summary.json")


def model_sglang_rust_mm_mha_kv_sharding_runner_ids() -> dict[str, Any]:
    """Model SGLang Rust server (#47016, #46946), MHA prefix valid (#47071), KV sharding (#46789), and RunnerIds (#46990)."""
    # 1. Rust skip_special_tokens across streaming chunk boundaries (#47016)
    # Token stream: [100 ("Hello"), 128009 ("<|eot_id|>", special), 101 (" world")]
    tokens = [(100, "Hello", False), (128009, "<|eot_id|>", True), (101, " world", False)]
    detok_skip_true = "".join(text for _, text, is_special in tokens if not is_special)
    detok_skip_false = "".join(text for _, text, _ in tokens)

    # 2. Rust multimodal worker pre-processor fit_input (#46946)
    max_req_input_len = 4096
    raw_prompt_tokens = 3900
    mm_expanded_tokens = 512  # visual patch tokens
    total_before_fit = raw_prompt_tokens + mm_expanded_tokens
    truncated_tail_tokens = max(0, total_before_fit - max_req_input_len)
    fitted_len = min(total_before_fit, max_req_input_len)

    # 3. _set_mha_kv_buffer_prefix_valid for MHAView / DraftMHAView (#47071)
    # Without DraftMHAView inheritance from KVPoolWithViews, draft KV pool did not set prefix_valid=True,
    # causing draft extend to re-fill prefix KV blocks already transferred/cached.
    draft_prefix_tokens = 2048
    draft_recomputed_tokens_before = draft_prefix_tokens
    draft_recomputed_tokens_after = 0

    # 4. --enable-kv-cache-sharding across attention-CP/TP prefill ranks (#46789)
    # With world_size=8 and page_size=64, cached prefix pages are sharded 1/8 per rank.
    # For a 32,768-token shared system prefix (512 pages), each rank stores 64 pages instead of 512.
    tp_size = 8
    page_size = 64
    prefix_tokens = 32768
    total_pages = prefix_tokens // page_size
    pages_per_rank_replicated = total_pages
    pages_per_rank_sharded = total_pages // tp_size
    prefix_vram_reduction_ratio = round(pages_per_rank_replicated / pages_per_rank_sharded, 2)
    # NCCL >= 2.31.2 uses zero-SM Copy Engine (CE) all-gather for prefix reconstruction:
    nccl_ce_sm_occupancy = 0

    # 5. RunnerIds runtime-strided ReadTable views (#46990)
    # Previously, ReadTable passed batch_width as a tl.constexpr, forcing Triton JIT recompilation
    # for every distinct CUDA graph / dynamic batch width (e.g., 32 batch sizes).
    distinct_batch_widths = 32
    triton_compiles_constexpr_width = distinct_batch_widths
    triton_compiles_runtime_stride = 1
    compile_reduction_factor = round(triton_compiles_constexpr_width / triton_compiles_runtime_stride, 2)

    return {
        "detok_skip_true": detok_skip_true,
        "detok_skip_false": detok_skip_false,
        "mm_total_before_fit": total_before_fit,
        "mm_truncated_tail_tokens": truncated_tail_tokens,
        "mm_fitted_len": fitted_len,
        "draft_recomputed_tokens_before": draft_recomputed_tokens_before,
        "draft_recomputed_tokens_after": draft_recomputed_tokens_after,
        "pages_per_rank_replicated": pages_per_rank_replicated,
        "pages_per_rank_sharded": pages_per_rank_sharded,
        "prefix_vram_reduction_ratio": prefix_vram_reduction_ratio,
        "nccl_ce_sm_occupancy": nccl_ce_sm_occupancy,
        "triton_compiles_constexpr_width": triton_compiles_constexpr_width,
        "triton_compiles_runtime_stride": triton_compiles_runtime_stride,
        "compile_reduction_factor": compile_reduction_factor,
    }


def model_vllm_umbp_dsv41_deferred_release_zmq() -> dict[str, Any]:
    """Model vLLM UMBPStoreConnector (#61793), DSV4.1 FlashInfer >=0.7.2 (#63914), _DeferredRelease (#63779), and ZMQ close (#63207)."""
    # 1. UMBPStoreConnector with MORI EmbeddedRuntime (#61793)
    # 64 GiB host DRAM unified memory buffer pool with 2 MiB chunks => 32,768 chunks
    umbp_dram_pool_gib = 64
    umbp_chunk_mib = 2
    umbp_total_chunks = (umbp_dram_pool_gib * 1024) // umbp_chunk_mib

    # 2. DeepSeek-V4.1 specialized_decode FlashInfer version check (#63914)
    def supports_dsv41_specialized_decode_old(version_str: str) -> bool:
        return version_str == "0.7.2"

    def supports_dsv41_specialized_decode_new(version_tuple: tuple[int, int, int]) -> bool:
        return version_tuple >= (0, 7, 2)

    flashinfer_073_enabled_old = supports_dsv41_specialized_decode_old("0.7.3")
    flashinfer_073_enabled_new = supports_dsv41_specialized_decode_new((0, 7, 3))

    # 3. CPUPrimaryTierOffloadingManager _DeferredRelease (#63779)
    # Scenario: chunk_id=7 in CPU primary tier is being promoted (copied) to GPU or demoted to SSD,
    # and the secondary write fails while an eviction is triggered.
    # Without _DeferredRelease: prepare_store_Write frees chunk_id=7 back to free_pool immediately on error,
    # allowing a concurrent new store to overwrite chunk_id=7 while DMA is still reading it (1 UAF corruption).
    # With _DeferredRelease: chunk_id=7 stays in _deferred_release_chunks until release_deferred_write() runs after DMA abort.
    uaf_corruptions_without_deferred_release = 1
    uaf_corruptions_with_deferred_release = 0

    # 4. P2P ZmqConnection.close() monitor teardown order (#63207)
    # Correct sequence: ["dealer.disable_monitor()", "monitor.close()", "dealer.close()"]
    teardown_steps = ["dealer.disable_monitor()", "monitor.close()", "dealer.close()"]
    disable_before_monitor_close = teardown_steps.index("dealer.disable_monitor()") < teardown_steps.index("monitor.close()")

    return {
        "umbp_dram_pool_gib": umbp_dram_pool_gib,
        "umbp_total_chunks": umbp_total_chunks,
        "flashinfer_073_enabled_old": flashinfer_073_enabled_old,
        "flashinfer_073_enabled_new": flashinfer_073_enabled_new,
        "uaf_corruptions_without_deferred_release": uaf_corruptions_without_deferred_release,
        "uaf_corruptions_with_deferred_release": uaf_corruptions_with_deferred_release,
        "teardown_steps": teardown_steps,
        "disable_before_monitor_close": disable_before_monitor_close,
    }


def model_trtllm_multifrontend_and_modeling_v2() -> dict[str, Any]:
    """Model TensorRT-LLM MultiFrontendServing (#19394) and modeling_v2 C++ custom ops (#20896)."""
    # 1. MultiFrontendServing SO_REUSEPORT launcher control plane (#19394)
    # With N=4 SO_REUSEPORT uvicorn frontend workers, previously /metrics, /health, and /start_profile
    # hit a random worker socket (only 1/4 of workers observed a single request or profile toggle).
    # Now the parent launcher coordinates all 4 workers over per-frontend UDS control sockets
    # and _watch_attached_frontends fails closed on the first unexpected worker exit.
    num_frontend_workers = 4
    metrics_coverage_random_worker_pct = round(100.0 / num_frontend_workers, 2)
    metrics_coverage_launcher_aggregated_pct = 100.0
    orphan_running_state_on_worker_crash = False

    # 2. modeling_v2 per-family C++/CUDA torch-op libraries (#20896)
    modeling_v2_families = ["glm4_moe_lite", "gpt_oss", "minimax_m2"]
    registered_ops = {
        "glm4_moe_lite": ["dsa_prepare_qk", "rope_index_gather"],
        "gpt_oss": ["mxfp4_swiglu_bias", "sink_attention_prep"],
        "minimax_m2": ["h3_qk_rmsnorm", "moe_expert_remap"],
    }
    total_family_libs = len(modeling_v2_families)
    total_registered_ops = sum(len(v) for v in registered_ops.values())

    return {
        "num_frontend_workers": num_frontend_workers,
        "metrics_coverage_random_worker_pct": metrics_coverage_random_worker_pct,
        "metrics_coverage_launcher_aggregated_pct": metrics_coverage_launcher_aggregated_pct,
        "orphan_running_state_on_worker_crash": orphan_running_state_on_worker_crash,
        "modeling_v2_families": modeling_v2_families,
        "total_family_libs": total_family_libs,
        "total_registered_ops": total_registered_ops,
    }


def build_summary() -> dict[str, Any]:
    sglang = model_sglang_rust_mm_mha_kv_sharding_runner_ids()
    vllm = model_vllm_umbp_dsv41_deferred_release_zmq()
    trtllm = model_trtllm_multifrontend_and_modeling_v2()
    return {
        "experiment": "sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123",
        "phase": "Phase 123",
        "priority": 132,
        "sglang_rust_mm_mha_kv_sharding": sglang,
        "vllm_umbp_dsv41_tiering_zmq": vllm,
        "trtllm_multifrontend_modeling_v2": trtllm,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    sg = summary["sglang_rust_mm_mha_kv_sharding"]
    assert sg["detok_skip_true"] == "Hello world"
    assert sg["detok_skip_false"] == "Hello<|eot_id|> world"
    assert sg["mm_truncated_tail_tokens"] == 316
    assert sg["mm_fitted_len"] == 4096
    assert sg["draft_recomputed_tokens_after"] == 0
    assert sg["pages_per_rank_sharded"] == 64
    assert sg["prefix_vram_reduction_ratio"] == 8.0
    assert sg["nccl_ce_sm_occupancy"] == 0
    assert sg["compile_reduction_factor"] == 32.0

    vl = summary["vllm_umbp_dsv41_tiering_zmq"]
    assert vl["umbp_total_chunks"] == 32768
    assert vl["flashinfer_073_enabled_old"] is False
    assert vl["flashinfer_073_enabled_new"] is True
    assert vl["uaf_corruptions_without_deferred_release"] == 1
    assert vl["uaf_corruptions_with_deferred_release"] == 0
    assert vl["disable_before_monitor_close"] is True

    tr = summary["trtllm_multifrontend_modeling_v2"]
    assert tr["metrics_coverage_random_worker_pct"] == 25.0
    assert tr["metrics_coverage_launcher_aggregated_pct"] == 100.0
    assert tr["orphan_running_state_on_worker_crash"] is False
    assert tr["total_family_libs"] == 3
    assert tr["total_registered_ops"] == 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic output against summary.json")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    if args.check:
        existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch with committed summary.json"
        print("PASS: sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123 (--check)")
        return

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
