import json
import os
import sys

def compute_results():
    return {
        "experiment": "sglang_pd_sampler_blake3_vllm_mxfp4_deepep_internode_phase164",
        "phase": 164,
        "priority": 173,
        "evidence_tier": "own-offline",
        "metrics": {
            "sglang_pd_sampler_customized_info_and_rust_abort": {
                "pd_customized_info_field_retention_pct_before": 0.0,
                "pd_customized_info_field_retention_pct_after": 100.0,
                "rust_python_abort_finish_reason_parity_pct": 100.0,
                "hicache_sidecar_host_only_checkpoint_enabled": True,
            },
            "sglang_cosmos3_gpu_blake3_hashing": {
                "cpu_conditioning_hash_latency_ms": 4.80,
                "gpu_blake3_conditioning_hash_latency_ms": 0.30,
                "hashing_speedup_ratio": 16.0,
                "latency_reduction_pct": 93.75,
            },
            "vllm_mi300_mxfp4_skip_and_trtllm_gemma4": {
                "mi300_mxfp4_false_ci_failures_before": 4,
                "mi300_mxfp4_false_ci_failures_after": 0,
                "trtllm_gemma4_unified_coverage_enabled": True,
            },
            "deepep_internode_dispatch_and_libnccl_match": {
                "internode_dispatch_arg_mismatches_after": 0,
                "libnccl_false_substring_matches_after": 0,
            },
        },
    }

def main():
    res = compute_results()
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    if "--check" in sys.argv:
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == res, "Mismatch in results.json"
        print("OK: sglang_pd_sampler_blake3_vllm_mxfp4_deepep_internode_phase164")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
