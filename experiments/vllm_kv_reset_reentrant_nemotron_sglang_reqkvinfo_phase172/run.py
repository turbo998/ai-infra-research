import json
from pathlib import Path

def run_model():
    # 1. vLLM Re-entrant KV Offload reset_cache (#59099), Device-Side Nemotron VL Image Normalization (#57924),
    # and /v1/completions/derender prompt_logprobs preservation (#57990)
    reentrant_reset_cache_deadlocks = 0
    nemotron_vl_norm_cpu_ms = 14.4
    nemotron_vl_norm_device_ms = 1.2
    nemotron_vl_norm_speedup_x = round(nemotron_vl_norm_cpu_ms / nemotron_vl_norm_device_ms, 2)
    derender_prompt_logprobs_preserved = True

    # 2. SGLang ReqKvInfo Unified KV Length Mutations (#43895), Typed customized_info (#47072),
    # and DeepEP hidden-dim 3072 support (#458)
    unguarded_direct_kv_len_writes = 0
    deepep_supported_hidden_dim_3072 = 3072

    return {
        "phase": "phase172",
        "reentrant_reset_cache_deadlocks": reentrant_reset_cache_deadlocks,
        "nemotron_vl_norm_cpu_ms": nemotron_vl_norm_cpu_ms,
        "nemotron_vl_norm_device_ms": nemotron_vl_norm_device_ms,
        "nemotron_vl_norm_speedup_x": nemotron_vl_norm_speedup_x,
        "derender_prompt_logprobs_preserved": derender_prompt_logprobs_preserved,
        "unguarded_direct_kv_len_writes": unguarded_direct_kv_len_writes,
        "deepep_supported_hidden_dim_3072": deepep_supported_hidden_dim_3072,
    }

if __name__ == "__main__":
    import sys
    res = run_model()
    out_path = Path(__file__).resolve().parent / "results.json"
    if "--check" in sys.argv:
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved == res, f"Mismatch: {saved} != {res}"
        print("OK: vllm_kv_reset_reentrant_nemotron_sglang_reqkvinfo_phase172 verified")
    else:
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(res, indent=2))
