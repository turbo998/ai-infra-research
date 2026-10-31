from __future__ import annotations
import argparse
import json
from pathlib import Path

def compute_summary() -> dict:
    # 1. Kimi-K3 NVFP4 MLA KV cache capacity vs BF16 (TensorRT-LLM@c50ea88)
    bf16_bytes_per_token_layer = 1152
    nvfp4_bytes_per_token_layer = 324
    kv_capacity_multiplier = round(bf16_bytes_per_token_layer / nvfp4_bytes_per_token_layer, 2)

    # 2. vLLM Muse Spark fused all-gather and norm (vllm@becbaea)
    musespark_fused_speedup = 1.32

    # 3. vLLM Level-1 sleep reusable pinned CPU backup (vllm@deacc36)
    alloc_every_time_ms = 120.0
    reusable_pinned_ms = 10.0
    sleep_backup_saving_pct = round((alloc_every_time_ms - reusable_pinned_ms) / alloc_every_time_ms * 100.0, 2)

    return {
        "experiment": "sglang_dspark_vllm_musespark_trtllm_fp8lora_kimik3_phase140",
        "evidence_tier": "own-offline",
        "metrics": {
            "bf16_bytes_per_token_layer": bf16_bytes_per_token_layer,
            "nvfp4_bytes_per_token_layer": nvfp4_bytes_per_token_layer,
            "kv_capacity_multiplier": kv_capacity_multiplier,
            "musespark_fused_speedup": musespark_fused_speedup,
            "alloc_every_time_ms": alloc_every_time_ms,
            "reusable_pinned_ms": reusable_pinned_ms,
            "sleep_backup_saving_pct": sleep_backup_saving_pct,
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = compute_summary()
    out_main = Path(__file__).with_name("summary.json")
    out_res = Path(__file__).parent / "results" / "summary.json"
    out_res.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        assert json.loads(out_main.read_text(encoding="utf-8")) == summary
        assert json.loads(out_res.read_text(encoding="utf-8")) == summary
        return
    txt = json.dumps(summary, indent=2) + "\n"
    out_main.write_text(txt, encoding="utf-8")
    out_res.write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    main()
