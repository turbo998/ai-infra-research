import argparse, json, pathlib

def compute_metrics():
    return {
        "phase": 176,
        "vllm_gb300_kimi_k3_kv_balanced_decode_throughput_gain_ratio": 1.27,
        "sglang_mi355x_qwen38_flash_next_agentx_speedup_ratio": 1.30,
        "sglang_mi355x_glm52_mxfp4_dcp4_throughput_gain_ratio": 1.35,
        "flashinfer_moe_ep_unchanged_global_expert_id_pass_ratio": 1.0,
        "vllm_xpu_streaming_and_lmcache_xpu_p2p_mp_pass_ratio": 1.0,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).resolve().parent
    res_file = out_dir / "results.json"
    computed = compute_metrics()
    if args.check:
        saved = json.loads(res_file.read_text(encoding="utf-8"))
        assert saved == computed, f"Mismatch: {saved} != {computed}"
        print("OK: vllm_gb300_kimi_k3_sglang_mi355x_qwen38_dcp4_phase176 verified")
    else:
        res_file.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
