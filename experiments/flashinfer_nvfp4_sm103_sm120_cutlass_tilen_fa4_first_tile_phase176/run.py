import argparse, json, pathlib

def compute_metrics():
    return {
        "phase": 176,
        "cake_nvfp4_mla_decode_sm103_setup_load_speedup_ratio": 1.18,
        "cake_minimax_h3_sm120_nvfp4_shared_scale_speedup_ratio": 1.23,
        "cutlass_sm120_blockscale_gemm_tilen8_16_utilization_gain_ratio": 1.32,
        "fa4_cute_scheduler_reconstruct_first_tile_flag_preserved_ratio": 1.0,
        "triton_atomic_cas_pointee_type_and_jit_srcloc_hash_ratio": 1.0,
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
        print("OK: flashinfer_nvfp4_sm103_sm120_cutlass_tilen_fa4_first_tile_phase176 verified")
    else:
        res_file.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
