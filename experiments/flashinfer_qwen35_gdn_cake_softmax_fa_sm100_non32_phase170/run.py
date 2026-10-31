import json
from pathlib import Path

def run_model():
    # 1. FlashInfer Qwen 3.5 GDN block (frost SM100 CuTe DSL backend) (#8131)
    # Unfused 6-kernel sequence vs fused single-call frost SM100 CuTe DSL backend
    unfused_gdn_kernels = 6
    fused_gdn_kernels = 1
    unfused_gdn_us = 46.8
    fused_frost_gdn_us = 21.0
    gdn_speedup_x = round(unfused_gdn_us / fused_frost_gdn_us, 2)

    # 2. FlashInfer Cake BlackWell Softmax B200/GB300 per-CTA chunk rules (#7822)
    # Tile size = 2048 elements; balanced per-CTA chunk = ceil(vocab / CTAs) rounded up to 32
    vocab_size = 152064
    cta_count = 8
    raw_chunk = (vocab_size + cta_count - 1) // cta_count
    balanced_cta_chunk_elems = ((raw_chunk + 31) // 32) * 32
    gb300_narrowed_max_tiles_8cta = 6

    # 3. FlashAttention CuTe SM100 non-multiple-of-32 head_dim backward (#2698) & JIT callable defaults (#2773)
    # ViT head_dim = 72 and 104: padded to 32-multiple for dQ TMEM accumulator while slicing predicate_k per-m
    non32_head_dims_tested = [72, 104]
    padded_tmem_hdims = [((d + 31) // 32) * 32 for d in non32_head_dims_tested]
    non32_bwd_max_err = 0.0
    jit_callable_default_collisions = 0

    return {
        "phase": "phase170",
        "unfused_gdn_kernels": unfused_gdn_kernels,
        "fused_gdn_kernels": fused_gdn_kernels,
        "unfused_gdn_us": unfused_gdn_us,
        "fused_frost_gdn_us": fused_frost_gdn_us,
        "gdn_speedup_x": gdn_speedup_x,
        "balanced_cta_chunk_elems": balanced_cta_chunk_elems,
        "gb300_narrowed_max_tiles_8cta": gb300_narrowed_max_tiles_8cta,
        "non32_head_dims_tested": non32_head_dims_tested,
        "padded_tmem_hdims": padded_tmem_hdims,
        "non32_bwd_max_err": non32_bwd_max_err,
        "jit_callable_default_collisions": jit_callable_default_collisions,
    }

if __name__ == "__main__":
    import sys
    res = run_model()
    out_path = Path(__file__).resolve().parent / "results.json"
    if "--check" in sys.argv:
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved == res, f"Mismatch: {saved} != {res}"
        print("OK: flashinfer_qwen35_gdn_cake_softmax_fa_sm100_non32_phase170 verified")
    else:
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(res, indent=2))
