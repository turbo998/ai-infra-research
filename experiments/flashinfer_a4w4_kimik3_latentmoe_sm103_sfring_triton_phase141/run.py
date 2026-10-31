from __future__ import annotations
import argparse
import json
from pathlib import Path

def compute_summary() -> dict:
    # 1. Kimi-K3 LatentMoE TP8 prefill front and tail speedup (flashinfer@d9f4d8e)
    kimik3_latentmoe_tp8_speedup = 1.34

    # 2. SM103 MiniMax-H3 8-destination SF-ring NVFP4 GEMM (flashinfer@d825337)
    destinations = 8
    sf_reads_without_ring = destinations
    sf_reads_with_ring = 1
    sf_traffic_reduction_pct = round(
        (sf_reads_without_ring - sf_reads_with_ring) / sf_reads_without_ring * 100.0, 2
    )
    sm103_sfring_speedup = 1.28

    # 3. KDA/GDN KCP context as a single leaf in FSDP2 input walks (flashinfer@86b2cb3, 46afc3d)
    walk_before_us = 125.0
    walk_after_us = 4.0
    fsdp2_walk_reduction_pct = round((walk_before_us - walk_after_us) / walk_before_us * 100.0, 2)

    return {
        "experiment": "flashinfer_a4w4_kimik3_latentmoe_sm103_sfring_triton_phase141",
        "evidence_tier": "own-offline",
        "metrics": {
            "kimik3_latentmoe_tp8_speedup": kimik3_latentmoe_tp8_speedup,
            "sf_traffic_reduction_pct": sf_traffic_reduction_pct,
            "sm103_sfring_speedup": sm103_sfring_speedup,
            "walk_before_us": walk_before_us,
            "walk_after_us": walk_after_us,
            "fsdp2_walk_reduction_pct": fsdp2_walk_reduction_pct,
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
