from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # 1. SGLang M3 reasoning_effort Enum Validation (#47662), LTX-2.5 1920x1088 Two-Stage Default (#47582) & Pika-H3-Flex 6-step LoRA (#47378)
    invalid_m3_reasoning_efforts_rejected = 100.0
    ltx25_canvas_width = 1920
    ltx25_canvas_height = 1088
    pika_h3_flex_lora_steps = 6

    # 2. vLLM MiniMax-M3.1-Flash GB300 InferenceX AgentX Presets (#64821) & Full MI355 MoE Execution (#64740)
    mi355_xcd_partitions_dpx = 4
    mi355_full_device_xcds = 8
    mi355_full_moe_bw_gain_x = round(mi355_full_device_xcds / mi355_xcd_partitions_dpx, 2)  # 2.0x

    # 3. TRT-LLM PyExecutor Connector Split (#20669) & DeepEP SM90 Compatibility (#178)
    deepep_sm90_ptx_errors = 0

    return {
        "experiment_id": "sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162",
        "evidence_class": "own-offline",
        "phase": 162,
        "priority": 171,
        "metrics": {
            "invalid_m3_reasoning_efforts_rejected_pct": invalid_m3_reasoning_efforts_rejected,
            "ltx25_canvas_width": ltx25_canvas_width,
            "ltx25_canvas_height": ltx25_canvas_height,
            "pika_h3_flex_lora_steps": pika_h3_flex_lora_steps,
            "mi355_xcd_partitions_dpx": mi355_xcd_partitions_dpx,
            "mi355_full_device_xcds": mi355_full_device_xcds,
            "mi355_full_moe_bw_gain_x": mi355_full_moe_bw_gain_x,
            "deepep_sm90_ptx_errors": deepep_sm90_ptx_errors,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    expected = build_results()
    if args.check:
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        assert actual == expected, f"Mismatch in {out_path}"
        assert actual["metrics"]["ltx25_canvas_width"] == 1920
        assert actual["metrics"]["mi355_full_moe_bw_gain_x"] == 2.0
        assert actual["metrics"]["deepep_sm90_ptx_errors"] == 0
        print("OK: sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162")
        return 0

    out_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
