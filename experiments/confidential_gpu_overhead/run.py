"""CPU-only offline model for Confidential GPU (CC-On / PPCIE), RIM/SPDM attestation, and bounce-buffer overhead.

This experiment does not require confidential GPU hardware or NVML/NSCQ drivers. It evaluates:
1. Reference Integrity Manifest (RIM) golden measurement vs runtime SPDM measurement verification rules
   implemented in `NVIDIA/nvtrust` (`guest_tools/gpu_verifiers/local_gpu_verifier/src/verifier/verifier.py`):
   - Active index conflict detection between Driver RIM and VBIOS RIM (`InvalidMeasurementIndexError`).
   - Conditional skipping of measurement index 35 when `OPAQUE_FIELD_ID_NVDEC0_STATUS == DISABLED`.
   - Alternative digest list matching and byte-size validation (`size == len(hex_str) // 2`).
2. Protected PCIe (PPCIE) 8-GPU + 4-NVSwitch bidirectional PDI (`x-nvidia-gpu-switch-pdis`,
   `x-nvidia-switch-pdi`, `x-nvidia-switch-gpu-pdis`) topology graph validation rules implemented in
   `guest_tools/ppcie-verifier-sdk-cpp/ppcie/verifier/src/topology/validate_topology.py` and fail-closed
   GPU ready-state gating (`verification.py`).
3. Host-to-Device (H2D) / Device-to-Host (D2H) staging bounce-buffer + AES-GCM-256 CPU/CE encryption overhead
   and multi-GPU collective communication latency across four modes:
   - `standard_non_cc`: Direct DMA over PCIe Gen5 (`56 GB/s`) and unencrypted NVLink (`400 GB/s`)
   - `single_gpu_cc_on_sw_aes`: CC-On with CPU SW bounce-buffer memcpy + AES-GCM (`8.5 GB/s` effective H2D, GPUDirect P2P disabled)
   - `single_gpu_cc_on_hw_dma_opt`: CC-On with pinned staging + pipelined CE AES-GCM (`18.4 GB/s` effective H2D)
   - `multi_gpu_ppcie_nvswitch`: 8-GPU + 4-NVSwitch PPCIE mode (link-encrypted NVLink P2P restored at `372 GB/s`, H2D via CC bounce buffer)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"
DISABLED_PDI = "0000000000000000"


def load_input(path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_rim_measurements(case: dict[str, Any]) -> dict[str, Any]:
    driver_rim = case["driver_rim"]
    vbios_rim = case["vbios_rim"]
    runtime_msr = {int(k): v for k, v in case["runtime_measurements"].items()}
    nvdec0_status = case.get("nvdec0_status", "ENABLED")
    is_msr_35_valid = nvdec0_status != "DISABLED"

    golden: dict[int, dict[str, Any]] = {}
    for idx_str, item in driver_rim.items():
        idx = int(idx_str)
        if item["active"]:
            golden[idx] = {"item": item, "source": "Driver"}

    for idx_str, item in vbios_rim.items():
        idx = int(idx_str)
        if item["active"] and idx in golden:
            return {
                "case_id": case["id"],
                "verified": False,
                "failure_reason": f"InvalidMeasurementIndexError: index {idx} active in both Driver and VBIOS RIM",
                "mismatched_indices": [idx],
                "skipped_msr_35": not is_msr_35_valid,
            }
        if item["active"]:
            golden[idx] = {"item": item, "source": "Firmware"}

    if len(golden) > len(runtime_msr):
        return {
            "case_id": case["id"],
            "verified": False,
            "failure_reason": "GoldenMeasurementCountExceedsRuntime",
            "mismatched_indices": [],
            "skipped_msr_35": not is_msr_35_valid,
        }

    mismatched: list[int] = []
    for idx, entry in sorted(golden.items()):
        if idx == 35 and not is_msr_35_valid:
            continue
        g_item = entry["item"]
        rt_val = runtime_msr.get(idx, "")
        matched = False
        for alt in g_item["values"]:
            if alt == rt_val and int(g_item["size"]) == len(rt_val) // 2:
                matched = True
                break
        if not matched:
            mismatched.append(idx)

    return {
        "case_id": case["id"],
        "verified": len(mismatched) == 0,
        "failure_reason": "OK" if len(mismatched) == 0 else f"MismatchedIndices:{mismatched}",
        "mismatched_indices": mismatched,
        "skipped_msr_35": not is_msr_35_valid,
    }


def verify_ppcie_topology(case: dict[str, Any]) -> dict[str, Any]:
    expected_gpus = int(case.get("expected_gpu_count", 8))
    expected_switches = int(case.get("expected_switch_count", 4))
    gpu_claims = case["gpu_claims"]
    switch_claims = case["switch_claims"]

    if len(gpu_claims) != expected_gpus or len(switch_claims) != expected_switches:
        return {
            "case_id": case["id"],
            "topology_valid": False,
            "gpu_ready_state_enabled": False,
            "failure_reason": f"DeviceCountMismatch:gpus={len(gpu_claims)},switches={len(switch_claims)}",
        }

    unique_switches_from_gpus: set[str] = set()
    for g_claim in gpu_claims:
        pdis = {p.upper() for p in g_claim["x-nvidia-gpu-switch-pdis"]}
        pdis.discard(DISABLED_PDI)
        if len(pdis) != expected_switches:
            return {
                "case_id": case["id"],
                "topology_valid": False,
                "gpu_ready_state_enabled": False,
                "failure_reason": f"GpuSwitchPdiCountMismatch:found={len(pdis)},expected={expected_switches}",
            }
        unique_switches_from_gpus.update(pdis)

    if len(unique_switches_from_gpus) != expected_switches:
        return {
            "case_id": case["id"],
            "topology_valid": False,
            "gpu_ready_state_enabled": False,
            "failure_reason": f"TotalUniqueSwitchesFromGpusMismatch:{len(unique_switches_from_gpus)}",
        }

    unique_gpus_from_switches: set[str] = set()
    switch_pdis: set[str] = set()
    for s_claim in switch_claims:
        s_pdi = s_claim["x-nvidia-switch-pdi"].upper()
        g_pdis = {p.upper() for p in s_claim["x-nvidia-switch-gpu-pdis"]}
        if len(g_pdis) != expected_gpus:
            return {
                "case_id": case["id"],
                "topology_valid": False,
                "gpu_ready_state_enabled": False,
                "failure_reason": f"SwitchGpuPdiCountMismatch:found={len(g_pdis)},expected={expected_gpus}",
            }
        unique_gpus_from_switches.update(g_pdis)
        switch_pdis.add(s_pdi)

    if len(unique_gpus_from_switches) != expected_gpus:
        return {
            "case_id": case["id"],
            "topology_valid": False,
            "gpu_ready_state_enabled": False,
            "failure_reason": f"TotalUniqueGpusFromSwitchesMismatch:{len(unique_gpus_from_switches)}",
        }

    if len(switch_pdis) != expected_switches:
        return {
            "case_id": case["id"],
            "topology_valid": False,
            "gpu_ready_state_enabled": False,
            "failure_reason": f"UniqueSwitchPdiMismatch:{len(switch_pdis)}",
        }

    unknown_switches = unique_switches_from_gpus - switch_pdis
    if unknown_switches:
        return {
            "case_id": case["id"],
            "topology_valid": False,
            "gpu_ready_state_enabled": False,
            "failure_reason": f"UnknownSwitchPdisReportedByGpu:{sorted(unknown_switches)}",
        }

    return {
        "case_id": case["id"],
        "topology_valid": True,
        "gpu_ready_state_enabled": True,
        "failure_reason": "OK",
    }


def evaluate_cc_dataplane_overhead(spec: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_h2d_mib = float(spec["prompt_kv_offload_h2d_mib"])
    allreduce_mib = float(spec["tp_allreduce_payload_mib_per_token"])
    gpu_compute_ms = float(spec["gpu_decode_compute_ms_per_token"])
    prefill_compute_ms = float(spec["gpu_prefill_compute_ms"])
    attestation_cold_ms = float(spec["spdm_rim_ocsp_cold_attestation_ms"])

    rows: list[dict[str, Any]] = []
    for mode in spec["modes"]:
        h2d_bw_gbps = float(mode["effective_h2d_bw_gbps"])
        p2p_bw_gbps = float(mode["effective_gpu_p2p_bw_gbps"])
        auth_tag_overhead_ratio = float(mode["aes_gcm_metadata_overhead_ratio"])

        h2d_bytes = prompt_h2d_mib * (1024 ** 2) * (1.0 + auth_tag_overhead_ratio)
        h2d_transfer_ms = (h2d_bytes / (h2d_bw_gbps * 1e9)) * 1000.0
        ttft_ms = prefill_compute_ms + h2d_transfer_ms

        allreduce_bytes = allreduce_mib * (1024 ** 2) * (1.0 + auth_tag_overhead_ratio)
        allreduce_ms = (allreduce_bytes / (p2p_bw_gbps * 1e9)) * 1000.0
        tpot_ms = gpu_compute_ms + allreduce_ms
        decode_tps = 1000.0 / tpot_ms

        rows.append(
            {
                "id": mode["id"],
                "label": mode["label"],
                "effective_h2d_bw_gbps": h2d_bw_gbps,
                "effective_gpu_p2p_bw_gbps": p2p_bw_gbps,
                "cold_attestation_barrier_ms": attestation_cold_ms if mode["requires_attestation"] else 0.0,
                "h2d_staging_transfer_ms": round(h2d_transfer_ms, 4),
                "ttft_with_host_kv_load_ms": round(ttft_ms, 4),
                "tp_allreduce_ms_per_token": round(allreduce_ms, 4),
                "tpot_ms": round(tpot_ms, 4),
                "decode_tokens_per_sec": round(decode_tps, 4),
                "ttft_slowdown_vs_non_cc": 0.0,
                "tpot_slowdown_vs_non_cc": 0.0,
            }
        )

    base_ttft = next(r["ttft_with_host_kv_load_ms"] for r in rows if r["id"] == "standard_non_cc")
    base_tpot = next(r["tpot_ms"] for r in rows if r["id"] == "standard_non_cc")
    for r in rows:
        r["ttft_slowdown_vs_non_cc"] = round(r["ttft_with_host_kv_load_ms"] / base_ttft, 4)
        r["tpot_slowdown_vs_non_cc"] = round(r["tpot_ms"] / base_tpot, 4)

    return rows


def run_experiment(spec: dict[str, Any]) -> dict[str, Any]:
    rim_cases = [verify_rim_measurements(c) for c in spec["rim_verification_cases"]]
    topo_cases = [verify_ppcie_topology(c) for c in spec["ppcie_topology_cases"]]
    dataplane_rows = evaluate_cc_dataplane_overhead(spec["dataplane_section"])

    non_cc = next(r for r in dataplane_rows if r["id"] == "standard_non_cc")
    sw_cc = next(r for r in dataplane_rows if r["id"] == "single_gpu_cc_on_sw_aes")
    hw_cc = next(r for r in dataplane_rows if r["id"] == "single_gpu_cc_on_hw_dma_opt")
    ppcie = next(r for r in dataplane_rows if r["id"] == "multi_gpu_ppcie_nvswitch")

    return {
        "experiment_id": "confidential_gpu_overhead",
        "measurement_type": "own-offline",
        "Clear_Boundary": "CPU-only analytical and protocol verification model derived from NVIDIA/nvtrust source code; not a physical H100/B200 TEE benchmark.",
        "rim_verification_results": rim_cases,
        "ppcie_topology_results": topo_cases,
        "dataplane_overhead_comparison": dataplane_rows,
        "headline_findings": {
            "non_cc_ttft_ms": non_cc["ttft_with_host_kv_load_ms"],
            "sw_cc_ttft_ms": sw_cc["ttft_with_host_kv_load_ms"],
            "hw_cc_ttft_ms": hw_cc["ttft_with_host_kv_load_ms"],
            "sw_cc_tpot_ms": sw_cc["tpot_ms"],
            "ppcie_tpot_ms": ppcie["tpot_ms"],
            "non_cc_tpot_ms": non_cc["tpot_ms"],
            "ppcie_vs_sw_cc_decode_speedup": round(ppcie["decode_tokens_per_sec"] / sw_cc["decode_tokens_per_sec"], 4),
        },
    }


def write_csv(results: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "id",
        "effective_h2d_bw_gbps",
        "effective_gpu_p2p_bw_gbps",
        "h2d_staging_transfer_ms",
        "ttft_with_host_kv_load_ms",
        "tp_allreduce_ms_per_token",
        "tpot_ms",
        "decode_tokens_per_sec",
        "ttft_slowdown_vs_non_cc",
        "tpot_slowdown_vs_non_cc",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["dataplane_overhead_comparison"]:
            writer.writerow({k: row[k] for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    spec = load_input(args.input)
    results = run_experiment(spec)

    if args.check:
        expected = json.loads(args.output_json.read_text(encoding="utf-8"))
        if results != expected:
            raise SystemExit("confidential_gpu_overhead results.json is out of date; re-run without --check.")
        print("confidential_gpu_overhead check passed.")
        return 0

    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(results, args.output_csv)
    print(f"Wrote {args.output_json} and {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
