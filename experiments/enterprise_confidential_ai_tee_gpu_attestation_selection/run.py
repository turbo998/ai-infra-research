from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Confidential AI (CPU+GPU TEE, Remote Attestation & Secure Key Release) Selection (Round 386)
    # Workload: 70B Regulated Enterprise LLM (Core Chip IP / Sovereign Banking / Healthcare PHI, 100 RPS peak).

    # Tier 1: Standard Cloud GPU VM (Private Link + TLS/Disk Encryption, Plaintext Host DRAM & GPU HBM)
    t1_throughput_overhead_pct = 0.0
    t1_deployment_lead_time_days = 1
    t1_data_in_use_memory_protection_pct = 0.0  # Vulnerable to host hypervisor/memory dump & unencrypted weights in RAM
    t1_monthly_tco_usd = 24_000

    # Tier 2: 100% Air-Gapped On-Premises Self-Built GPU Data Center
    t2_throughput_overhead_pct = 0.0
    t2_deployment_lead_time_days = 270  # 9 months procurement + data center power/cooling retrofit
    t2_data_in_use_memory_protection_pct = 95.0  # Physical isolation, but still vulnerable to local root/insider admin
    t2_monthly_tco_usd = 62_000  # High CapEx amortization + idle peak buffer + 24/7 hardware ops

    # Tier 3: Cloud Confidential VM + NVIDIA H100/H200 CC Mode (No Automated Remote Attestation / SKR)
    t3_throughput_overhead_pct = 5.4  # PCIe bounce-buffer AES-GCM-256 encryption overhead on 70B LLM
    t3_deployment_lead_time_days = 2
    t3_data_in_use_memory_protection_pct = 92.0  # Hardware memory encryption, but lacks cryptographic SKR gate
    t3_monthly_tco_usd = 26_200

    # Tier 4: Zero-Trust Confidential AI Enclave (CPU SEV-SNP/TDX + GPU H100/H200 CC + Remote Attestation + HSM SKR)
    t4_throughput_overhead_pct = 5.8
    t4_deployment_lead_time_days = 2
    t4_data_in_use_memory_protection_pct = 100.0  # Cryptographic proof + HSM key release only into verified TEE
    t4_monthly_tco_usd = 27_800

    tco_savings_t4_vs_t2_pct = round((t2_monthly_tco_usd - t4_monthly_tco_usd) / t2_monthly_tco_usd * 100.0, 1)
    lead_time_speedup_t4_vs_t2_x = round(t2_deployment_lead_time_days / t4_deployment_lead_time_days, 1)
    price_premium_t4_vs_t1_pct = round((t4_monthly_tco_usd - t1_monthly_tco_usd) / t1_monthly_tco_usd * 100.0, 1)

    tiers = [
        {
            "tier_id": "tier1_standard_cloud_gpu_plaintext_ram",
            "name": "Tier 1: 标准云端 GPU 虚拟机 (Private Link + 磁盘加密，但内存与显存运行态明文裸奔)",
            "throughput_overhead_pct": t1_throughput_overhead_pct,
            "deployment_lead_time_days": t1_deployment_lead_time_days,
            "data_in_use_protection_score_pct": t1_data_in_use_memory_protection_pct,
            "monthly_tco_usd": t1_monthly_tco_usd,
            "model_ip_protection": "Plaintext Weights on Disk/RAM (High IP Theft Risk)",
            "status": "Fails Strict Data-in-Use / Sovereign Audits",
        },
        {
            "tier_id": "tier2_airgapped_onprem_datacenter",
            "name": "Tier 2: 100% 物理隔离本地自建机房 (Air-Gapped On-Prem，采购与机房改造周期长达 9 个月)",
            "throughput_overhead_pct": t2_throughput_overhead_pct,
            "deployment_lead_time_days": t2_deployment_lead_time_days,
            "data_in_use_protection_score_pct": t2_data_in_use_memory_protection_pct,
            "monthly_tco_usd": t2_monthly_tco_usd,
            "model_ip_protection": "Physical Custody Only (Vulnerable to Local Root Admin)",
            "status": "Slowest Delivery (270 Days) & Highest TCO ($62K/mo)",
        },
        {
            "tier_id": "tier3_cloud_cvm_gpu_cc_basic",
            "name": "Tier 3: 云端机密虚拟机 + GPU 硬件机密计算 (AMD SEV-SNP/TDX + H100/H200 CC，无密钥绑定)",
            "throughput_overhead_pct": t3_throughput_overhead_pct,
            "deployment_lead_time_days": t3_deployment_lead_time_days,
            "data_in_use_protection_score_pct": t3_data_in_use_memory_protection_pct,
            "monthly_tco_usd": t3_monthly_tco_usd,
            "model_ip_protection": "Encrypted RAM/PCIe, Unattested Boot Key",
            "status": "Confidential Compute Baseline",
        },
        {
            "tier_id": "tier4_confidential_ai_remote_attestation_skr",
            "name": "Tier 4: 零信任机密 AI 飞地 (CPU+GPU 硬件 TEE + 双端远程证明 + HSM 安全密钥释放 SKR)",
            "throughput_overhead_pct": t4_throughput_overhead_pct,
            "deployment_lead_time_days": t4_deployment_lead_time_days,
            "data_in_use_protection_score_pct": t4_data_in_use_memory_protection_pct,
            "monthly_tco_usd": t4_monthly_tco_usd,
            "model_ip_protection": "Encrypted Model Image + HSM Key Released Only into Verified Enclave",
            "status": "Regulated & Sovereign Cloud Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_confidential_ai_tee_gpu_attestation_selection",
        "round": 386,
        "title": "Enterprise Confidential AI (CPU+GPU TEE, Remote Attestation & HSM Key Release) Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_nvidia_h100_cc_and_azure_confidential_gpu_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model_scale": "70B-FP8",
            "encryption_cipher": "AES-GCM-256 over PCIe SPDM Session",
            "peak_concurrency_rps": 100,
        },
        "headline_findings": {
            "tier4_throughput_overhead_pct": t4_throughput_overhead_pct,
            "tier4_data_in_use_protection_pct": t4_data_in_use_memory_protection_pct,
            "tco_savings_t4_vs_t2_pct": tco_savings_t4_vs_t2_pct,
            "lead_time_speedup_t4_vs_t2_x": lead_time_speedup_t4_vs_t2_x,
            "price_premium_t4_vs_t1_pct": price_premium_t4_vs_t1_pct,
        },
        "architecture_tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json")
    args = parser.parse_args()

    results = build_results()
    out_path = Path(__file__).with_name("results.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist; run without --check first.")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != results:
            raise SystemExit("results.json mismatch with deterministic build_results().")
        print("OK: enterprise_confidential_ai_tee_gpu_attestation_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
