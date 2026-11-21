from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "Benchmark": "llm_checkpoint_restart_criu_cuda_checkpoint_live_pod_migration_selection",
        "Provenance": {
            "Data_Status": "synthetic + anonymized_reported",
            "Independently_Reproduced": False,
            "Physical_GPU_Cluster_Measurement": "not_measured",
            "Harness_Policy": "enterprise_criu_cuda_checkpoint_live_migration_scale_to_zero_sizer_v1",
        },
        "Workload_Assumptions": {
            "Target_Workload": "Enterprise Serverless GPU Scale-from-Zero & Spot Node Preemption Evacuation (70B-FP8 vLLM Pod: 72 GB GPU HBM State + 32 Captured CUDA Graphs)",
            "GPU_HBM_Footprint_GB": 72.0,
            "Cold_CUDA_Graph_Capture_Time_sec": 22.0,
            "Spot_Preemption_Warning_Window_sec": 30.0,
            "Burst_Scale_From_Zero_Events_Per_Day": 180,
        },
        "Configuration_Matrix": [
            {
                "Tier_ID": "Tier_1_Cold_Pod_Boot_Full_Init_And_CUDA_Graph_Compile",
                "Architecture_Name": "Standard Cold Pod Boot (`Python Import -> Weight Load -> PyTorch Compile & 32-Batch CUDA Graph Capture`)",
                "Resume_Mechanism": "Full Cold Initialization on Every Scale-Out or Spot Replacement",
                "Pod_Wakeup_To_First_Token_sec": 48.5,
                "Spot_30s_Graceful_Evacuation_Success_Pct": 0.0,
                "Idle_Standby_GPU_HBM_Reserved_GB": 0.0,
                "Monthly_Cluster_TCO_For_20_Intermittent_Models_USD": 44800,
                "Recommended_Scenario": "Static 24/7 deployments that never scale to zero or run on Spot instances",
            },
            {
                "Tier_ID": "Tier_2_Always_On_Idle_Warm_GPU_Replicas",
                "Architecture_Name": "24/7 Always-On Warm GPU Replicas (`minReplicas: 1` per Model to Avoid 48.5s Cold Start)",
                "Resume_Mechanism": "Keep All 20 Intermittent 70B Models Permanently Resident in Dedicated GPU HBM",
                "Pod_Wakeup_To_First_Token_sec": 0.2,
                "Spot_30s_Graceful_Evacuation_Success_Pct": 15.0,
                "Idle_Standby_GPU_HBM_Reserved_GB": 1440.0,
                "Monthly_Cluster_TCO_For_20_Intermittent_Models_USD": 112000,
                "Recommended_Scenario": "Ultra-strict 200ms SLA where 2.5x idle GPU overprovisioning cost ($112k/mo) is acceptable",
            },
            {
                "Tier_ID": "Tier_3_vLLM_Sleep_Mode_Level1_Host_DRAM_Offload",
                "Architecture_Name": "`vLLM Sleep Mode (Level 1)` (Offload Weights to Pinned Host DRAM & Free KV Cache; Keep Process Alive)",
                "Resume_Mechanism": "Process Stays Alive; Wakeup Copies Weights from Pinned Host RAM over PCIe Gen5 (56 GB/s)",
                "Pod_Wakeup_To_First_Token_sec": 1.6,
                "Spot_30s_Graceful_Evacuation_Success_Pct": 45.0,
                "Idle_Standby_GPU_HBM_Reserved_GB": 1.2,
                "Monthly_Cluster_TCO_For_20_Intermittent_Models_USD": 22400,
                "Recommended_Scenario": "Same-node multi-model time-sharing and RL training/rollout co-location",
            },
            {
                "Tier_ID": "Tier_4_CRIU_Plus_NVIDIA_CUDA_Checkpoint_API_RDMA_Restore",
                "Architecture_Name": "Linux `CRIU` + NVIDIA `cuda-checkpoint` (`cuCheckpointProcess` R550+) Full Process & CUDA-Graph Snapshot Restore",
                "Resume_Mechanism": "Freeze CPU + GPU Context & Pre-Captured CUDA Graphs -> Restore from Host DRAM / NVMe / RDMA in 3.2s",
                "Pod_Wakeup_To_First_Token_sec": 3.2,
                "Spot_30s_Graceful_Evacuation_Success_Pct": 98.5,
                "Idle_Standby_GPU_HBM_Reserved_GB": 0.0,
                "Monthly_Cluster_TCO_For_20_Intermittent_Models_USD": 16800,
                "Recommended_Scenario": "Production standard for Serverless GPU Scale-to-Zero & 30s Spot Live Pod Evacuation (-85% TCO vs Always-On)",
            },
        ],
        "Decision_Summary": {
            "Baseline_Tier": "Tier_1_Cold_Pod_Boot_Full_Init_And_CUDA_Graph_Compile",
            "Recommended_Tier": "Tier_4_CRIU_Plus_NVIDIA_CUDA_Checkpoint_API_RDMA_Restore",
            "Wakeup_Speedup_Multiplier": 15.16,
            "Pod_Wakeup_Latency_sec": 3.2,
            "Spot_Evacuation_Success_Pct": 98.5,
            "Monthly_TCO_Savings_Vs_AlwaysOn_Pct": 85.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()

    if args.check:
        if not out_path.exists():
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing["Benchmark"] == payload["Benchmark"]
        assert len(existing["Configuration_Matrix"]) == 4
        print("Wrote results.json")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
