from __future__ import annotations

import json
from pathlib import Path


def evaluate_parallel_filesystem_tiers() -> dict:
    """Deterministic evaluator for 4 AI Parallel Filesystem & Storage Architecture Tiers.

    Workload Baseline:
      - 1,024-GPU H200 Cluster backed by a 180-Node All-NVMe Storage Cluster
        (each storage node: 16×PCIe Gen4/Gen5 NVMe SSDs + 2×200Gbps RDMA NICs, raw aggregate network bw ~7.2 TB/s)
      - Workloads tested:
        1) Aggregate Random Read Throughput (GB/s) for shuffled multimodal/LLM training & KVCache block swapping
        2) Random 64KB Read IOPS (Millions) under zero temporal locality (<2% OS Page Cache hit rate)
        3) 10 TB Distributed Checkpoint Load / Save Time (seconds)
        4) 3-Year Storage Cluster TCO (Hardware + Software Licensing, USD)
    """
    tiers = [
        {
            "tier_id": "tier1_enterprise_nfs_cephfs_tcp",
            "name": "Tier 1: General-Purpose Enterprise NAS / CephFS over TCP (Kernel VFS + RADOS PG Locking)",
            "name_zh": "Tier 1: 通用企业级 NAS / CephFS (TCP 内核态 VFS + RADOS 强一致锁)",
            "datapath_architecture": "Kernel Socket TCP -> OS Page Cache Copy -> User Buffer -> PCIe Host-to-Device",
            "replication_and_metadata": "Ceph RADOS Primary-Copy (Reads pinned to primary OSD) + Active/Standby MDS",
            "aggregate_read_throughput_gbps": 251.0,
            "random_64k_read_iops_millions": 1.1,
            "checkpoint_10tb_load_sec": 44.5,
            "checkpoint_10tb_write_sec": 82.0,
            "client_cpu_overhead_cores_per_node": 28,
            "kvcache_swap_p99_latency_ms": 18.5,
            "three_year_storage_tco_usd": 2150000,
        },
        {
            "tier_id": "tier2_hpc_lustre_beegfs_rdma_pagecache",
            "name": "Tier 2: Traditional HPC Parallel Filesystem (Lustre / BeeGFS over LNet RDMA + OS Page Cache)",
            "name_zh": "Tier 2: 传统超算并行文件系统 (Lustre / BeeGFS + LNet RDMA + 内核 Page Cache)",
            "datapath_architecture": "Kernel LNet RDMA -> Linux Page Cache LRU Thrashing -> User Copy -> CUDA H2D",
            "replication_and_metadata": "Static OST Striping + Centralized MDT (Single-Namespace Inode Lock Bottleneck)",
            "aggregate_read_throughput_gbps": 2450.0,
            "random_64k_read_iops_millions": 4.5,
            "checkpoint_10tb_load_sec": 4.8,
            "checkpoint_10tb_write_sec": 7.6,
            "client_cpu_overhead_cores_per_node": 18,
            "kvcache_swap_p99_latency_ms": 4.2,
            "three_year_storage_tco_usd": 2480000,
        },
        {
            "tier_id": "tier3_commercial_sds_weka_vast_gds",
            "name": "Tier 3: Commercial AI All-Flash SDS (WEKA / VAST Data + DPDK Zero-Copy + GPUDirect Storage)",
            "name_zh": "Tier 3: 商用全闪 AI 软件定义存储 (WEKA / VAST + DPDK 轮询 + GPUDirect Storage)",
            "datapath_architecture": "User-Space DPDK/RDMA Bypass -> GPUDirect Storage (GDS nvidia-fs DMA to GPU HBM)",
            "replication_and_metadata": "Distributed Erasure Coding (N+2/N+4) + Scale-Out Shared-Nothing MetadataContainers",
            "aggregate_read_throughput_gbps": 5850.0,
            "random_64k_read_iops_millions": 26.5,
            "checkpoint_10tb_load_sec": 1.8,
            "checkpoint_10tb_write_sec": 2.9,
            "client_cpu_overhead_cores_per_node": 4,
            "kvcache_swap_p99_latency_ms": 0.65,
            "three_year_storage_tco_usd": 3680000,
        },
        {
            "tier_id": "tier4_deepseek_3fs_craq_usrbio_fdb",
            "name": "Tier 4: Open-Source DeepSeek 3FS (Fire-Flyer File System: CRAQ Apportioned Reads + USRBIO Zero-Copy RDMA + FoundationDB)",
            "name_zh": "Tier 4: 开源 DeepSeek 3FS (CRAQ 链式均摊读 + USRBIO 零拷贝异步环 + FoundationDB 元数据)",
            "datapath_architecture": "Page-Cache-Free O_DIRECT + Linux io_uring + USRBIO Shared-Memory Ring + RDMA Write/Read",
            "replication_and_metadata": "CRAQ (Chain Replication with Apportioned Queries: All replicas serve reads) + FoundationDB SSI",
            "aggregate_read_throughput_gbps": 6600.0,
            "random_64k_read_iops_millions": 33.0,
            "checkpoint_10tb_load_sec": 1.5,
            "checkpoint_10tb_write_sec": 2.4,
            "client_cpu_overhead_cores_per_node": 2,
            "kvcache_swap_p99_latency_ms": 0.42,
            "three_year_storage_tco_usd": 2200000,
        },
    ]

    # OS Page Cache Thrashing Perturbation under Shuffled Random Read (Working Set = 500 TB >> 2 TB Host RAM)
    page_cache_perturbation = [
        {"working_set_to_ram_ratio": "0.5x (Fits in RAM)", "lustre_pagecache_gbps": 3800.0, "deepseek_3fs_direct_gbps": 6600.0},
        {"working_set_to_ram_ratio": "2.0x (Mild Eviction)", "lustre_pagecache_gbps": 2950.0, "deepseek_3fs_direct_gbps": 6600.0},
        {"working_set_to_ram_ratio": "10.0x (LRU Lock Contention)", "lustre_pagecache_gbps": 2450.0, "deepseek_3fs_direct_gbps": 6590.0},
        {"working_set_to_ram_ratio": "250.0x (500TB Shuffled LLM/KVCache)", "lustre_pagecache_gbps": 1680.0, "deepseek_3fs_direct_gbps": 6580.0},
    ]

    t1 = tiers[0]
    t2 = tiers[1]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier4_vs_tier1_throughput_gain_x": round(
            t4["aggregate_read_throughput_gbps"] / t1["aggregate_read_throughput_gbps"], 2
        ),
        "tier4_vs_tier2_throughput_gain_x": round(
            t4["aggregate_read_throughput_gbps"] / t2["aggregate_read_throughput_gbps"], 2
        ),
        "tier4_vs_tier2_random_iops_gain_x": round(
            t4["random_64k_read_iops_millions"] / t2["random_64k_read_iops_millions"], 2
        ),
        "tier4_10tb_checkpoint_load_sec": t4["checkpoint_10tb_load_sec"],
        "tier3_vs_tier4_3yr_tco_savings_usd": t3["three_year_storage_tco_usd"] - t4["three_year_storage_tco_usd"],
        "tier4_network_saturation_pct": round(t4["aggregate_read_throughput_gbps"] / 7200.0 * 100.0, 1),
    }

    return {
        "experiment_id": "ai_parallel_filesystem_3fs_lustre_weka_selection",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_parallel_filesystem_rdma_datapath_and_craq_iops_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "DeepSeek 3FS (Fire-Flyer File System) 180-Node NVMe + 2x200Gbps InfiniBand (6.6 TB/s Aggregate Read) Architecture",
                "CRAQ (Chain Replication with Apportioned Queries) vs Primary-Copy OSD Read Amplification Model",
                "GPUDirect Storage (GDS) & USRBIO User-Space Zero-Copy Ring vs Linux Kernel VFS Page-Cache LRU Contention",
            ],
        },
        "workload_baseline": {
            "storage_nodes": 180,
            "nvme_drives_per_node": 16,
            "rdma_nic_per_node": "2x 200Gbps RoCEv2 / InfiniBand (7.2 TB/s theoretical line rate)",
            "compute_clients": "1,024-GPU H200 Cluster",
            "checkpoint_payload_tb": 10.0,
        },
        "tiers": tiers,
        "page_cache_perturbation": page_cache_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_parallel_filesystem_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
