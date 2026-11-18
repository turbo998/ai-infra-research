#!/usr/bin/env python3
"""AI Container Image Distribution (Nydus / Dragonfly P2P) & SquashFS Python Import Storm Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_container_and_import_tiers(
    cluster_nodes: int = 512,
    gpus_per_node: int = 8,
    image_size_gb: float = 25.0,
) -> dict[str, Any]:
    """Evaluate 4 container image distribution and Python environment mount tiers across 512 GPU nodes."""
    total_ranks = cluster_nodes * gpus_per_node  # 4,096 ranks
    # Each Python rank doing 'import torch, transformers, vllm' issues ~18,000 stat/open calls on sys.path
    naive_mds_rpcs_millions = round((total_ranks * 18000) / 1e6, 2)  # 73.73M metadata RPCs

    tiers = [
        {
            "tier_id": "tier1_central_harbor_gzip_plus_shared_nfs_conda",
            "name": "Tier 1: 中心 Harbor 仓库 gzip 串行拉取 + 共享 NFS/Lustre 散装 Conda 环境",
            "image_pull_decompress_s": 395.0,
            "python_import_torch_s": 185.0,
            "total_job_cold_start_s": 580.0,
            "shared_mds_metadata_rpcs_millions": naive_mds_rpcs_millions,
            "actual_network_bytes_per_node_gb": image_size_gb,
            "verdict": "512 节点并发拉 25GB 挤爆中心仓库网卡，且 4,096 进程并发 stat() 打挂存储 MDS 元数据服务 ✕",
        },
        {
            "tier_id": "tier2_dragonfly_p2p_zstd_whole_image",
            "name": "Tier 2: Dragonfly / Kraken P2P 分发 + zstd 多线程解压 (仍全量拉取 + NFS 散装)",
            "image_pull_decompress_s": 38.0,
            "python_import_torch_s": 142.0,
            "total_job_cold_start_s": 180.0,
            "shared_mds_metadata_rpcs_millions": naive_mds_rpcs_millions,
            "actual_network_bytes_per_node_gb": image_size_gb,
            "verdict": "P2P 解决了镜像下载带宽瓶颈，但全量解压仍需 38s，且未解决共享存储 import torch 元数据风暴 ⚠️",
        },
        {
            "tier_id": "tier3_nydus_estargz_lazy_pull_fuse",
            "name": "Tier 3: Nydus RAFS / OCI eStargz 块级按需懒加载 (User-Space FUSE)",
            "image_pull_decompress_s": 4.5,
            "python_import_torch_s": 12.4,
            "total_job_cold_start_s": 16.9,
            "shared_mds_metadata_rpcs_millions": 1.2,
            "actual_network_bytes_per_node_gb": round(image_size_gb * 0.062, 2),
            "verdict": "仅拉取启动所需的 6.2% (1.55GB) 数据块即可秒级拉起容器，但用户态 FUSE 在高并发读 .so 时有上下文切换开销 ✅",
        },
        {
            "tier_id": "tier4_nydus_erofs_p2p_plus_squashfs_loopback_conda",
            "name": "Tier 4: Nydus 内核态 EROFS + P2P 预热 + SquashFS 单文件挂载 Conda 环境",
            "image_pull_decompress_s": 1.6,
            "python_import_torch_s": 2.2,
            "total_job_cold_start_s": 3.8,
            "shared_mds_metadata_rpcs_millions": 0.004,
            "actual_network_bytes_per_node_gb": round(image_size_gb * 0.062, 2),
            "verdict": "EROFS fscache 零 FUSE 开销 + 15 万个 .py/.so 打包为单个 .sqsh 镜像挂载，512 节点 3.8s 极速就绪 (快 152x) 🏆",
        },
    ]

    return {
        "experiment_id": "ai_container_image_nydus_dragonfly_squashfs_selection",
        "round": 187,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "cluster_nodes": cluster_nodes,
            "gpus_per_node": gpus_per_node,
            "total_ranks": total_ranks,
            "image_size_gb": image_size_gb,
            "naive_mds_rpcs_millions": naive_mds_rpcs_millions,
            "cold_start_speedup_x": round(580.0 / 3.8, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_container_and_import_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
