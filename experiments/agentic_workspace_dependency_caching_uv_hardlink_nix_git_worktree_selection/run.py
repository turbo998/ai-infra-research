from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "agentic_workspace_dependency_caching_uv_hardlink_nix_git_worktree_selection",
        "title": "Coding Agent Workspace & Dependency Caching Selection: Git Worktree + uv/pnpm Hardlink/Reflink Store vs. Cold Clone & pip/npm",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "daily_agent_worktrees": 10000,
            "repo_git_size_gb": 1.6,
            "python_and_node_deps_gb": 3.2,
        },
        "tiers": [
            {
                "tier": "Tier 1: Cold git clone + Uncached pip install & npm install per Sandbox",
                "workspace_ready_sec": 165.0,
                "disk_footprint_gb_per_workspace": 4.80,
                "registry_churn_gb_per_day": 48000.0,
                "network_rate_limit_failure_pct": 6.80,
                "recommended_scenario": "Anti-pattern (wastes 2.75 minutes and 4.8GB disk I/O per agent task)",
            },
            {
                "tier": "Tier 2: Per-Repo Pre-Baked Fat Docker Image (18 GB Image per Repo)",
                "workspace_ready_sec": 4.5,
                "disk_footprint_gb_per_workspace": 1.20,
                "registry_churn_gb_per_day": 2400.0,
                "network_rate_limit_failure_pct": 0.20,
                "recommended_scenario": "Fixed benchmark suites (SWE-bench static repo snapshots) without frequent lockfile edits",
            },
            {
                "tier": "Tier 3: Git --reference-if-able + Local DevPI / Verdaccio HTTP Proxy Cache",
                "workspace_ready_sec": 28.0,
                "disk_footprint_gb_per_workspace": 3.30,
                "registry_churn_gb_per_day": 450.0,
                "network_rate_limit_failure_pct": 0.05,
                "recommended_scenario": "Legacy pip/npm workflows (still bottlenecked by 28s zip decompression of 85k files)",
            },
            {
                "tier": "Tier 4 (Recommended): Bare-Repo Git Worktree / OverlayFS + uv/pnpm Content-Addressed Hardlink/Reflink Store",
                "workspace_ready_sec": 0.80,
                "disk_footprint_gb_per_workspace": 0.04,
                "registry_churn_gb_per_day": 12.0,
                "network_rate_limit_failure_pct": 0.01,
                "recommended_scenario": "Production Coding Agent fleets (0.80s workspace ready — 206x faster, 99.2% less disk)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
