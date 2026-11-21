# Coding Agent Workspace & Dependency Caching Selection Experiment (`Git Worktree + uv/pnpm Hardlink/Reflink Store` vs. `Cold Clone & pip/npm`)

Reproducible analytical model comparing 4 Coding Agent workspace provisioning architectures across `10,000` daily agent task runs (`1.6 GB` git repo + `3.2 GB` Python/Node dependencies):
- **Tier 1 (`Cold git clone + Uncached pip/npm install`)**: `165.0 sec` startup, `4.80 GB` disk per workspace, `6.80%` network rate-limit failure rate.
- **Tier 2 (`Per-Repo Pre-Baked 18GB Fat Docker Image`)**: `4.5 sec` startup, `2,400 GB/day` registry build/push churn across 500 repos.
- **Tier 3 (`Git --reference-if-able + Local HTTP Wheel/npm Proxy`)**: `28.0 sec` startup (bottlenecked by decompressing `85,000` small files), `3.30 GB` disk per workspace.
- **Tier 4 (`Bare-Repo Git Worktree / OverlayFS + uv/pnpm Content-Addressable Hardlink/Reflink Store`, Recommended)**: **`0.80 sec` total workspace readiness (`206x` faster than Tier 1)**, **`0.04 GB` (`40 MB`) disk footprint per workspace (`-99.2%`)**, and `99.99%` offline cache reliability.

## Run & Verify

```bash
python experiments/agentic_workspace_dependency_caching_uv_hardlink_nix_git_worktree_selection/run.py
python experiments/agentic_workspace_dependency_caching_uv_hardlink_nix_git_worktree_selection/run.py --check
```
