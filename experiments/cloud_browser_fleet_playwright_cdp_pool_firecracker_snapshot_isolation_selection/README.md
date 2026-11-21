# Enterprise Cloud Browser Fleet & CDP Pool Selection Experiment (`Firecracker CoW Snapshot MicroVMs` vs. `Shared BrowserContexts` & `Monolithic Containers`)

Reproducible analytical model comparing 4 headless browser infrastructure tiers across `1,000` concurrent AI Agent browser sessions:
- **Tier 1 (`Monolithic Cold Chromium per Container` with default `64MB /dev/shm`)**: `3,500 ms` cold startup, `1,800 GB` RAM, `14.5%` session crash/OOM rate (`$42,000/month`).
- **Tier 2 (`Shared Chromium Process with 25 Playwright BrowserContexts`)**: `85 ms` startup, `340 GB` RAM, **but `4.8%` blast-radius crash rate and zero cross-tenant kernel isolation**.
- **Tier 3 (`Tuned K8s Pod Pool` with `2GB /dev/shm` + `dumb-init` + Warm Standby)**: `220 ms` startup, `950 GB` RAM, `0.45%` crash rate (`$24,500/month`).
- **Tier 4 (`Firecracker / Cloud Hypervisor Copy-on-Write Memory-Snapshot MicroVM Pool + CDP Gateway + Idle Pause`, Recommended)**: **`140 ms` warm snapshot restore (`25x` faster than cold boot)**, **`480 GB` RAM (`-73.3%` via CoW page sharing)**, **`0.05%` crash rate (`99.95%` reliability)**, and full hardware VM isolation at **`$14,800/month` (`-64.8%` cost vs. Tier 1)**.

## Run & Verify

```bash
python experiments/cloud_browser_fleet_playwright_cdp_pool_firecracker_snapshot_isolation_selection/run.py
python experiments/cloud_browser_fleet_playwright_cdp_pool_firecracker_snapshot_isolation_selection/run.py --check
```
