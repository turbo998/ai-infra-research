# Online-Offline GPU Co-Location & Opportunistic Backfill Selection Experiment (`KV-Watermark Priority Preemption` vs. `Siloed Clusters`)

Reproducible analytical model comparing 4 online-offline GPU scheduling architectures (`48` peak online H100 GPUs + `1.2B` daily offline batch tokens):
- **Tier 1 (`Siloed 48 Online + 16 Offline GPUs = 64 GPUs`)**: `42.5%` 24h utilization, `380 ms` online P99 TTFT, `$203,200/month`.
- **Tier 2 (`Un-Isolated Shared FIFO Queue on 48 GPUs`)**: `86.0%` utilization, **but online P99 TTFT spikes to `4,850 ms` (`44.0%` SLA Goodput)**.
- **Tier 3 (`Coarse CronHPA Day/Night Pod Swap`)**: `72.0%` utilization, `410 ms` P99 TTFT, `52` GPUs (`$165,100/month`).
- **Tier 4 (`KV-Watermark-Gated Opportunistic Backfill + vLLM Priority Preemption --scheduling-policy priority`, Recommended)**: **`91.5%` 24h utilization on `48` GPUs (`-$50,800/month` saved)** while keeping online P99 TTFT at **`395 ms` (`99.4%` SLA Goodput)**.

## Run & Verify

```bash
python experiments/gpu_cluster_online_offline_co_location_opportunistic_backfill_peak_shaving_selection/run.py
python experiments/gpu_cluster_online_offline_co_location_opportunistic_backfill_peak_shaving_selection/run.py --check
```
