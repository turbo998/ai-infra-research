# Enterprise Zero-Downtime Model Weight Hot-Swapping, vLLM Sleep Mode & KV-Aware Drain Selection (`#494`)

Reproducible analytical model comparing 4 production model rollout & weight-switching architectures across a 32-GPU enterprise serving cluster (`140 GB` model checkpoint):
- **Tier 1 (`Naive K8s RollingUpdate`)**: `185.0s` cold reload, `12.4%` dropped active SSE streams, and `+50%` surge GPUs.
- **Tier 2 (`Full Blue-Green Double Fleet`)**: `0.5s` switch and `0%` drops, but wastes `+100%` standby GPUs (`$92,000/month`).
- **Tier 3 (`KV-Aware Graceful Drain + GDS Disk Streamer`)**: `28.5s` switch and `0%` drops with `+12.5%` surge GPUs.
- **Tier 4 (`Host-DRAM Pinned Pre-Staging + vLLM Sleep Mode Level-1 + KV-Aware Drain`)**: Pre-loads new weights into idle Host DDR5 Pinned RAM in the background and swaps weights into HBM over PCIe Gen5 in **`1.8 seconds` (`102.8x` faster)** with **`0.0%` dropped requests** and **`0` extra standby GPUs**.

## Run
```bash
python experiments/enterprise_zero_downtime_model_hotswap_sleep_mode_kv_drain_selection/run.py
```
