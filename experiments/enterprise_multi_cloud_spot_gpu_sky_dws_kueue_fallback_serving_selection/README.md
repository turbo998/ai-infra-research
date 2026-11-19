# Enterprise Multi-Cloud Spot GPU, `Kueue ProvisioningRequest` (`DWS`) & `SkyPilot` Drain-Aware Serving (`R457`)

Deterministic evaluation suite comparing four cloud GPU capacity-procurement and Spot preemption-resilience architectures across a `96-GPU` enterprise fleet (`30%` On-Demand base + `70%` elastic Spot/DWS across `4` regions/SKU pools):

1. **Tier 1**: Single-AZ On-Demand GPU Cluster + Reactive Node-by-Node Autoscaler (`28.4%` stockout failure, `22.0%` partial-gang idle waste, `420 s` recovery, `$215,000/mo` GPU TCO)
2. **Tier 2**: Raw Single-Region Spot GPUs Without Termination Handler (`34.0%` stockout failure, `100.0%` preemption dropped requests, `510 s` recovery, `$142,000/mo` GPU TCO)
3. **Tier 3**: Multi-AZ Karpenter / NAP Spot Pool (`9.2%` stockout failure, `4.8%` preemption dropped requests, `8.4%` partial-gang waste, `$118,000/mo` GPU TCO)
4. **Tier 4**: `Kueue ProvisioningRequest` (`DWS / Capacity Blocks`) + `SkyPilot` Multi-Cloud Spot Broker + `IMDS 30s` Drain-Aware Serving (`0.1%` stockout failure, `0.0%` dropped requests, `0.0%` partial-gang waste, `14.0 s` recovery, `$79,000/mo` GPU TCO)

## Usage

```bash
python experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/run.py
python experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/run.py --check
python -m unittest tests/test_enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection.py
```
