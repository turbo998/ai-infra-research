# Enterprise Multi-Team Shared Model-as-a-Service (`Internal MaaS Platform`) & GPU Sharing Selection (`R481`)

Deterministic evaluation suite comparing four enterprise internal Model-as-a-Service (`MaaS`) and multi-tenant GPU sharing architectures across `25` internal engineering & business teams (`85%` shared foundation model calls, `15%` custom small models & fine-tuning jobs):

1. **Tier 1**: Siloed Departmental GPU Subscriptions & Dedicated Nodes (`22.0%` GPU utilization, `14.0 days` onboarding lead time, `80x H100`, `$310,000/mo` TCO)
2. **Tier 2**: Un-Isolated GPU Time-Slicing / CUDA MPS Across Multi-Team Containers (`68.0%` utilization, `31.5%` noisy-neighbor OOM/latency interference, `40x H100`, `$178,000/mo` TCO)
3. **Tier 3**: Static Hardware `NVIDIA MIG` Slices Only (No Shared API-Level Model Pool) (`54.0%` utilization, `0.0%` interference, `48x H100`, `$164,000/mo` TCO)
4. **Tier 4**: 3-Layer Enterprise Internal `MaaS` (`API-Level Multi-Tenant Pool` + `NVIDIA MIG` for Custom Small Models + `Kueue` Cohort Fair-Sharing) (`86.5%` GPU utilization, `0.0%` noisy-neighbor interference, `0.02 days = 30 min` onboarding, `24x H100`, `$82,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/run.py
python experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/run.py --check
python -m unittest tests/test_enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection.py
```
