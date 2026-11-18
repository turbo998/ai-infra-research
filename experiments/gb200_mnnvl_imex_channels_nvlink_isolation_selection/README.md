# GB200 / B200 Multi-Node NVLink (`MNNVL`), `IMEX` Cross-Node Memory Channels & Multi-Tenant Fabric Isolation Selection (`gb200_mnnvl_imex_channels_nvlink_isolation_selection`)

Deterministic Python 3.12 analytical evaluator for **SA Guide**: comparing `Unmounted IMEX Silent NIC Fallback` (Tier 1), `Shared Global channel0 + Single Rack Partition` (Tier 2), `K8s DRA Per-Job Ephemeral IMEX channelN Isolation` (Tier 3), and `K8s DRA Ephemeral IMEX + NMX-C Hardware NVSwitch Partitioning` (Tier 4).

## Reproduction

```bash
python -m experiments.gb200_mnnvl_imex_channels_nvlink_isolation_selection.run
python -m unittest tests/test_gb200_mnnvl_imex_channels_nvlink_isolation_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_mnnvl_specs`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
