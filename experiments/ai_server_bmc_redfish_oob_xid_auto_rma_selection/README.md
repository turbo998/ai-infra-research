# AI Server Out-of-Band Telemetry (`BMC / Redfish` vs. `IPMI`), GPU `XID` Fault Classification & Auto-RMA Remediation Selection (`ai_server_bmc_redfish_oob_xid_auto_rma_selection`)

Deterministic Python 3.12 analytical evaluator for **SA Guide**: comparing `Reactive dmesg + Legacy IPMI Polling` (Tier 1), `In-Band DCGM Exporter + Blanket Node Cordon` (Tier 2), `XID 3-Tier Severity Classifier + HBM Row-Remapper Gate` (Tier 3), and `Out-of-Band Redfish SSE Telemetry (SMBPBI) + 4-Stage Auto-Remediation` (Tier 4).

## Reproduction

```bash
python -m experiments.ai_server_bmc_redfish_oob_xid_auto_rma_selection.run
python -m unittest tests/test_ai_server_bmc_redfish_oob_xid_auto_rma_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_hardware_ras_specs`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
