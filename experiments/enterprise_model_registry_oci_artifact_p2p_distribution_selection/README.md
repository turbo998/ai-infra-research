# Enterprise Model Asset Registry, Supply-Chain Security & Multi-Node P2P Weight Distribution Selection (`R484`)

Deterministic evaluation suite comparing four enterprise model weight management, supply-chain security, and Kubernetes cluster distribution architectures across a `32-node` GPU scale-out (`140 GB` model checkpoint, `180` monthly rollout/scale events):

1. **Tier 1**: Live `huggingface_hub` Internet Pulls on Pod Startup (`1,420s` 32-pod ready time, `100.0%` `pickle` RCE / upstream tamper exposure, `4,480 GB` egress/rollout, `28.4%` rollout failure rate, `$146,000/mo` TCO)
2. **Tier 2**: Baking `140 GB` Model Weights Inside Monolithic Docker Container Images (`980s` ready time, `4,960 GB` egress/rollout, `$124,000/mo` TCO)
3. **Tier 3**: Private S3 / Blob Object Storage `initContainer` Sync per Pod (`295s` ready time, `4,480 GB` egress/rollout, `$68,000/mo` TCO)
4. **Tier 4**: `OCI Model Spec (ORAS)` Registry (`Harbor / ACR`) + `Safetensors + Sigstore Cosign` Gate + `Dragonfly / Spegel` P2P Distribution + Shared NVMe CSI Cache (`28.0s` 32-pod ready time, `0.0%` supply-chain RCE risk, `140 GB` registry egress/rollout, `0.0%` failure rate, `$21,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_model_registry_oci_artifact_p2p_distribution_selection/run.py
python experiments/enterprise_model_registry_oci_artifact_p2p_distribution_selection/run.py --check
python -m unittest tests/test_enterprise_model_registry_oci_artifact_p2p_distribution_selection.py
```
