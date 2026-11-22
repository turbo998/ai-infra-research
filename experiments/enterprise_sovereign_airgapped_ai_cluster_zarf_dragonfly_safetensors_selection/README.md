# Enterprise Sovereign & Air-Gapped AI Cluster Delivery (`Zarf / Hauler` + `Dragonfly P2P OCI` + `Cosign Safetensors` + `Local HSM`) Selection (`#503`)

Reproducible analytical model comparing 4 offline air-gapped AI cluster delivery and model supply-chain architectures across a 64-node (`512-GPU`) sovereign datacenter:
- **Tier 1 (`Ad-Hoc USB Tarballs: docker save + pip wheel + .pt Pickle`)**: `18.0%` first-time deployment success (driver/glibc mismatches), `21.5 days` delivery lead time, `870 min (14.5h)` 64-node rollout, and critical `.pt` Pickle RCE risk.
- **Tier 2 (`Local Harbor Mirror + Manual Helm Scripts`)**: `62.5%` first-time success, `6.5 days` lead time, `240 min` rollout.
- **Tier 3 (`Declarative Zarf/Hauler Bundle + Local Harbor`)**: `96.0%` first-time success, `0.8 days` lead time, `68 min` rollout.
- **Tier 4 (`Zarf/Hauler Bundle + Dragonfly P2P OCI + Cosign-Signed Safetensors + Local HSM`)**: Raises first-time deployment success to **`99.8%`**, cuts delivery lead time **`53.8x` (`21.5 days -> 0.4 days`)**, and accelerates 64-node image/weight rollout **`271.9x` (`870 min -> 3.2 min`)** with zero Pickle RCE vulnerability.

## Run
```bash
python experiments/enterprise_sovereign_airgapped_ai_cluster_zarf_dragonfly_safetensors_selection/run.py
```
