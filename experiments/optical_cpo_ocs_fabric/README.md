# Optical Interconnect CPO / LPO / OCS 100k-GPU Fabric Model

This offline stdlib analytical model (`evidence_class: own-offline`) evaluates the power, link-flap MTBF, and synchronous training goodput of five 800G optical interconnect architectures across cluster scales from `16,384` to `100,000` GPUs:

1. `pluggable_dsp_3tier_800g` — 3-Tier Clos with 800G DSP OSFP pluggables (`6` optical endpoints/GPU, `16.5 pJ/bit`, `450k` hr MTBF).
2. `pluggable_lpo_3tier_800g` — 3-Tier Clos with 800G LPO linear drive OSFP (`6` endpoints/GPU, `9.0 pJ/bit`, `620k` hr MTBF).
3. `cpo_elsfp_3tier_800g` — Quantum-X800 / Spectrum-X800 CPO with external liquid-cooled laser source (`ELSFP`, `4.5 pJ/bit`, `2.0M` hr MTBF, `2` ports/OE, `8` ports/ELSFP).
4. `ocs_mems_sp_ineliminated_800g` — MEMS OCS spine-eliminated direct-ToR topology (`2` active optical endpoints/GPU, `16.5 pJ/bit`, `15 ms` circuit reconfiguration).
5. `ocs_cpo_hybrid_800g` — Piezo/MEMS OCS + ToR CPO hybrid (`2` endpoints/GPU, `4.5 pJ/bit`, `1.2 ms` reconfiguration).

## Usage

```bash
python experiments/optical_cpo_ocs_fabric/run.py
python experiments/optical_cpo_ocs_fabric/run.py --check
```
