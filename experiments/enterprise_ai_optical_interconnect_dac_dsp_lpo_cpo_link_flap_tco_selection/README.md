# Enterprise AI Cluster Optical & Copper Interconnect (`DAC/ACC` vs. `Pluggable DSP` vs. `LPO` vs. `CPO`) & Link-Flap TCO Model (`#462`)

Reproducible analytical model comparing 100% pluggable DSP optics, DAC + unmonitored DSP optics, LPO linear pluggable optics (`-50%` power, `5 ns` PHY latency), and Distance-Tiered Hybrid (`DAC/ACC <5m + LPO <50m + DSP >50m`) with `Pre-FEC BER` predictive link-flap isolation.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_ai_optical_interconnect_dac_dsp_lpo_cpo_link_flap_tco_selection/run.py --check`
