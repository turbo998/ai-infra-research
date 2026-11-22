# Enterprise GPU Power Capping (`700W` vs. `550W/500W` Cubic $V\text{-}f$ Sweet-Spot), Role-Aware PD DVFS & Power-Whip Smoothing (`#492`)

Reproducible analytical model demonstrating how CMOS cubic dynamic power scaling ($P \propto V^2 f \propto f^3$) and Role-Aware PD Power Capping (`650W` Prefill / `500W` Decode) allow deploying **25% more GPUs (`1,280` vs. `1,024`)** under a fixed `1.5 MW` datacenter power envelope while retaining `98.8%` Decode throughput.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection/run.py --check`
