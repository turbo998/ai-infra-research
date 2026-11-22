# Enterprise Hybrid / Multi-Cloud GPU Bursting (`SkyPilot / Kueue MultiKueue`, Egress Tax Avoidance & `P80` Base Sizing) (`#487`)

Reproducible analytical model comparing 100% On-Premises Peak Over-Provisioning (`128 GPUs`, `$2.76M/yr`), Naive Cross-WAN PD/Storage Bursting (`$48.5k/mo` egress bill, `+940ms` lag), Single-Cloud Cold Bursting, and `On-Prem P80 Base (48 GPUs) + SkyPilot/MultiKueue Whole-Request Bursting` (`$1.185M/yr`, `-57.1%` TCO).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_hybrid_multicloud_gpu_bursting_skypilot_multikueue_egress_tco_selection/run.py --check`
