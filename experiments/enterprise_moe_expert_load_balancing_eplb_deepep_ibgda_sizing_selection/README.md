# Enterprise Sparse MoE Expert Load Balancing (`EPLB`) & Dual-Mode `DeepEP` (`Normal` Prefill vs. `IBGDA` Low-Latency Decode) Sizing Model (`#454`)

Reproducible analytical model comparing static MoE expert placement with blocking NCCL `All-to-All`, static placement with `DeepEP`, hierarchical `EPLB` (`256 + 32` redundant hot experts), and full `EPLB + Dual-Mode DeepEP + Two-Microbatch Overlap`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_moe_expert_load_balancing_eplb_deepep_ibgda_sizing_selection/run.py --check`
