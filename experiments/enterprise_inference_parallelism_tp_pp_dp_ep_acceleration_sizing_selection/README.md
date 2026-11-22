# Enterprise Inference Parallelism (`TP / PP / DP / EP`) & Acceleration Selection Model (`#445`)

Reproducible analytical model comparing Cross-Node Pure TP, Intra-Node NVLink TP × Inter-Node PP, Attention DP + Expert EP (`DeepEP`), and Full-Stack Acceleration (`Block-FP8 + Radix Prefix Cache + Native MTP/EAGLE-3 with Batch<=48 Auto-Gate`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_inference_parallelism_tp_pp_dp_ep_acceleration_sizing_selection/run.py --check`
