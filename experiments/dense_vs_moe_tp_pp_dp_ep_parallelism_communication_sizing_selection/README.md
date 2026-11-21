# Experiment: Dense (70B/405B) vs. Large MoE (671B) Inference Parallelism Selection & Communication Sizing (`#397`)

Reproducible analytical model comparing `TP`, `PP`, `Attention DP`, `Expert EP (DeepEP)`, `Speculative Decoding (MTP / EAGLE-3)`, `Prefix Caching`, and `FP8` quantization across Dense and MoE architectures.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/dense_vs_moe_tp_pp_dp_ep_parallelism_communication_sizing_selection/run.py`
- **Verify**: `python experiments/dense_vs_moe_tp_pp_dp_ep_parallelism_communication_sizing_selection/run.py --check`
