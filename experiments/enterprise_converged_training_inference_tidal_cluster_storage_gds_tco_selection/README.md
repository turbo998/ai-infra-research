# Milestone #460: Enterprise Converged Training-Inference Tidal Cluster (`Daytime PD Serving + Nighttime GRPO/SFT Backfill`) & High-Speed Storage (`3FS / Lustre / WEKA + GDS + 2.1s Async Checkpoint`) Model (`#460`)

Reproducible analytical model comparing siloed training/inference clusters with synchronous checkpoints, shared K8s clusters with pod restarts, Kueue preemption with dedicated Lustre/WEKA, and a Converged Tidal AI Factory (`2.8s Sleep Switch + Hyperconverged 3FS GDS + 2.1s Two-Stage Async Checkpoint`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection/run.py --check`
