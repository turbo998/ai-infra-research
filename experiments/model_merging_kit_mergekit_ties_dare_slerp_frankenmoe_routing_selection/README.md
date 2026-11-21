# Enterprise Model Merging & Multi-Specialist Consolidation Selection Experiment (`MergeKit DARE-TIES / DELLA / SLERP` vs. `4 Isolated 70B Pools` vs. `Joint Retraining`)

Reproducible accuracy retention, synthesis cost, and serving GPU footprint model comparing 4 model consolidation strategies across `4` domain-specialized `70B` checkpoints.

## Run
```bash
python experiments/model_merging_kit_mergekit_ties_dare_slerp_frankenmoe_routing_selection/run.py
python experiments/model_merging_kit_mergekit_ties_dare_slerp_frankenmoe_routing_selection/run.py --check
```
