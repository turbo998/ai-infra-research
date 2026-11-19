# Enterprise Mining (`AHS` Haulage Blending `MILP`, `SAG` Mill `MPC` & `GISTM InSAR` Tailings Agent) (`R454`)

Deterministic evaluation suite comparing four open-pit/underground mining haulage, mineral-processing comminution/flotation, and tailings dam monitoring architectures across a `250,000 tonnes/day` mining complex (`140` 300-tonne haul trucks, `4` SAG/Ball grinding lines, `320` tailings piezometer/InSAR nodes):

1. **Tier 1**: Fixed Truck-Shovel Locking + Single-Loop SAG Mill PID + Monthly Manual Tailings Walk (`16.4` stalls/mo, `23.6%` haulage waste, `14.8%` grinding/recovery loss, `$1,420,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Mine Dispatch & Mill Log Chatbot (`14.8` stalls/mo, `21.2%` haulage waste, `16.2%` grinding/recovery loss, `$1,560,000/mo` loss)
3. **Tier 3**: Siloed GPS Fleet Dispatch + Uncoupled Mill Regression ML (`5.6` stalls/mo, `9.8%` haulage waste, `6.4%` grinding/recovery loss, `$520,000/mo` loss)
4. **Tier 4**: Geometallurgical Blending `AHS MILP` + 3D Fragmentation Feed-Forward `SAG MPC` + `GISTM InSAR/VWP` Tailings Agent (`0.3` stalls/mo, `1.8%` haulage waste, `1.2%` grinding/recovery loss, `0.0%` tailings alarm escape, `$135,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection/run.py
python experiments/enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection/run.py --check
python -m unittest tests/test_enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection.py
```
