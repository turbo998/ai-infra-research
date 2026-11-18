# Enterprise Energy Grid Virtual Power Plant (VPP) Dispatch, Substation Inspection VLM & Five-Preventions Switching Agent Evaluator

Deterministic evaluator comparing four electric utility, VPP dispatch, and substation inspection agent architectures across 2,400 MW aggregated VPP capacity, 320 110kV/220kV substations, 450,000 monthly drone thermal/RGB images, and 3,800 monthly switching orders:
- **Tier 1**: Manual Spreadsheet VPP Bidding + Manual Drone Photo Review & Switching Orders (`76.5%` forecast accuracy, `0.4%` interlock violation rate, `72.0%` hotspot recall, `$295,000/mo`)
- **Tier 2**: Prompt-Only Text LLM Predicting MW Curves & Drafting Unverified Switching Orders (`68.0%` forecast accuracy, `14.5%` Five-Preventions safety violation rate, `64.5%` hotspot recall, `$420,000/mo`)
- **Tier 3**: NWP + Time-Series Forecaster + Text-Only Grid Code RAG (`87.2%` forecast accuracy, `4.8%` interlock violation rate, `81.0%` hotspot recall, `$128,000/mo`)
- **Tier 4**: NWP/Timeseries + Stochastic `MILP` VPP Solver + Radiometric `ΔT` VLM + `IEC 61850 / Five-Preventions` Topology Interlock Agent (`94.4%` forecast accuracy, `0.0%` interlock violation rate, `97.2%` hotspot recall, `$41,000/mo`)

## Usage
```bash
python experiments/enterprise_energy_grid_vpp_dispatch_substation_inspection_agent_selection/run.py
python experiments/enterprise_energy_grid_vpp_dispatch_substation_inspection_agent_selection/run.py --check
```
