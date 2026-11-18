# Enterprise Autonomous Warehouse Robotics (`AMR / ASRS`), Multi-Agent Path Finding (`MAPF`), 3D Cubing & Waveless Agent (`R450`)

Deterministic evaluation suite comparing four Goods-to-Person (`G2P`) autonomous warehouse robotics, Multi-Agent Path Finding (`MAPF`), waveless order release, and 3D carton cubing architectures across an `800-AMR` fulfillment center (`32` pick stations, `160,000` daily shipped parcels):

1. **Tier 1**: Discrete 4-Hour Wave Picking + Independent Greedy `A*` AMR Routing + Scalar Volume-Sum Box Selection (`18.5%` aisle deadlock/congestion, `24.0%` station starvation, `34.0%` carton void air, `185 UPH`, `$790,000/mo` waste)
2. **Tier 2**: Prompt-Only General LLM Warehouse Task & Box-Size Dispatcher (`22.0%` deadlock/congestion, `26.5%` station starvation, `37.5%` void air, `165 UPH`, `$910,000/mo` waste)
3. **Tier 3**: Decentralized `MARL` Fleet Router Without Space-Time Reservation Tables (`5.8%` congestion, `9.4%` starvation, `19.2%` void air, `310 UPH`, `$340,000/mo` waste)
4. **Tier 4**: Windowed `MAPF` (`RHCR CBS / PIBT` Space-Time Reservation) + Continuous Waveless Order-Release `MILP` + 3D Extreme-Point Cubing + Velocity Pod Slotting (`0.0%` deadlock, `1.1%` station starvation, `6.4%` carton void air, `495 UPH`, `$68,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection/run.py
python experiments/enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection/run.py --check
python -m unittest tests/test_enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection.py
```
