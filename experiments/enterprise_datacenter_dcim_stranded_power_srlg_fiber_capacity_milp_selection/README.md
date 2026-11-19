# Enterprise Data Center `DCIM` 5D Stranded-Power `MILP`, 3-Phase `PDU` Balancing & `SRLG` Fiber Agent (`R465`)

Deterministic evaluation suite comparing four data center and colocation infrastructure management (`DCIM`), rack placement, 3-phase PDU load balancing, and fiber cross-connect routing architectures across a `40 MW` data center (`1,600` racks, `48,000` fiber/DAC cross-connects):

1. **Tier 1**: Static Nameplate Power Visio/Excel Sheets + Greedy Rack Placement + Shortest-Path Patching (`31.5%` stranded power, `18.4%` SRLG shared-conduit violation, `14.6%` 3-phase PDU trip risk, `$1,150,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Rack Elevation & Patch-Panel Planner (`28.0%` stranded power, `24.5%` SRLG violation, `19.2%` PDU trip risk, `$1,280,000/mo` loss)
3. **Tier 3**: Standard DCIM Inventory DB + Peak Telemetry Polling (`14.2%` stranded power, `6.8%` SRLG violation, `5.4%` PDU trip risk, `$480,000/mo` loss)
4. **Tier 4**: 5D Capacity-Tetris `MILP` + 3-Phase `L1/L2/L3 PDU` Balancer + `Suurballe SRLG` Disjoint Fiber & Rack `VLM` Agent (`2.1%` stranded power, `0.0%` SRLG violation, `0.0%` PDU trip risk, `0.0%` cabling rework, `$92,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection/run.py
python experiments/enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection/run.py --check
python -m unittest tests/test_enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection.py
```
