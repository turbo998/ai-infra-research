# Enterprise Smart Building & Data Center BMS HVAC Copilot: Brick Schema, Physics-Informed MPC & BACnet Clamp Evaluator

Deterministic evaluator comparing four smart building and data-center HVAC optimization agent architectures across 45 campus/data-center buildings, 180 chillers and cooling towers, and 350,000 `BACnet / Modbus` telemetry points:
- **Tier 1**: Legacy BMS Fixed Schedules + Reactive Local PID Loops (`4.5%` HVAC energy savings, `1.5%` thermal excursion rate, `160.0 h/building` onboarding, `$420,000/mo`)
- **Tier 2**: Prompt-Only LLM Directly Writing `BACnet` Chiller & AHU Setpoints (`12.0%` savings, `13.8%` `ASHRAE` thermal / chiller-surge violation rate, `18.0 h/building`, `$510,000/mo`)
- **Tier 3**: `Brick Schema` Tagging + Read-Only Fault Detection & Diagnostics (`FDD`) Copilot (`11.8%` savings, `0.8%` excursion rate, `12.0 h/building`, `$215,000/mo`)
- **Tier 4**: LLM `Brick Schema` Ontology Auto-Mapper + Physics-Informed Model Predictive Control (`MPC`) Chiller/TOU Solver + Local PLC `BACnet` Safety Clamp Agent (`26.4%` energy savings, `0.0%` thermal / surge violation rate, `4.5 h/building`, `$64,000/mo`)

## Usage
```bash
python experiments/enterprise_smart_building_datacenter_hvac_mpc_energy_agent_selection/run.py
python experiments/enterprise_smart_building_datacenter_hvac_mpc_energy_agent_selection/run.py --check
```
