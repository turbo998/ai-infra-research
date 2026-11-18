# Enterprise Water Utility Leak Localization & Wastewater Treatment (`WWTP`) Aeration/Dosing `MPC` Agent (`R442`)

Deterministic evaluation suite comparing four municipal/industrial water utility pipe-leak localization, wastewater biological nitrification aeration (`ASM1`), and chemical coagulant dosing architectures across a `4,500 km` distribution network (`650,000 m³/day`, `95` District Metered Areas `DMAs`, `5,400 kW` aeration blower capacity):

1. **Tier 1**: Manual Acoustic Walking Surveys + Fixed `DO = 2.5 mg/L` Blower PID (`26.5%` Non-Revenue Water loss, `31.0%` energy/chemical overuse, `4.8%` effluent permit breach, `1,800m` leak radius, `$610,000/mo` TCO)
2. **Tier 2**: Prompt-Only General LLM SCADA Alarm & Operator Logbook Advisor (`25.0%` NRW loss, `27.5%` energy/chemical overuse, `9.2%` permit breach, `$690,000/mo` TCO)
3. **Tier 3**: Unconstrained Black-Box Time-Series Neural Net (`16.4%` NRW loss, `14.2%` overuse, `2.4%` permit breach, `420m` leak radius, `$295,000/mo` TCO)
4. **Tier 4**: `EPANET` Hazen-Williams Hydraulic Graph Inverse Solver + `ASM1` Ammonia-Cascade `MPC` (`ABAC`) + PLC Alkalinity/Airflow Safety Clamp (`7.8%` NRW loss, `2.1%` energy/chemical overuse, `0.0%` permit breach, `35m` leak radius, `$64,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_water_utility_leak_detection_wastewater_aeration_mpc_agent_selection/run.py
python experiments/enterprise_water_utility_leak_detection_wastewater_aeration_mpc_agent_selection/run.py --check
python -m unittest tests/test_enterprise_water_utility_leak_detection_wastewater_aeration_mpc_agent_selection.py
```
