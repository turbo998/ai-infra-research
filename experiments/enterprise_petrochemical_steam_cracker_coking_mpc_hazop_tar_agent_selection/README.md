# Enterprise Petrochemical Steam-Cracker Coking `MPC`, `IEC 61511 SIS` & Turnaround (`TAR`) Agent (`R466`)

Deterministic evaluation suite comparing four petrochemical olefin steam-cracker (`Naphtha / Ethane Pyrolysis`), radiant-coil coking control, `IEC 61511` process safety, and plant turnaround (`TAR`) scheduling architectures across a `1.5M tonnes/yr` ethylene complex (`10` cracking furnaces, `480` radiant coils, `3,500` turnaround work orders):

1. **Tier 1**: Fixed Coil-Outlet-Temperature (`COT`) DCS PID + Fixed 45-Day Decoking + Manual TAR (`18.4` decoking/trip events/yr, `6.8%` olefin yield loss, `3.4%` tube rupture risk, `$1,350,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM DCS Alarm & Operator Shift-Log Advisor (`16.2` events/yr, `7.5%` yield loss, `4.8%` tube rupture/runaway risk, `$1,480,000/mo` loss)
3. **Tier 3**: Unconstrained Empirical Neural-Net Soft Sensor (`6.4` events/yr, `2.9%` yield loss, `1.1%` tube rupture risk, `9.2%` TAR slip, `$490,000/mo` loss)
4. **Tier 4**: Online `NIR PIONA` Feed-Forward + Coking-Kinetic `TMT MPC` + `IEC 61511 SIS` Clamp + `TAR SIMOPS MILP` Agent (`0.4` events/yr, `0.5%` yield loss, `0.0%` tube rupture risk, `1.4%` TAR slip, `$110,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/run.py
python experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/run.py --check
python -m unittest tests/test_enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection.py
```
