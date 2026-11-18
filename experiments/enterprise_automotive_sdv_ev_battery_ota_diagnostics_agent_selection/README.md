# Enterprise Automotive Connected-Vehicle (`SDV`) Fleet Telemetry, EV Battery Early Warning & OTA Diagnostics Agent (`R436`)

Deterministic evaluation suite comparing four connected-vehicle (`SDV`) telemetry, EV battery safety prognostics, dealer warranty diagnostics, and OTA rollout architectures across a `500,000-vehicle` 800V EV fleet (`96` series cells/pack, `18,000` monthly dealer service tickets):

1. **Tier 1**: Legacy Dealer OBD-II Scan + Static Cloud BMS Thresholds (`0.001` days / 90s thermal runaway warning, `34.0%` dealer No-Trouble-Found NTF rate, `12.5%` bad OTA fleet exposure, `$890,000/mo` TCO)
2. **Tier 2**: Raw Full-Stream 100Hz Cloud CAN Upload + Prompt-Only LLM DTC Lookup (`0.5` days warning, `29.5%` NTF rate, `48.0 GB/vehicle/mo` cellular bill, `$1,120,000/mo` TCO)
3. **Tier 3**: Cloud Downsampled 60-Second BMS Anomaly Detector (`3.0` days warning, `16.8%` NTF rate, `3.5%` OTA exposure, `$420,000/mo` TCO)
4. **Tier 4**: Edge 100Hz Ring-Buffer Trigger + Cloud `dQ/dV` Electro-Thermal Physics + `UDS ISO 14229` Contactor Fault Tree + Stratified `A/B Canary OTA` Gate (`21.0` days early warning, `2.4%` dealer NTF rate, `0.0%` fleet brick exposure, `0.65 GB/vehicle/mo`, `$94,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_automotive_sdv_ev_battery_ota_diagnostics_agent_selection/run.py
python experiments/enterprise_automotive_sdv_ev_battery_ota_diagnostics_agent_selection/run.py --check
python -m unittest tests/test_enterprise_automotive_sdv_ev_battery_ota_diagnostics_agent_selection.py
```
