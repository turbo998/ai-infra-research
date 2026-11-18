# Enterprise Railway & High-Speed Rolling-Stock `WILD / HBD`, Catenary `VLM` & Track-Possession `MILP` Agent (`R451`)

Deterministic evaluation suite comparing four railway rolling-stock wayside monitoring (`WILD / HBD / TADS`), overhead catenary pantograph inspection, and nightly track-possession maintenance scheduling architectures across a `4,200 km` rail network (`48,000` wheelsets, `160` wayside detector sites, `$65,000/hr` mainline delay cost):

1. **Tier 1**: Fixed-Mileage Wheel Lathe Turning + Single-Threshold Wayside Alarms (`19.2` mainline disruptions/mo, `24.5%` false train-stop rate, `21.8%` wheelset rim waste, `$960,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Wayside Alarm & Track-Inspector Log Chatbot (`17.5` disruptions/mo, `27.0%` false train-stop rate, `4.8%` catenary/rail defect escape, `$1,080,000/mo` loss)
3. **Tier 3**: Standalone Wayside Sensor Anomaly ML (`6.8` disruptions/mo, `8.4%` false train-stop rate, `10.5%` wheelset waste, `$410,000/mo` loss)
4. **Tier 4**: Longitudinal RFID Axle-Kalman `WILD/HBD/TADS` + Roof Catenary Arc/Stagger `VLM` + `EN 13848` Track-Possession `MILP` Agent (`0.4` disruptions/mo, `0.3%` false train-stop rate, `1.6%` wheelset waste, `0.0%` catenary/rail defect escape, `$84,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection/run.py
python experiments/enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection/run.py --check
python -m unittest tests/test_enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection.py
```
