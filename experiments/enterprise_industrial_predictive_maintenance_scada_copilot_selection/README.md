# Enterprise Industrial Predictive Maintenance, PLC/SCADA Anomaly Triage & Field-Service Copilot Evaluator

Deterministic evaluator comparing four industrial predictive maintenance and field-service copilot architectures across 12 smart factories, 8,000 critical equipment assets, 120,000 SCADA/PLC sensor tags, and 45,000 pages of OEM wiring schematics & manuals:
- **Tier 1**: Legacy Static SCADA Threshold Alarms + Manual PDF/Paper Schematic Lookup (`61.0%` root-cause accuracy, `38.5%` false alarm rate, `6.8 h` MTTR, `$185,000/mo`)
- **Tier 2**: Raw High-Frequency Sensor CSV Stuffed Directly into Text LLM Prompt (`54.5%` root-cause accuracy, `44.0%` false alarm rate, `5.2 h` MTTR, `$240,000/mo`)
- **Tier 3**: Edge Time-Series Anomaly Detector + Text-Only Maintenance Manual RAG (`79.2%` root-cause accuracy, `8.4%` false alarm rate, `2.6 h` MTTR, `$76,000/mo`)
- **Tier 4**: Edge FFT/TimesFM Anomaly Triage + Multimodal Schematic RAG (`ColPali / VLM`) + SAP PM / Maximo Spare-Part & LOTO Safety Agent (`94.6%` root-cause accuracy, `2.1%` false alarm rate, `1.1 h` MTTR, `$28,000/mo`)

## Usage
```bash
python experiments/enterprise_industrial_predictive_maintenance_scada_copilot_selection/run.py
python experiments/enterprise_industrial_predictive_maintenance_scada_copilot_selection/run.py --check
```
