# Enterprise Pipeline Midstream `RTTM` Leak Detection, `ILI` Corrosion (`ASME B31G`) & Drilling Stuck-Pipe Agent (`R446`)

Deterministic evaluation suite comparing four midstream pipeline leak detection (`API 1130`), Inline Inspection (`ILI` smart-pig) corrosion fitness-for-service (`ASME B31G / API 579`), and directional drilling stuck-pipe prevention architectures across a `6,800 km` transmission network (`140,000` ILI anomalies, `14` active drilling rigs, `$45,000` per integrity dig):

1. **Tier 1**: 5-Year ILI Pigging Excel + Static 5% SCADA Line-Balance (`180.0 min` pinhole leak detection, `19.5%` false shut-in rate, `38.0%` unnecessary integrity dig overkill, `64.0 hrs/mo` stuck-pipe NPT, `$980,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM SCADA Alarm & Mud-Log Summarizer (`95.0 min` detection, `24.0%` false shut-in, `41.5%` dig overkill, `58.0 hrs/mo` NPT, `$1,140,000/mo` loss)
3. **Tier 3**: Standalone Acoustic/Flow ML Without RTTM Linepack or B31G Physics (`28.0 min` detection, `8.2%` false shut-in, `16.5%` dig overkill, `26.0 hrs/mo` NPT, `$460,000/mo` loss)
4. **Tier 4**: `API 1130` Compressible `RTTM` + Negative-Pressure-Wave (`NPW`) Acoustic Time-of-Flight + `DTW` Girth-Weld `ASME B31G` Gate + Soft-String `Torque-Drag` Agent (`1.8 min` leak detection, `0.3%` false shut-in, `1.4%` dig overkill, `3.5 hrs/mo` stuck-pipe NPT, `$96,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection/run.py
python experiments/enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection/run.py --check
python -m unittest tests/test_enterprise_pipeline_midstream_leak_ili_corrosion_drilling_agent_selection.py
```
