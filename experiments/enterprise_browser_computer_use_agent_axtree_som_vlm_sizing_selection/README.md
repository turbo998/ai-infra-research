# Enterprise Browser-Use & GUI Computer-Use Agent (`Pruned AXTree + SoM 7B UI-VLM` vs. `1080p Screenshot VLM Loop`) Sizer (`#502`)

Reproducible analytical model comparing 4 enterprise Browser-Use / Computer-Use RPA architectures across `500 concurrent browser sessions` executing 15-step ERP/CRM workflows:
- **Tier 1 (`Pure 1080p Screenshot Loop + 72B VLM`)**: `71.2%` workflow completion rate (pixel coordinate drift), `48.5s` E2EL, requires **`64x H200 GPUs`** (`$145 / 1k workflows`).
- **Tier 2 (`Raw Unpruned HTML Dump + Text LLM`)**: `64.5%` completion rate (`185,000` HTML tokens/workflow), `48x H200 GPUs`.
- **Tier 3 (`Uniform Set-of-Marks SoM Screenshot + 7B UI-VLM`)**: `88.4%` completion rate, `14.2s` E2EL, `16x H200 GPUs`.
- **Tier 4 (`Hybrid Pruned AXTree 85% + SoM 7B UI-VLM Fallback 15% + Warm CDP Context Pool`)**: Raises workflow completion rate to **`95.6%` (`+24.4 pts`)**, speeds up E2EL **`6.22x` (`7.8s` vs. `48.5s`)**, and serves 500 concurrent browser agents on just **`6x H200 GPUs` (`-90.6%` GPUs)** at **`$8.20 / 1,000 workflows`**.

## Run
```bash
python experiments/enterprise_browser_computer_use_agent_axtree_som_vlm_sizing_selection/run.py
```
