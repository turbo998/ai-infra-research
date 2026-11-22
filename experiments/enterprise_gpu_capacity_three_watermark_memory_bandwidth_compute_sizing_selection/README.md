# Enterprise LLM GPU Capacity Planning (`3-Watermark Sizer`: Memory, HBM Bandwidth, Compute) Model (`#447`)

Reproducible analytical model comparing weight-only naive GPU sizing, GDDR6 bandwidth-bound L40S sizing, 2-node 16x H100 80GB sizing, and 3-Watermark balanced single-node 8x H200 141GB / B200 180GB sizing (`DeepSeek-V3/R1 671B FP8 + 14B MTP`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection/run.py --check`
