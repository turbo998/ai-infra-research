# Hybrid Edge-Cloud AI & On-Device SLM Selection Experiment (`Apple MLX` vs. `Qualcomm QNN / Intel OpenVINO NPU` vs. `llama.cpp / ExecuTorch`)

Reproducible fleet capacity and TCO model comparing 4 edge-cloud deployment strategies across a `50,000`-seat enterprise workforce (`45 queries/day/user`) for local autocomplete `TTFT`, required cloud `H200` GPUs, composite quality, PII zero-egress compliance, and monthly cloud TCO.

## Run
```bash
python experiments/edge_on_device_slm_npu_apple_mlx_qualcomm_qnn_hybrid_cloud_routing_selection/run.py
python experiments/edge_on_device_slm_npu_apple_mlx_qualcomm_qnn_hybrid_cloud_routing_selection/run.py --check
```
