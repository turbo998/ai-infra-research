# LLM Container Registry & 140GB Weight Distribution Selection Experiment (`Nydus/Dragonfly P2P` + `Run:ai Model Streamer / GDS / RDMA Broadcast` vs. `Baked OCI Image`)

Reproducible 64-node scale-out cold-start latency, central registry egress traffic, and idle GPU waste model comparing 4 container and weight distribution architectures.

## Run
```bash
python experiments/llm_weight_loading_container_registry_p2p_dragonfly_gds_safetensors_selection/run.py
python experiments/llm_weight_loading_container_registry_p2p_dragonfly_gds_safetensors_selection/run.py --check
```
