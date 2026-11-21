# Kubernetes Gateway API Inference Extension (GIE & Endpoint Picker EPP) Selection Experiment (`K8s GIE / EPP` vs. `Nginx Round-Robin` vs. `Envoy Least-Request` vs. `Consistent Hash`)

Reproducible latency, preemption rate, and Goodput model comparing 4 Kubernetes L7 load-balancing architectures across a `24`-pod (`48x H200`) `vLLM` cluster serving heterogeneous context lengths and `80` active LoRA adapters.

## Run
```bash
python experiments/kubernetes_gateway_api_inference_extension_gie_epp_kv_lora_routing_selection/run.py
python experiments/kubernetes_gateway_api_inference_extension_gie_epp_kv_lora_routing_selection/run.py --check
```
