# Enterprise K8s Gateway API Inference Extension (`InferencePool` + `EPP`) & `llm-d` KV/LoRA Routing (`R452`)

Deterministic evaluation suite comparing four Kubernetes ingress and traffic-routing architectures across a `64-GPU (H100 80GB)` cluster (`16` `vLLM/SGLang` replicas, `1,200 RPS` mixed-length traffic, `48` departmental LoRA adapters):

1. **Tier 1**: Standard Kubernetes `Service` / L4-L7 Round-Robin Ingress (`4,480 ms` P99 TTFT, `16.5%` prefix/LoRA hit rate, `14.8%` KV preemption drop, `$148,000/mo` GPU TCO)
2. **Tier 2**: HTTP Least-Connections + CPU/Memory HPA Ingress (`3,120 ms` P99 TTFT, `21.0%` prefix/LoRA hit rate, `8.6%` KV preemption drop, `$126,000/mo` GPU TCO)
3. **Tier 3**: Client-Side Consistent-Hash Sticky Tenant Routing (`1,450 ms` P99 TTFT, `64.0%` prefix/LoRA hit rate, `5.4%` KV preemption drop, `$92,000/mo` GPU TCO)
4. **Tier 4**: K8s Gateway API Inference Extension (`InferencePool` + `EPP`) + `llm-d` KV/LoRA Composite Scorer + 3-Band Criticality Shedder (`340 ms` P99 TTFT, `89.4%` prefix/LoRA hit rate, `0.0%` KV preemption drop, `0.1%` critical SLO breach, `$54,000/mo` GPU TCO)

## Usage

```bash
python experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/run.py
python experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/run.py --check
python -m unittest tests/test_enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection.py
```
