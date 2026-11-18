# Enterprise Edge-Cloud Hybrid AI & On-Device NPU SLM Routing Evaluator 

Deterministic analytical evaluator comparing four enterprise Edge-Cloud inference architectures across `50,000` endpoints (`AI PCs, smart vehicles, retail store edge boxes, factory tablets`) and `30,000,000` queries/month:

1. **Tier 1 (`tier1_cloud_only_wan_inference`)**: 100% Cloud-Only 70B Inference over WAN (`94.5%` accuracy, `96.2%` availability under WAN/tunnel blips, `620.0 ms` P95 TTFT, `$68,000/mo` cloud GPU bill).
2. **Tier 2 (`tier2_edge_only_npu_3b_slm`)**: 100% On-Device Edge-Only 3B SLM (`72.0%` accuracy, `100.0%` availability, `45.0 ms` P95 TTFT, `$0/mo` cloud GPU bill).
3. **Tier 3 (`tier3_static_rule_edge_cloud_split`)**: Static Keyword Edge-Cloud Split (`55%` Edge / `45%` Cloud, `86.4%` accuracy, `310.0 ms` P95 TTFT, `$31,500/mo`).
4. **Tier 4 (`tier4_confidence_cascade_edge_pii_scrub_hybrid`)**: Confidence-Gated Edge-Cloud Cascade (`78%` Local NPU 3B SLM + `22%` Low-Confidence Escalation to Cloud 70B with On-Device PII Scrubbing & Offline Fallback, `95.2%` accuracy [`+23.2 pts` vs. Tier 2], `99.98%` availability, `110.0 ms` blended P95 TTFT [`5.64x` faster than Cloud-Only], `$14,800/mo` [`-78.2%` cloud cost savings vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_edge_cloud_hybrid_npu_slm_routing_selection/run.py
python experiments/enterprise_edge_cloud_hybrid_npu_slm_routing_selection/run.py --check
```
