# Enterprise Multi-LoRA Serving at Scale (`vLLM / SGLang Multi-LoRA`, `S-LoRA Unified Paging` & HBM Adapter Cache Sizer) (`#468`)

Reproducible analytical model comparing Dedicated Merged Pods, Unbatched Weight Hot-Swap, `vLLM/SGLang` Multi-LoRA Heterogeneous Batching, and `S-LoRA` Unified Paging + Locality-Aware Gateway Routing for `128` enterprise tenant adapters on `Llama-3.3-70B FP8`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_multi_lora_serving_slora_punica_hbm_cache_tco_selection/run.py --check`
