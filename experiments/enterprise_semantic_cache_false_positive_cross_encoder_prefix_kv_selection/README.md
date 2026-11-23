# Enterprise Semantic Cache False-Positive Governance & `Exact Prefix KV Cache` Dual-Layer Sizer (`#524`)

Reproducible analytical model comparing **Naive Cosine `>= 0.85` Semantic Caching (`14.2%` false-positive wrong answers ✕)**, **Zero Caching (`920ms` TTFT, `32 GPUs` ✕)**, and **Dual-Layer Guarded Caching (`Tenant/Entity Slot Match + Cross-Encoder >= 0.96` [28.5% response hit, `0.08%` error] + `SGLang Radix Prefix KV Cache` [82% token KV reuse], `145ms` P95 TTFT, `10x H200 GPUs`, `-68.8%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_semantic_cache_false_positive_cross_encoder_prefix_kv_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-semantic-cache-false-positive-cross-encoder-prefix-kv-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-semantic-cache-false-positive-cross-encoder-prefix-kv-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/semantic-cache-false-positive-cross-encoder-prefix-kv-lab.html](../../docs/h5/semantic-cache-false-positive-cross-encoder-prefix-kv-lab.html)
