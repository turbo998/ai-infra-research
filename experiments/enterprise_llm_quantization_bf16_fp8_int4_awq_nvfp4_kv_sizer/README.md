# `#651` Enterprise LLM Quantization Matrix (`BF16` vs. `Block FP8` vs. `INT4 AWQ` vs. `Blackwell NVFP4`) & `FP8 KV Cache` Sizer

Reproducible analytical verification for `#651`:
- **Chinese FAQ**: [`../../docs/components/enterprise-llm-quantization-bf16-fp8-int4-awq-nvfp4-kv-sizer-faq.md`](../../docs/components/enterprise-llm-quantization-bf16-fp8-int4-awq-nvfp4-kv-sizer-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-llm-quantization-bf16-fp8-int4-awq-nvfp4-kv-sizer-faq.en.md`](../../docs/components/enterprise-llm-quantization-bf16-fp8-int4-awq-nvfp4-kv-sizer-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/llm-quantization-bf16-fp8-int4-awq-nvfp4-kv-lab.html`](../../docs/h5/llm-quantization-bf16-fp8-int4-awq-nvfp4-kv-lab.html)

```bash
python experiments/enterprise_llm_quantization_bf16_fp8_int4_awq_nvfp4_kv_sizer/run.py
pytest tests/test_enterprise_llm_quantization_bf16_fp8_int4_awq_nvfp4_kv_sizer.py -q
```
