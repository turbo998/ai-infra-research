# Enterprise Multi-LoRA Serving (`64+` Adapters on Shared 70B Base via `vLLM / SGLang S-LoRA`) Sizer (`#517`)

Reproducible analytical model comparing **Dedicated Merged-Weight Pods (`128x H200`)**, **Blocking Disk LoRA Swap (`1,650ms` TTFT)**, and **Shared-Base Multi-LoRA Serving (`vLLM / SGLang S-LoRA + Paged HBM/DRAM Adapter Pool`, `8x H200`, `-93.8%` GPUs)**.

## Run

```bash
python experiments/enterprise_multi_lora_serving_slora_punica_shared_base_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-multi-lora-serving-slora-punica-shared-base-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-multi-lora-serving-slora-punica-shared-base-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/multi-lora-serving-slora-punica-shared-base-sizer-lab.html](../../docs/h5/multi-lora-serving-slora-punica-shared-base-sizer-lab.html)
