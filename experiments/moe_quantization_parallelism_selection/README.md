# Large Model & MoE Quantization (`BF16` vs. `FP8` vs. `NVFP4/MXFP4`) & Parallelism (`TP` vs. `EP + DeepEP`) Sizer

Deterministic Roofline, communication-overlap, accuracy-retention, and TCO model backing the bilingual Architect Selection FAQ Guides:
- [中文 FAQ 指南](../../docs/components/moe-quantization-parallelism-selection-faq.md)
- [English FAQ Guide](../../docs/components/moe-quantization-parallelism-selection-faq.en.md)
- [7-Card 4:5 Benchmark Report](../../docs/reports/moe-quantization-parallelism-selection-cards.html)

## Usage

```bash
python experiments/moe_quantization_parallelism_selection/run.py --check
```
