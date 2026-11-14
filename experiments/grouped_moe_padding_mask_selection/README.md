# Round159: grouped MoE padding mask/coverage/sentinel selection

Original precomputed route IDs,integer weight numerators/unit10andscalar dispatch;no logits,grouped-topk,GEMMs,EPLB,CUDA/AITERorDeepEPexecution.

```powershell
python -m experiments.grouped_moe_padding_mask_selection.run --check
python -m unittest discover -s tests -p test_grouped_moe_padding_mask_selection.py -q
```

`--measure` replaces CPU samples;ordinary generation retains sealed values.Primary calls16→8,live-output misses0/4,logical route payload128Bunchanged.Eight policy cases include AITER/env/grouping/coverage distinctions.Unsupported skip consumers and legacy undercoverage raise original explicit errors;native legacy outcomes remain unknown.

[中文 FAQ](../../docs/components/grouped-moe-padding-mask-coverage-sentinel-selection-faq.md) / [English FAQ](../../docs/components/grouped-moe-padding-mask-coverage-sentinel-selection-faq.en.md).CPUzero-weight baseline produces identical outputs but executes extra scalar calls;not native FLOPs/throughput.No browser visual acceptance.
