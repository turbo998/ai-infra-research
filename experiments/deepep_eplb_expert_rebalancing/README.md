# DeepSeek EPLB Redundant Expert Packing & SGLang 5-Tier Online P2P Weight Migration

Deterministic stdlib reproduction of `deepseek-ai/EPLB` (`eplb.py`: `balanced_packing`, `replicate_experts`, `rebalance_experts_hierarchical`) and `sgl-project/sglang` (`python/sglang/srt/eplb/eplb_manager.py` & `expert_location_updater.py`).

## Run & Verify

```bash
python experiments/deepep_eplb_expert_rebalancing/run.py --check
```
