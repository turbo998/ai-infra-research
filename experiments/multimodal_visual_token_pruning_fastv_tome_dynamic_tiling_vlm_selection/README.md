# Multimodal Visual Token Pruning & Compression Selection Experiment (`Progressive FastV / PyramidDrop` vs. `Unpruned Dynamic Tiling` & `Perceiver Resampler`)

Reproducible analytical model comparing 4 visual token compression tiers on a `72B` (`80-layer`) VLM processing `8,192` high-res visual patches:
- **Tier 1 (`Unpruned Dynamic Tiling` across all 80 layers)**: `1,820 ms` Prefill TTFT, `2.62 GB` KV Cache/request, `94.2%` DocVQA/ChartQA accuracy (`22` concurrent requests/node).
- **Tier 2 (`Pre-LLM 64-Token Perceiver Resampler` before Layer 1)**: `240 ms` TTFT, **but fine-grained OCR/table accuracy collapses to `68.5%`**.
- **Tier 3 (`2x2 Pixel-Shuffle / Token Merging ToMe` before LLM)**: `680 ms` TTFT, `1.15 GB` KV Cache, `92.8%` accuracy.
- **Tier 4 (`2x2 Merger + Progressive Layer-Wise Attention Pruning FastV / PyramidDrop` [`100%` in L1–2 $\to$ `50%` in L3–16 $\to$ `20%` in L17–80], Recommended)**: **`480 ms` Prefill TTFT (`3.79x` faster)**, **`0.58 GB` KV Cache (`-77.9%` HBM)**, **`98` concurrent requests/node (`4.45x` capacity)**, and **`94.0%` lossless OCR/Chart accuracy**.

## Run & Verify

```bash
python experiments/multimodal_visual_token_pruning_fastv_tome_dynamic_tiling_vlm_selection/run.py
python experiments/multimodal_visual_token_pruning_fastv_tome_dynamic_tiling_vlm_selection/run.py --check
```
