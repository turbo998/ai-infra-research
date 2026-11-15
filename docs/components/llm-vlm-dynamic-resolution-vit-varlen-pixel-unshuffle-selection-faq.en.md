# Multimodal `VLM` Dynamic-Resolution `ViT` & Visual Token Compression (`Qwen2.5-VL / InternVL`): Padded Batching (`68%` Waste) & Global $O(N^2)$ ViT Explosion vs. `Varlen Windowed ViT` + `2x2 Pixel Unshuffle` (`4x` Fewer LLM Tokens) Selection Guide

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Grounded in [`experiments/llm_vlm_vision_encoder_vit_padding_varlen_pixel_unshuffle_selection/results.json`](../../experiments/llm_vlm_vision_encoder_vit_padding_varlen_pixel_unshuffle_selection/results.json).
> **中文版本**: [多模态视觉大模型（`VLM: Qwen2.5-VL / InternVL`）高分辨率推理与训练瓶颈：图像 `Padding` 补齐浪费（`68%` 空白 Patch）与全图 $O(N^2)$ `ViT` 算力爆炸 vs. `Varlen Windowed ViT`（`cu_seqlens` + 局部窗口注意力）+ `2x2 Pixel Unshuffle` 视觉 Token `4x` 压缩选型指南](./llm-vlm-dynamic-resolution-vit-varlen-pixel-unshuffle-selection-faq.md)

![Multimodal VLM Varlen Windowed ViT & 2x2 Pixel Unshuffle Summary](../../assets/svg/llm-vlm-dynamic-resolution-vit-varlen-pixel-unshuffle-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **Slashing High-Resolution Multimodal `VLM` Time-To-First-Token (`TTFT`) by `12.03x` (`1,420.0 ms -> 118.0 ms`) and Cutting Visual `KV Cache` Footprint by `-75.0%` (`14,000 -> 3,500` LLM Tokens/Image) While Reaching `95.1%` `DocVQA` Accuracy via `Varlen Windowed ViT` (`cu_seqlens_window`) + `2x2 Pixel Unshuffle` (`Qwen2.5-VL / InternVL` Architecture, `Tier 4`)**: A single high-resolution `1960x1400` document or GUI screenshot sliced into `14x14` patches generates **`14,000` visual patches**. How we batch, encode, and project those `14,000` patches determines whether VLM serving stalls at `1.42 seconds` per image or runs in `118 ms`:
  1. **Tier 1 — Padded 3D `[Batch, MaxPatches, D]` `ViT` + Uncompressed `1:1` Patch-to-LLM Projection (`14,000` LLM Tokens, `1,420.0 ms` TTFT)**:
     - Padding variable-resolution images in a batch to `MaxPatches = 14,000` wastes `68%` of ViT compute on blank `[PAD]` tokens;
     - Worse, feeding all `14,000` patches `1:1` into a `72B` LLM backbone explodes quadratic LLM Prefill FLOPs and consumes `14,000` KV Cache slots per image!
  2. **Tier 2 — Ragged `Varlen Global ViT` (`cu_seqlens`) Without Visual Token Compression (`14,000` LLM Tokens, `680.0 ms` TTFT)**:
     - Concatenates image patches into a 1D ragged tensor (`0%` padding waste), **but** still runs `32` layers of full-image $O(N^2)$ ViT self-attention and feeds `14,000` tokens into the `72B` LLM!
  3. **Tier 3 — `Varlen Global ViT` + `2x2 Pixel Unshuffle / PatchMerger` (`14,000 -> 3,500` LLM Tokens, `245.0 ms` TTFT, `95.0%` DocVQA)**:
     - Concatenates spatially adjacent **`2 x 2 = 4` ViT patches along the channel dimension (`[2, 2, D_vit] -> [1, 4 * D_vit]`)** and projects them via a 2-layer MLP into **1 LLM token**—cutting LLM visual tokens and KV Cache memory by **`4.0x` (`-75%`)** with zero OCR accuracy loss! However, global $O(N^2)$ attention across `14,000` patches inside the `32`-layer ViT still takes `135 ms`.
  4. **Tier 4 — `Varlen Windowed ViT` (`28` Local `112x112` Windowed Layers + `4` Global Full-Attention Layers via `cu_seqlens`) + `2x2 Pixel Unshuffle` + `nvJPEG` GPU Decode (`3,500` LLM Tokens, `118.0 ms` TTFT, `12.03x` Speedup, `95.1%` DocVQA)**:
     - Replaces global $O(N^2)$ attention in `28` of the `32` ViT layers with **`8x8`-patch (`112x112` px) local Windowed Attention** dispatched via `flash_attn_varlen_func(cu_seqlens_window)` ($O(N)$ linear scaling!), keeping only `4` layers (`7, 15, 23, 31`) for global cross-window interaction!

---

## 1. 📌 Executive Summary

1. **Always Compress Visual Patches by `4x` via `2x2 Pixel Unshuffle` Before Entering the LLM (`Tier 3 & 4`)**: The LLM backbone (`7B–72B+`) is `10x–100x` heavier per token than the ViT (`0.4B–0.6B`). Merging `2x2` spatial patches into `4 * D_vit` channel vectors before the MLP projector reduces LLM sequence length and KV Cache footprint by `75%` while preserving `100%` of high-frequency OCR features.
2. **Implement Windowed + Global ViT Using Standard `flash_attn_varlen_func` (`Tier 4`)**: Permuting image patches into `8x8` (`64`-patch) window-contiguous order allows the exact same `flash_attn_varlen_func` kernel to run local Windowed Attention (by passing `cu_seqlens_window` with step `64`) on 28 layers and Global Attention (by passing `cu_seqlens_full`) on 4 layers!

---

## 2. 🎯 Core Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why doesn't `2x2 Pixel Unshuffle` (`14,000 -> 3,500` tokens) hurt fine-grained OCR accuracy on dense PDF tables?
- **Answer**: Unlike average pooling (which blurs spatial details), **Pixel Unshuffle (Space-to-Depth)** lossless-concatenates all 4 patches' feature vectors into a `4x`-wider vector `[4 * D_vit]` (`4 * 1280 = 5120` dims) *after* all 32 ViT layers have already extracted fine-grained `14x14` character features, and then projects `[5120 -> 3584]` into the LLM's high-dimensional hidden space!

### Q2: How do we prevent a single adversarial `8K x 8K` user image from causing an OOM spike in production VLM serving?
- **Answer**: Always enforce bounded dynamic resolution (`min_pixels = 256 * 28 * 28`, `max_pixels = 1280 * 28 * 28` or `16384 * 28 * 28`) in the image processor so every image is aspect-ratio-preservingly resized to fit within a strict upper bound of `28x28` merged token cells.

---

## 3. 🧭 Visual Diagrams

```mermaid
flowchart LR
    A["Tier 1: Padded 3D ViT + 14K LLM Tokens (1,420.0 ms TTFT ✕)"] --> B["Tier 3: Varlen Global ViT + 2x2 Pixel Unshuffle 3.5K Tokens (245.0 ms ✅)"]
    B --> C["Tier 4: Varlen Windowed ViT + 2x2 Pixel Unshuffle + nvJPEG (118.0 ms · 12.0x Faster 🏆)"]
```

---

## 4. 📊 4-Tier Architect Selection Matrix

| Metric | Tier 1: Padded ViT + `1:1` | Tier 2: `Varlen` Global + `1:1` | Tier 3: `Varlen` Global + `2x2 Unshuffle` | Tier 4: **`Varlen Windowed ViT` + `2x2 Unshuffle`** |
| :--- | :--- | :--- | :--- | :--- |
| **LLM Visual Tokens / Image** | **`14,000` (4x KV Bloat)** ✕ | **`14,000`** ✕ | **`3,500` (`-75.0%` KV Cache)** ✅ | **`3,500` (`-75.0%` LLM KV Cache)** 🏆 |
| **End-to-End `VLM TTFT` (`ms`)** | **`1,420.0 ms`** ✕ | `680.0 ms` ⚠️ | `245.0 ms` ✅ | **`118.0 ms` (`12.03x` Speedup)** 🏆 |
| **High-Res `DocVQA` Accuracy** | `94.6%` | `94.8%` | `95.0%` | **`95.1%` (Lossless Space-to-Depth)** 🏆 |

---

## 5. 🧪 Quantitative Reference & Tooling

- **Evaluator & Sealed Data**: [`experiments/llm_vlm_vision_encoder_vit_padding_varlen_pixel_unshuffle_selection/run.py`](../../experiments/llm_vlm_vision_encoder_vit_padding_varlen_pixel_unshuffle_selection/run.py) · [`results.json`](../../experiments/llm_vlm_vision_encoder_vit_padding_varlen_pixel_unshuffle_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-vlm-dynamic-resolution-vit-varlen-pixel-unshuffle-sizer.html`](../reports/llm-vlm-dynamic-resolution-vit-varlen-pixel-unshuffle-sizer.html)
- **7-Card 4:5 Visual Report**: [`docs/reports/llm-vlm-dynamic-resolution-vit-varlen-pixel-unshuffle-selection-cards.html`](../reports/llm-vlm-dynamic-resolution-vit-varlen-pixel-unshuffle-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & Production Checklist

- **✅ Good fit**: High-resolution OCR, document understanding, GUI agents, and video VLMs using `Varlen Windowed ViT` (`cu_seqlens`) + `2x2 Pixel Unshuffle`.
- **⛔ Not for**: Padded 3D ViT batching or feeding uncompressed `14x14` ViT patches `1:1` into large LLM backbones.
- [ ] Enable `flash_attn_varlen_func` with dual `cu_seqlens_window` and `cu_seqlens_full` inside the Vision Encoder.
- [ ] Apply `2x2 Pixel Unshuffle` (`PatchMerger`) before the LLM backbone and bound per-request `max_pixels`.
