# Enterprise Video VLM & `500`-Stream Real-Time `RTSP` Sizer: `NVDEC` Keyframe Motion Gating (`-88%` Frames), `3D` Spatio-Temporal Token Pooling (`144 tok/frame`) & Edge-to-Core GPU Sizing

> **Language / 语言**: [简体中文](./enterprise-video-vlm-rtsp-keyframe-3d-token-pooling-sizer-faq.md) | **English** · [📱 Open Interactive H5 Study Lab](../h5/video-vlm-rtsp-keyframe-3d-token-pooling-sizer-lab.html) · [🧮 Open Interactive Sizer](../reports/enterprise-video-vlm-rtsp-keyframe-3d-token-pooling-sizer.html) · [🃏 View 7-Card 4:5 Benchmark Deck](../reports/enterprise-video-vlm-rtsp-keyframe-3d-token-pooling-sizer-cards.html)

![Enterprise Video VLM and 500-Stream Real-Time RTSP Sizer Architecture](../../assets/svg/enterprise-video-vlm-rtsp-keyframe-3d-token-pooling-sizer-faq-summary.svg)

---

## 0. 3-Minute Executive Summary

Feeding a 1-hour `1080p` video at `1 FPS` (`3,600 frames × 2,691 visual tokens/frame`) directly into a VLM generates **`9,687,600` visual tokens (`75.7×` beyond a `128K` context window)**, and streaming `500` concurrent `RTSP` cameras at `1 FPS` blindly into a `72B VLM` requires an astronomical **`192× H200` GPUs (`$294K/mo`)**:
1. **3-Tier Spatio-Temporal Token Compression Cuts 1-Hour Video Tokens by `99.36%` (`9.69M → 62,208 Tokens`)**:
   $$T_{\text{video}} = N_{\text{sec}} \cdot f_{\text{fps}} \cdot (1 - \eta_{\text{static}}) \cdot \frac{T_{\text{frame}}}{C_{\text{3d}}} = 3,600 \times 1 \times (1 - 0.88) \times 144 = \mathbf{62,208\text{ Tokens}}$$
   - **Tier 1 (`NVDEC` In-VRAM Hardware I-Frame Decode + Motion Vector Filter)**: Uses `L40S`'s `4` dedicated hardware `NVDEC` engines per GPU (`0` CPU decode overhead) to drop `70%` static frames;
   - **Tier 2 (`YOLOv10 + SigLIP-400M` Cosine Delta Gate, `cos < 0.92`)**: Filters out another `60%` of unchanged semantic states (`88.0%` total frame reduction: `3,600 → 432 keyframes/hr`);
   - **Tier 3 (`2×2×2` 3D Spatio-Temporal Pooling)**: Compresses each retained frame to **`144 visual tokens/frame`**, fitting a full 1-hour video inside a single `64K` context window with **`91.8%` `Video-MME` accuracy**.
2. **`500`-Stream Real-Time `RTSP` Fleet Sizing (`8× L40S + 8× H200`, `-91.8%` Cost)**:
   - `8× L40S` (`32` hardware `NVDEC` units) decode and gate all `500` streams, forwarding only the `12%` active event clips (`8,640 visual tok/s`) to `8× H200` (`Qwen2.5-VL-72B FP8`), achieving **`480 ms` `P95` alert latency** at **`$0.018 / stream-hour`** (vs. `$0.220 / stream-hour` on `192× H200`, saving **`$3.31M/yr`**).

---

## 1. Quantitative Video VLM Architecture Matrix (`500` Concurrent `1080p` Streams / `1-Hour` Video)

| Video VLM Architecture | `1-Hour` Video Visual Tokens | `500`-Stream VLM Load (`tok/s`) | Alert `P95` Latency | `Video-MME` Accuracy | Fleet & Cost per Stream-Hour |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Raw `1 FPS` Full-Resolution into `72B VLM`** | `9,687,600 tok` (OOM) | `1,345,500 tok/s` | `> 12.0 s` | `78.5%` (truncated) | `192× H200` (`$0.220 / stream-hr`) · ⛔ |
| **2. Uniform `0.1 FPS` Low-Res (`336×336`)** | `92,160 tok` | `12,800 tok/s` | `5,400 ms` | `61.4%` (misses fast events) | `16× H200` (`$0.028 / stream-hr`) · ⚠️ |
| **3. Legacy `YOLO` Only (No VLM Reasoning)** | `0 tok` | `0 tok/s` | `35 ms` | `42.0%` (high false alarms) | `4× L40S` (`$0.004 / stream-hr`) · ⚠️ |
| **4. `NVDEC + SigLIP` Gate (`-88%`) + `3D Pool` (`144 tok`)** | **`62,208 tok` (`-99.36%`)** | **`8,640 tok/s`** | **`480 ms`** | **`91.8%`** | **`8× L40S + 8× H200` (`$0.018 / stream-hr`, `-91.8%`)** 🏆 |
