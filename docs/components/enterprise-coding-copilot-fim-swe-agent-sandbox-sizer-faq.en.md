# Enterprise `10,000`-Developer Coding Copilot (`<150ms` FIM) & Autonomous SWE Agent Architecture: Dual-Model Routing, `Tree-sitter AST` Prefix Cache & `Firecracker MicroVM` Sandbox Sizer

> **Language / 语言**: [简体中文](./enterprise-coding-copilot-fim-swe-agent-sandbox-sizer-faq.md) | **English** · [📱 Open Interactive H5 Study Lab](../h5/coding-copilot-fim-swe-agent-sandbox-sizer-lab.html) · [🧮 Open Interactive Sizer](../reports/enterprise-coding-copilot-fim-swe-agent-sandbox-sizer.html) · [🃏 View 7-Card 4:5 Benchmark Deck](../reports/enterprise-coding-copilot-fim-swe-agent-sandbox-sizer-cards.html)

![Enterprise 10,000-Developer Coding Copilot FIM and SWE Agent Sizer](../../assets/svg/enterprise-coding-copilot-fim-swe-agent-sandbox-sizer-faq-summary.svg)

---

## 0. 3-Minute Executive Summary

When sizing a private AI software engineering platform for **`10,000` enterprise developers (`3,500 DAU`)**, combining inline autocomplete (`FIM`) and multi-file autonomous coding (`SWE Agent`) on a single `70B/671B` GPU pool violates the `< 120 ms` FIM latency SLA and inflates hardware spend by `91%`:
1. **Dual-Track Heterogeneous GPU Pool (`16× L40S` FIM + `16× H200` SWE Agent)**:
   - **Track A (`85%` of requests — Inline Ghost-Text `FIM`, `P95 TTFT <= 120 ms`)**: Served by **`16× L40S` running `14B FP8` + Prompt Lookup Decoding (`PLD 2.8×`)**. Enforcing **`75 ms` client keystroke debounce + instant `Cancel-on-Keystroke`** eliminates **`62.0%`** of obsolete in-flight requests (`392 RPS → 149 RPS`) and holds `P95 TTFT` at **`68 ms`**;
   - **Track B (`15%` of requests, `72%` of tokens — Multi-File `SWE Agent`)**: Served by **`16× H200` running `70B/671B FP8`**.
2. **`Tree-sitter AST` Cold-to-Hot Prefix Ordering Lifts Repo KV Hit Rate from `31.0%` to `84.2%`**:
   - Ordering context strictly from static repo rules & AST import signatures (`Cold`) to the active cursor function (`Hot`) cuts `32K` SWE Agent `TTFT` from `2,450 ms` to **`420 ms` (`-82.9%`)**.
3. **`Firecracker MicroVM` Sandbox Pool & FinOps ROI**:
   - `4× 128-Core CPU` nodes run `240` concurrent `125 ms` snapshot-restored MicroVMs (`1.4 GB/VM` with KSM/CoW). Total monthly TCO is **`$38,400/mo` (`$3.84/dev/mo`)** vs. **`$190,000/mo`** for commercial SaaS seats — **saving `$1.819M/yr` (`-79.8%`) with `100%` source-code sovereignty**.

---

## 1. Quantitative Architecture Matrix (`10,000` Developers · `3,500 DAU`)

| Coding Platform Architecture | FIM `P95 TTFT` | SWE Agent `32K TTFT` | Code Execution Isolation | Monthly Fleet & TCO | Cost / Dev / Month & Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Commercial SaaS Seat Licenses** | `140 ms` | `950 ms` | Source leaves VPC | `$190,000 / mo` (`10K × $19`) | `$19.00` · ⚠️ **Blocked for regulated IP** |
| **2. Single `70B` Pool (No Debounce / Raw Order)** | `385 ms` (miss) | `2,450 ms` (`31%` hit) | Standard Docker (unsafe) | `48× H200` (`$73,400 / mo`) | `$7.34` · ⛔ **Violates `<120ms` FIM SLA** |
| **3. Dual-Track + Raw Prompt Order** | `75 ms` | `1,820 ms` (`31%` hit) | `gVisor` containers | `16× L40S + 24× H200` (`$48,200 / mo`) | `$4.82` · ✅ **Viable but heavy Prefill** |
| **4. Dual-Track (`16×L40S + 16×H200`) + AST Cache + MicroVM** | **`68 ms`** | **`420 ms`** (`84.2%` hit) | **`Firecracker MicroVM` (`125ms`)** | **`16× L40S + 16× H200 + 4× CPU` (`$38,400 / mo`)** | **`$3.84` (saves `$1.82M/yr` / `-79.8%`)** 🏆 |
