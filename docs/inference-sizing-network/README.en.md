# AI Infra Inference, Fabric & Capacity Academy

**Language**: [简体中文](README.md) | English · [Open interactive academy](../h5-academy/index.html?lang=en)

For enterprise **Cloud & AI Solution Architects**, connect “why users see slow responses” to throughput, memory, GPUs and fabric requirements. No acronym soup and no simulated claims presented as measurements. Each chapter includes bilingual lessons, an SVG workbench, hand calculations, FAQ, a quiz and production checklist.

## Six-chapter route

| Chapter | Decision you can make | Interactive entry |
|---|---|---|
| User-facing metrics | Define TTFT, mean TPOT, tail ITL, E2EL, RPS, goodput and cost scopes | [01 Metrics and lifecycle](../h5-academy/01-metrics.html?lang=en) |
| TPM/TPS and capacity | Convert QPS into input/output demand, Little's Law concurrency, instance floors and normalized PTU | [02 Capacity planning](../h5-academy/02-tpm-tps-capacity.html?lang=en) |
| Three GPU bounds | Separate loaded weights, KV, runtime, HBM/GDDR bandwidth and prefill compute | [03 GPU sizing](../h5-academy/03-gpu-sizing.html?lang=en) |
| PD disaggregation | Budget colocated/chunked interference, KV transfer, pool cost and decision boundaries | [04 PD and KV handoff](../h5-academy/04-pd-disaggregation.html?lang=en) |
| Parallelism and acceleration | Understand TP/PP/DP/EP partitions, collectives, speculation, prefix reuse and FP8 tradeoffs | [05 Parallelism](../h5-academy/05-parallelism-acceleration.html?lang=en) |
| AI fabric | Explain GPUDirect RDMA, RoCEv2, NVLink/NVSwitch, rails and NCCL algbw/busbw | [06 Fabric](../h5-academy/06-rdma-roce-nvlink.html?lang=en) |

**30-minute route**: lifecycle in 01 → default capacity example in 02 → three GPU bounds in 03.

**Evening deep route**: read all six, changing one variable at a time. Increase concurrency from 1 to 32, then context from 2K to 32K. Observe bottleneck movement across memory, MoE expert union, goodput and KV fabric load.

## Bilingual FAQ and evidence boundaries

- [Chinese FAQ, hand calculations, decision matrix and checklist](faq.md)
- [English FAQ, hand calculations, decision matrix and checklist](faq.en.md)
- [Source catalog and calculation boundaries](../h5-academy/sources.json), reviewed 2026-11-21.

**Not a measured benchmark report**: SLO targets, workload distributions, prices, tradeoff and P99 curves are original teaching assumptions labeled `synthetic`. PD request-P99 TPOT uses 200 request means; pooled-P99 ITL uses the same 6200 gaps. Hardware specifications and software capabilities are `specification` / `implementation`, not performance promises. No independently reproduced hardware throughput or latency.

Key corrections: L40S uses GDDR6 and has no NVLink; B200 192 GB design/what-if capacity is distinct from DGX B200's delivered 180 GB/GPU specification; loading DeepSeek MTP adds 14B beyond 671B main weights; an 80–85% margin is architecture planning, not vLLM's current default.

## Offline and mobile

Keep `docs/h5-academy/` and double-click `index.html`. All seven HTML files embed CSS, SVG, formulas, native JS and both languages. No CDN, external fonts/images or automatic network requests. Each chapter can be copied and opened alone; cross-chapter navigation requires neighboring files.

Language/theme controls preserve selections on cross-chapter navigation. Mobile chapter cards swipe horizontally; wide figures/tables scroll within their own regions. Keyboard controls, explicit invalid-input errors, one-token N/A, print/PDF and reduced-motion preferences are supported. No user-data uploads or third-party analytics.

## Rebuild and verify

Run from the repository root (Node.js standard library only; no npm install):

```powershell
node docs\h5-academy\build.mjs
node --test docs\h5-academy\academy.test.mjs
node docs\h5-academy\build.mjs --check
python scripts\validate_research.py
```

With Windows Microsoft Edge installed, run an additional true offline desktop/mobile check without third-party packages:

```powershell
node docs\h5-academy\browser-check.mjs
```

An isolated temporary profile loads all seven pages at 1440px Chinese/light and 390px English/dark. It checks field boundaries/options/errors, quizzes, theme/language, page overflow, JS/CSP and automatic resource requests. A probe is added only to temporary test copies, never the delivered HTML.

Closure checks also cover 320px English/dark, 360px Chinese/light, native keyboard controls, selected text/graphic contrast and concise assistive-status announcements. Each chapter starts with one “First experiment tonight” and chart coordinates can expand into tables. This is evidence for those checks, not a complete WCAG conformance certification.

Check only an affected page with:

```powershell
node docs\h5-academy\browser-check.mjs --only 03-gpu-sizing.html
```

Sources stay in `h5-academy`: `content.mjs` contains bilingual lessons, `academy-math.cjs` the tested formula kernel, `academy-ui.js` interactions, `academy.css` themes and `build.mjs` resource inlining plus script CSP hashes. Regenerate after edits; `--check` catches stale generated pages.

Generation rejects missing/duplicate source IDs, obvious private/credential-bearing URLs, missing language fragments and false hardware-measurement attribution so citations cannot silently disappear. Sources and formulas still require human technical review.

## Navigation integration

At unified publication, link the academy from the repository's inference/network learning entries and archive seven HTML pages plus FAQ links in the personal homepage's AI Infra learning section. Preserve relative paths. This isolated delivery directory does not modify shared catalogs or the root README, avoiding parallel integration conflicts.
