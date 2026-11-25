# Enterprise Computer-Use & Browser GUI Agent (`CUA / Playwright / OmniParser`) Infrastructure Sizer: Screenshot Visual Token Tax, Hybrid DOM + Set-of-Marks Grounding & 500-Session Sandbox Pool (`#667`)

> **Reader's Guide (3 Minutes for Key Takeaways, 10 Minutes for CUA Token & Sandbox Sizing Math)**
> - **Interactive H5 Study Lab**: [Open `computer-use-browser-gui-sandbox-sizer-lab.html`](../h5/computer-use-browser-gui-sandbox-sizer-lab.html)
> - **Interactive HTML Sizer**: [Open `enterprise-computer-use-browser-gui-sandbox-sizer.html`](../reports/enterprise-computer-use-browser-gui-sandbox-sizer.html)
> - **7-Card 4:5 Visual Deck**: [Open `enterprise-computer-use-browser-gui-sandbox-sizer-cards.html`](../reports/enterprise-computer-use-browser-gui-sandbox-sizer-cards.html)
> - **中文版本**: [阅读中文版指南 (`#667`)](./enterprise-computer-use-browser-gui-sandbox-sizer-faq.md)

![Computer-Use Agent Infrastructure Summary](../../assets/svg/enterprise-computer-use-browser-gui-sandbox-sizer-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Overview)

As enterprises replace brittle legacy RPA scripts with **Computer-Use Agents (`CUA / Playwright Browser Agents / GUI Agents`)** across ERP order entry, banking reconciliation, and procurement portals, Solution Architects repeatedly hit two infrastructure walls: **(1) Screenshot Visual Token Explosion** (`1920×1080` screenshots every step consume `~2,020` visual tokens/frame; a `15-step` task retaining visual history burns **`30,300` visual tokens**, inflating step `TTFT` to `1,420 ms` and requiring **`28× H200` GPUs** for `500` concurrent sessions), and **(2) Desktop VDI Sandbox Bloat** (`500` full Windows/Linux VMs at `8 GB RAM / 4 vCPU` consume **`4.0 TB` RAM across `16` dual-socket hosts** with `28-second` cold boots):

1. **Slash Visual Tokens by `78.4%` (`30,300 → 6,540` tok/task) via Hybrid `Accessibility/DOM` Tree (`75%` steps) + `OmniParser` Set-of-Marks (`SoM`) ROI Crops (`25%` steps) + 2-Frame Sliding Window**:
   - **Never retain full screenshot history**: Evict raw screenshots older than `t - 1` (retaining only the current screen `S_t`, previous screen `S_{t-1}`, and an `80-token` structured action log).
   - **Use Pruned DOM/Accessibility Trees first, `OmniParser SoM` second**: For `75%` of standard web steps, pass pruned interactive DOM nodes (`~320` tokens/step) instead of `2,020-token` images; for the `25%` of canvas/legacy ERP steps, overlay discrete **`Set-of-Marks` (`[1]..[42]`) bounding-box IDs** via `OmniParser`. This boosts UI action accuracy from `84.5%` to **`94.8% (+10.3 pts)`**, cuts step `TTFT` from `1,420 ms` to **`310 ms` (`4.58×` faster)**, and shrinks the `500`-session VLM fleet from `28` down to **`6× H200` GPUs (`-78.6%`)**.
2. **Boost Sandbox Host Density by `4.44×` via Containerized `Headless Chromium + Xvfb / Firecracker` Snapshot Pools (`1.8 GB/session`)**:
   - Replacing full `8 GB` VDI VMs with pre-warmed `Chromium + Xvfb` microVM snapshots shrinks `500`-session host memory from `4.0 TB` (`16` hosts) to **`0.9 TB` (`4` hosts, `-75%`)** and cuts sandbox startup from `28 s` to **`140 ms`**.

---

## 2. Computer-Use Agent Architecture Decision Matrix (`500` Concurrent Sessions)

| Architecture Mode | Cumulative Visual/UI Tokens (15 Steps) | Step `TTFT` (`H200`) | UI Action Accuracy | VLM Fleet (`H200`) | Sandbox CPU Hosts (`128C/512GB`) | Sandbox Cold Start | Recommended Scenario |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1. Naive Full `1080p` Screenshot History + Full VDI** | `30,300 tok` | `1,420 ms` | `84.5%` | **`28 GPUs`** | **`16 Hosts (4.0 TB)`** | `28.0 s` | ❌ **Anti-pattern (severe token & RAM bloat)** |
| **2. `1080p` 2-Frame Sliding Window + Full VDI** | `14,500 tok (-52.1%)` | `540 ms` | `86.2%` | **`14 GPUs`** | `16 Hosts` | `28.0 s` | Legacy thick-client `.exe` apps without web UI |
| **3. Hybrid `A11y/DOM + OmniParser SoM` + `Chromium` Pool** | **`6,540 tok (-78.4%)`** | **`310 ms`** | **`94.8%`** | **`6 GPUs (-78.6%)`** | **`4 Hosts (0.9 TB, -75%)`** | **`0.14 s`** | **Production enterprise web ERP, finance & CRM agents** |

---

## 3. Pre-Production Checklist

- [ ] **Evict Screenshots Older Than `t - 1`**: Keep at most `2` visual frames in KV context; summarize prior steps as compact structured text actions.
- [ ] **Replace Raw `(x, y)` Pixel Guessing with `Set-of-Marks (SoM)` Box IDs**: Bind every click/type action to a deterministic DOM element ID or `OmniParser` bounding-box index.
- [ ] **Enforce Human-in-the-Loop (HITL) Approval Gates on Destructive UI Actions**: Pause execution and surface a cropped confirmation screenshot before clicking buttons matching payment, wire transfer, or record deletion rules.
