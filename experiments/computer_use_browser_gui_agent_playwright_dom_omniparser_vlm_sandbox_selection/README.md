# Enterprise Computer-Use & Browser GUI Agent Selection Experiment (`Playwright A11y DOM` vs. `Pure Pixel VLM` vs. `Hybrid OmniParser v2 SoM`)

Reproducible analytical model comparing 4 Computer-Use / Browser Agent grounding architectures across a `15-step` enterprise workflow (`1920x1080` ERP/CRM/Canvas mix):
- **Tier 1 (`Raw Full-DOM HTML Dump`)**: `41.5%` task success, `85,000` tokens/step, `3,200 ms/step`, `$0.42/task`.
- **Tier 2 (`Pure Pixel Screenshot VLM (x, y) Coordinate Prediction`)**: `68.4%` task success, `1,850 ms/step`, `$0.24/task`.
- **Tier 3 (`Pruned Accessibility A11y Tree Playwright aria-ref`)**: `78.5%` task success, `420 ms/step`, `$0.06/task` (fails on `<canvas>` / unlabelled icons).
- **Tier 4 (`Hybrid A11y Tree + OmniParser v2 Set-of-Mark Bounding-Box IDs + Multi-Action Batching`, Recommended)**: **`91.6%` task success**, `540 ms/step` (`3.4x` faster than pure VLM), and `$0.09/task`.

## Run & Verify

```bash
python experiments/computer_use_browser_gui_agent_playwright_dom_omniparser_vlm_sandbox_selection/run.py
python experiments/computer_use_browser_gui_agent_playwright_dom_omniparser_vlm_sandbox_selection/run.py --check
```
