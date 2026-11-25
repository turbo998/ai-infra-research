# Enterprise Computer-Use & Browser GUI Agent (`CUA / Playwright / OmniParser`) Infrastructure Sizer (`#667`)

Reproducible analytical model quantifying the **Screenshot Visual Token Tax** of `15-step` `1080p` Computer-Use Agent workflows (`30,300` visual tokens/task requiring `28× H200` for `500` concurrent sessions), how **Hybrid Accessibility/DOM Tree + `OmniParser` Set-of-Marks (SoM) ROI Cropping + 2-Frame Sliding Window** slashes visual tokens by `78.4%` (`30,300 → 6,540` tokens/task, `1,420ms → 310ms` step `TTFT`, `28 → 6× H200`, `-78.6%` GPUs), and why **Containerized Chromium + `Xvfb` Snapshot Sandboxes (`1.8 GB/session`)** achieve `4.44×` higher host density than full Desktop VDI VMs (`8.0 GB/session`).

```bash
python experiments/enterprise_computer_use_browser_gui_sandbox_sizer/run.py
pytest tests/test_enterprise_computer_use_browser_gui_sandbox_sizer.py -q
```
