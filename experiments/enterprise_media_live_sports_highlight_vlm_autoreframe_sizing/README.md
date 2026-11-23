# Enterprise Media & Live Sports Automated Highlight & 9:16 Smart Reframe (`500 Concurrent Streams`): `Manual Editor` vs. `Dense 72B VLM` vs. `2ms Audio/Motion Trigger (96% Filter) + 7B Video-VLM + YOLOv11 9:16 Auto-Reframe` (`#553`)

- **Chinese FAQ**: [`../../docs/components/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-faq.md`](../../docs/components/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-faq.en.md`](../../docs/components/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/media-live-sports-highlight-vlm-autoreframe-lab.html`](../../docs/h5/media-live-sports-highlight-vlm-autoreframe-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-sizer.html`](../../docs/reports/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-cards.html`](../../docs/reports/enterprise-media-live-sports-highlight-vlm-autoreframe-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_media_live_sports_highlight_vlm_autoreframe_sizing/run.py
python -m pytest tests/test_enterprise_media_live_sports_highlight_vlm_autoreframe_sizing.py -q
```
