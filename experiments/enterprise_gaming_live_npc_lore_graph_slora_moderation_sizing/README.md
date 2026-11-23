# Enterprise Live-Service Gaming NPC & Real-Time Moderation (`1M Concurrent Players`): `Cloud 70B per Utterance` vs. `Static Dialogue Trees` vs. `4ms DeBERTa Shield + Lore Graph + S-LoRA 8B FP8 (95ms P95) + Async 70B Director` (`#551`)

- **Chinese FAQ**: [`../../docs/components/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-faq.md`](../../docs/components/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-faq.en.md`](../../docs/components/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/gaming-live-npc-lore-graph-slora-moderation-lab.html`](../../docs/h5/gaming-live-npc-lore-graph-slora-moderation-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-sizer.html`](../../docs/reports/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-cards.html`](../../docs/reports/enterprise-gaming-live-npc-lore-graph-slora-moderation-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_gaming_live_npc_lore_graph_slora_moderation_sizing/run.py
python -m pytest tests/test_enterprise_gaming_live_npc_lore_graph_slora_moderation_sizing.py -q
```
