# Enterprise Semiconductor EDA RTL & Formal SVA Verification Copilot: `Zero-Shot 70B` vs. `Manual DV` vs. `FSM Graph + 32B/70B RTL/SVA Generator + Formal Counterexample Loop` (`#554`)

- **Chinese FAQ**: [`../../docs/components/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-faq.md`](../../docs/components/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-faq.en.md`](../../docs/components/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/semiconductor-eda-rtl-sva-formal-verification-loop-lab.html`](../../docs/h5/semiconductor-eda-rtl-sva-formal-verification-loop-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-sizer.html`](../../docs/reports/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-cards.html`](../../docs/reports/enterprise-semiconductor-eda-rtl-sva-formal-verification-loop-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_semiconductor_eda_rtl_sva_formal_verification_loop_sizing/run.py
python -m pytest tests/test_enterprise_semiconductor_eda_rtl_sva_formal_verification_loop_sizing.py -q
```
