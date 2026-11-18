# Enterprise GenAI Marketing & Creative Supply Chain: Pixel-Locked Product Compositing, FTC/Ad-Law Gate & Contextual Bandit Evaluator

Deterministic evaluator comparing four enterprise GenAI marketing creative generation and compliance architectures across 180,000 monthly localized ad creatives, 25,000 CPG/retail SKUs, and 80M monthly personalized impressions:
- **Tier 1**: Manual Creative Agency Photoshoot/Retouching + Manual Legal Review (`94.0%` compliance pass rate, `0.8%` violation rate, `120.0 h` launch time, `$310,000/mo`)
- **Tier 2**: Unconstrained Text-to-Image Diffusion + Free-Form LLM Copy Direct Publishing (`58.5%` pass rate, `15.4%` FTC/Ad-Law or logo distortion violation rate, `0.8 h` launch time, `$395,000/mo`)
- **Tier 3**: Brand LoRA Fine-Tuned Diffusion + Static Keyword Blocklist (`79.2%` pass rate, `4.6%` violation rate, `6.5 h` launch time, `$135,000/mo`)
- **Tier 4**: Pixel-Locked SKU Foreground + `IC-Light` Relighting + Legal Claim-Substantiation Matrix + `C2PA` Signing + Fatigue-Capped Contextual Bandit Agent (`99.2%` pass rate, `0.0%` FTC/logo violation rate, `0.6 h` launch time, `$42,000/mo`)

## Usage
```bash
python experiments/enterprise_marketing_creative_brand_compliance_adtech_agent_selection/run.py
python experiments/enterprise_marketing_creative_brand_compliance_adtech_agent_selection/run.py --check
```
