# Enterprise Media, OTT Streaming & Global Gaming LiveOps Localization, Lip-Sync Dubbing & Compliance Agent (`R433`)

Deterministic evaluation suite comparing four global media/gaming localization, dubbing, age-rating compliance, and LiveOps retention architectures across a 5M DAU title shipping biweekly content patches across 18 languages (`250,000` words + `180` dubbed audio minutes per patch):

1. **Tier 1**: Manual Localization Studios + Rule-Based LiveOps Promos (`6.2%` lore violation, `4.5%` dubbing desync, `3.8%` rating violation escape, `1,008h` / 6-week patch lag, `$460,000/mo` TCO)
2. **Tier 2**: Raw Machine Translation + Unconstrained Zero-Shot TTS (`27.4%` lore violation, `42.0%` dubbing/UI overflow desync, `14.6%` rating escape, `6.0h` lag, `$385,000/mo` TCO)
3. **Tier 3**: Glossary RAG Translation + Propensity Churn Scoring (`5.8%` lore violation, `19.5%` dubbing desync, `4.2%` rating escape, `18.0h` lag, `$210,000/mo` TCO)
4. **Tier 4**: Glossary-Locked Isochronous Dubbing (`MuseTalk/Wav2Lip + FreeType Pixel Gate`) + Regional Age-Rating `VLM/AST` Gate (`ESRB/PEGI/GAPP`) + Causal Uplift (`CATE`) LiveOps Agent (`0.0%` lore violation, `0.4%` dubbing desync, `0.0%` rating escape, `2.5h` Sim-Ship lag, `$52,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_media_gaming_liveops_localization_compliance_agent_selection/run.py
python experiments/enterprise_media_gaming_liveops_localization_compliance_agent_selection/run.py --check
python -m unittest tests/test_enterprise_media_gaming_liveops_localization_compliance_agent_selection.py
```
