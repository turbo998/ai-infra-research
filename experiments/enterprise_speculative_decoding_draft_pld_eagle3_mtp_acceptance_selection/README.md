# Enterprise Speculative Decoding Selection (`Standalone Draft` vs. `Prompt Lookup PLD` vs. `Medusa` vs. `EAGLE-3 / Native MTP`) (`#486`)

Reproducible analytical model comparing Standalone 7B Draft Models, Zero-Weight Prompt Lookup Decoding (`PLD / N-gram`), Medusa Parallel Heads, and `EAGLE-3 / Native MTP` with dynamic active-batch gating (`B <= 24–32`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_speculative_decoding_draft_pld_eagle3_mtp_acceptance_selection/run.py --check`
