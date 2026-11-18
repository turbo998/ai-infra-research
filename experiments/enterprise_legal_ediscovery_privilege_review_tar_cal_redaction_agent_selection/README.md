# Enterprise Legal E-Discovery, Privilege Review, `TAR 2.0 CAL` & Redaction Agent (`R445`)

Deterministic evaluation suite comparing four enterprise litigation/DOJ second-request e-discovery, attorney-client privilege review, Technology Assisted Review (`TAR 2.0 CAL`), and redaction scrubbing architectures across a `2.4-million-document` custodian collection (`42,000` privileged documents, `$95/hr` contract attorney rate):

1. **Tier 1**: Linear Manual Contract-Attorney Review + Boolean Keyword Search (`58.0%` recall, `3.4%` inadvertent privilege waiver escape, `2.8%` redaction text-layer leak, `0.0%` review volume reduction, `$1,150,000/mo` cost)
2. **Tier 2**: Uncalibrated Zero-Shot General LLM Document Classifier (`76.5%` recall, `6.8%` privilege waiver escape on non-lawyer forwards, `4.1%` redaction leak, `$740,000/mo` cost)
3. **Tier 3**: Legacy `TAR 1.0` Static Seed Set (`81.0%` recall, `2.1%` privilege waiver escape, `1.4%` redaction leak, `58.0%` volume reduction, `$495,000/mo` cost)
4. **Tier 4**: `TAR 2.0 Continuous Active Learning (CAL)` + Counsel-Graph & Family-Attachment Privilege Lock + Flattened Raster Redaction Scrubber + `FRCP Rule 26(b)(5)` Privilege Log Agent (`96.4%` recall, `0.0%` privilege waiver escape, `0.0%` redaction leak, `84.5%` review volume reduction, `$138,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection/run.py
python experiments/enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection/run.py --check
python -m unittest tests/test_enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection.py
```
