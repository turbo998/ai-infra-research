# Enterprise Multimodal Contact Center QA & 100% Call Compliance Audit Architecture Evaluator 

Deterministic evaluator comparing four contact center quality assurance (`QA`) and regulatory compliance audit architectures across 1,000,000 monthly calls (5,000,000 audio minutes/month):
- **Tier 1**: Legacy `2%` Manual QA Sampling + Basic Keyword Spotting (`14.2%` violation recall, `72.0%` speaker attribution, `72.0 h` turnaround, `$42,000/mo`)
- **Tier 2**: `100%` Raw Audio Direct to Frontier Multimodal Audio-LLM (`86.5%` violation recall, `88.4%` speaker attribution, `0.25 h` turnaround, `$112,000/mo`)
- **Tier 3**: Stereo Dual-Channel `Whisper-Turbo` ASR + `100%` `70B` LLM Full-Rubric Scoring (`93.8%` violation recall, `99.2%` speaker attribution, `0.50 h` turnaround, `$28,500/mo`)
- **Tier 4**: Stereo Dual-Channel `Whisper-Turbo` + Acoustic Prosody/Silence Detector + `8B` Binary Filter + `15%` High-Risk `70B` Escalation (`96.4%` violation recall, `99.6%` speaker attribution, `0.08 h [5 min]` turnaround, `$5,400/mo`)

## Usage
```bash
python experiments/enterprise_contact_center_qa_compliance_selection/run.py
python experiments/enterprise_contact_center_qa_compliance_selection/run.py --check
```
