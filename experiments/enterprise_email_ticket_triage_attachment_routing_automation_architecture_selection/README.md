# Enterprise Shared-Mailbox & Ticket Triage Automation (`500K Emails/Mo`) Architecture Selection Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-email-ticket-triage-attachment-routing-automation-architecture-selection-faq.md`](../../docs/components/enterprise-email-ticket-triage-attachment-routing-automation-architecture-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-email-ticket-triage-attachment-routing-automation-architecture-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection/run.py
python experiments/enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection/run.py --check
python -m unittest tests/test_enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection.py
```
