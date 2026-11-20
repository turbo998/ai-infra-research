# Enterprise AI Gateway Zero-Trust Authentication (`Workload ID OIDC + Local JWKS + SPIFFE mTLS + Dual-Slot Rotation`) (`R502`)

Deterministic quantitative evaluation model comparing four enterprise AI Gateway authentication and credential-rotation architectures across **85 connected services**:
- **Tier 1**: Static Non-Expiring Provider API Keys (`sk-...`) Scattered in Git Repos, `.env` Files & Notebooks
- **Tier 2**: Central Vault Static Keys with Manual 90-Day Single-Slot Rotation
- **Tier 3**: OAuth2 JWT Bearer Tokens Verified via Synchronous Remote JWKS / Introspection Endpoint on Every Call
- **Tier 4**: Secretless Workload Identity (`Entra Workload ID / AWS IAM OIDC` `15m` JWT) + Local JWKS Signature Verify (`0.3ms`) + `SPIFFE mTLS` + Dual-Slot (`Primary/Secondary`) Zero-Downtime Key Rotation

## Verification

```bash
python experiments/enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection/run.py --check
python -m unittest tests/test_enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection.py
```
