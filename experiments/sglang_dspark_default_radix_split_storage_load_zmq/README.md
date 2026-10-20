# SGLang DSpark Simulated Acceptance Opt-In Default, RadixCache `UpdatePendingBackupOnNodeSplit`, Storage-Tier Waiting Prefill Load Deduction, and ZMQ ZAP Peer Allowlist

Deterministic Python 3.12 standard-library simulator (`own-offline`) verifying:
1. `sgl-project/sglang@f0b6590`: Disabling `SGLANG_SIMULATE_ACC_GREEDY` and `SGLANG_DSPARK_SIMULATE_FOLD` by default (`EnvBool(False)`).
2. `sgl-project/sglang@cc7ae37`: Routing pending-backup split bookkeeping through `UpdatePendingBackupOnNodeSplit` across both write-through (`ack_id != None`) and buffer-only (`ack_id is None`, including SWA leaf cap splits) paths with `0` orphaned buffer dependencies.
3. `sgl-project/sglang@0d7638b`: Deducting accepted storage-tier prefetch spans (`waiting_storage_span`) from `SchedulerLoadInquirer.get_num_waiting_uncached_tokens` (`12,544 -> 3,328` waiting uncached tokens, `-73.47%` false prefill load inflation) while ignoring unaccepted query results.
4. `sgl-project/sglang@194876b` & `468e26f`: Mitigating `CVE-2026-93034` via ZMQ ZAP `ThreadAuthenticator` peer IP normalization (`ipaddress.ip_address(p).compressed`, `zap_domain=b"sglang"`) and `SGLANG_BATCH_TOKENIZER_OUTPUTS` grouped vs per-request return sends.

## Run

```bash
python experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py --check
```
