# Dynamo Lock-Free KV State Agent Cursor & 1000-Slot ZMQ Ingest, NIXL Multi-QP WRITE Striping, Mooncake Ascend `mmap`/`memfd` & TorchTitan Compressed-Latent MLA CP

Deterministic verification script for:
- `ai-dynamo/dynamo@93ff081d4e` (PR `#16935`), `9906dbe5ee` (PR `#16568`), & `5235cfc93d` (PR `#16818`)
- `ai-dynamo/nixl@bb71164c4b` (PR `#2215`)
- `kvcache-ai/Mooncake@4137ac0026` (PR `#4756`) & `7a05855674` (PR `#4410`)
- `pytorch/torchtitan@0980e1cb1c` (PR `#4760`) & `9698a73362` (PR `#5394`)

## Run
```bash
python experiments/dynamo_kv_cursor_1000slot_nixl_multiqp_torchtitan_mla_cp/run.py --check
```
