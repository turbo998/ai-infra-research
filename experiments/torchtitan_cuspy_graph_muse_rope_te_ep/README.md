# TorchTitan Cuspy CUPTI Graph Annotations, Muse Spark QK-RoPE Region & TE JAX Fused NVL EP (`Phase 32`)

Deterministic Python 3.12 standard-library simulator verifying:
1. `pytorch/torchtitan` (`b93d5c0cf518b4404ceab998adc529fd60ebf301`): `get_cuspy_cuda_graph_annotation_config` (`backend="cupti"`, `key_by="exec"` when `enable_event_node_ids=True`), recovering `24/24` graph nodes (`+4` pre-capture stream nodes and `4/4` NCCL event-record nodes) and `(graph_id << 32) | graph_node_id` Python stack keys.
2. `pytorch/torchtitan` (`8f8dbfb50caacdaa2e1af3ebb1d1ee67f57b9a87`): Unified `muse_spark_attention_qk` region (`recompile_limit=16`) replacing Helion RoPE override (`3.75x` decode speedup at 256 seqs: `39.0 -> 10.4 us`, `0.0` numerical drift).
3. `NVIDIA/TransformerEngine` (`b7db08fd73129f5af71ff2ee0eaddacd3ff1a751`): `ep_prepare_and_dispatch_fwd` (`NVTE_EP_FUSED_PREPARE_DISPATCH=1`) cutting XLA FFI launches by `50%` (`2 -> 1`) with aliased `handle_mem` output and zero combine error under intra-expert row permutation.

## Run

```bash
python experiments/torchtitan_cuspy_graph_muse_rope_te_ep/run.py --check
```
