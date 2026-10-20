# Mooncake RDMA Auto-GID Recovery, LMCache In-Process GPU Connector Isolation, and NIXL UCX Phase Tracing

Deterministic Python 3.12 standard-library model verifying:
1. **Mooncake RDMA Auto-GID Data-Path Recovery (`kvcache-ai/Mooncake` `5ea6cf0d`)**: Epoch-stamped `RdmaCompletionToken` advances the GID candidate exactly once (`epoch 1 -> 2`) on `IBV_WC_RETRY_EXC_ERR` while ignoring `15/15` duplicate stale-epoch completions from the same burst, recovering `16/16` slices without blacklisting the NIC rail.
2. **LMCache In-Process GPU Connector Isolation (`LMCache/LMCache` `df19cd68`)**: Moving vLLM-coupled connectors into `lmcache.inprocess.v1.gpu_connector` and enforcing `_validate_vllm_device_features` upfront catches `3/3` unsupported accelerator/feature combinations (`ValueError`) before deep stream initialization crashes.
3. **NIXL UCX Sender/Receiver Phase Tracing (`ai-dynamo/nixl` `8d5660c1`)**: Emitting closed-vocabulary `nixl::submit`, `nixl::wire.submitted`, `nixl::notif.sent`, and `nixl::notif.received` phases while omitting poll-triggered `nixl::wire.completed` prevents `8.3191x` application polling skew (`195.5 us` vs `23.5 us`).
